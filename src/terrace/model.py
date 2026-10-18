"""A thin wrapper over a Hugging Face causal LM (Qwen3.5 text models) with what Terrace needs: tokenization, a chunked
prompt pass that returns the HF cache (optionally with a skip installed), and a prefix pass the fitters drive.

Any object with the same attributes works in its place (duck typing): lm (the decoder: embed_tokens, layers, norm),
tokenize(texts) -> list of id lists, device, hidden_size, cfg.prefill_chunk, skip_state, prefix_cache_batch(seqs) and
_prefill(ids, mask, layers, cache, keep_hidden, C, skip=None, pmask0=None) -> (features or None, cache).
OpenDecider's Backbone is one such object.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

import torch
from torch import nn

from .maps import skipping


class Model:
    def __init__(self, lm: nn.Module, tokenizer=None, lm_head: nn.Module | None = None, prefill_chunk: int = 0):
        self.lm = lm.eval().requires_grad_(False)
        self.tok = tokenizer
        self.lm_head = lm_head
        self.cfg = SimpleNamespace(prefill_chunk=prefill_chunk)
        self.hidden_size = lm.config.hidden_size
        self.layer_types = list(getattr(lm.config, "layer_types", ["full_attention"] * len(lm.layers)))
        self.skip_state = None              # SkipMaps / MLPSurrogates applied by prefill() (keep_hidden=False)
        pad = getattr(tokenizer, "pad_token_id", None)
        self.pad_id = pad if pad is not None else 0

    @classmethod
    def from_pretrained(cls, path: str, dtype: str | torch.dtype = "bfloat16", device: str = "cuda",
                        prefill_chunk: int = 4096, **kw) -> "Model":
        """Text-only causal LM from a local path or hub id (Qwen3.5 checkpoints load as Qwen3_5ForCausalLM)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dt = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        tok = AutoTokenizer.from_pretrained(path)
        full = AutoModelForCausalLM.from_pretrained(path, dtype=dt, device_map=device, **kw)
        return cls.wrap(full, tok, prefill_chunk)

    @classmethod
    def wrap(cls, causal_lm: nn.Module, tokenizer=None, prefill_chunk: int = 0) -> "Model":
        """From an already loaded *ForCausalLM (its .model is the decoder, .lm_head the head)."""
        return cls(causal_lm.model, tokenizer, causal_lm.lm_head, prefill_chunk)

    @property
    def device(self) -> torch.device:
        return self.lm.embed_tokens.weight.device

    # ------------------------------------------------------------------ tokens
    def tokenize(self, texts: Sequence) -> list[list[int]]:
        """Strings through the tokenizer (no special tokens); id lists pass through unchanged."""
        out, todo = [None] * len(texts), []
        for i, t in enumerate(texts):
            if isinstance(t, str):
                todo.append(i)
            else:
                out[i] = [int(x) for x in t]
        if todo:
            ids = self.tok([texts[i] for i in todo], add_special_tokens=False)["input_ids"]
            for i, x in zip(todo, ids):
                out[i] = x
        return out

    # ------------------------------------------------------------------ passes
    def forward(self, ids, mask, layers=("final",), past_key_values=None, use_cache=True, past_mask=None,
                position_ids=None):
        """One decoder call. Returns (last hidden state, cache)."""
        attn = mask.long()
        if past_key_values is not None:
            if past_mask is None:
                past = past_key_values.get_seq_length()
                past_mask = torch.ones(mask.shape[0], past, dtype=torch.bool, device=mask.device)
            attn = torch.cat([past_mask.long(), attn], 1)
        out = self.lm(input_ids=ids, attention_mask=attn, past_key_values=past_key_values, use_cache=use_cache,
                      position_ids=position_ids)
        return out.last_hidden_state, (out.past_key_values if use_cache else None)

    def _prefill(self, ids, mask, layers, cache, keep_hidden, C, skip=None, pmask0=None):
        """Prompt pass over (B, T) in chunks of C tokens, each continuing the cache (same result as one pass up to
        float error). skip: the installed SkipMaps (gets each chunk's padding mask for its EMA features)."""
        feats = []
        past = cache.get_seq_length() if cache is not None else 0
        pmask = pmask0 if pmask0 is not None else (
            torch.ones(ids.shape[0], past, dtype=torch.bool, device=mask.device) if past else None)
        for a in range(0, ids.shape[1], C):
            b = min(a + C, ids.shape[1])
            if skip is not None:
                skip._mask = mask[:, a:b]
            f, cache = self.forward(ids[:, a:b], mask[:, a:b], layers, past_key_values=cache, use_cache=True,
                                    past_mask=pmask)
            pmask = mask[:, :b] if pmask is None else torch.cat([pmask, mask[:, a:b]], 1)
            if keep_hidden:
                feats.append(f)
        return (torch.cat(feats, -2) if keep_hidden else None), cache

    @torch.no_grad()
    def prefill(self, ids, mask=None, layers=("final",), cache=None, keep_hidden: bool = False):
        """Prompt pass; with self.skip_state set (and keep_hidden False) the upper layers only write their caches, and
        the map's last `tail` tokens then run every layer on top. ids (B, T) or (T,); `layers` is accepted for
        signature compatibility (only the final hidden state is returned). Returns (features or None, cache)."""
        ids = ids if ids.dim() == 2 else ids[None]
        ids = ids.to(self.device)
        mask = torch.ones_like(ids, dtype=torch.bool) if mask is None else mask.to(self.device)
        C = self.cfg.prefill_chunk or ids.shape[1]
        sk = self.skip_state
        if sk is None or keep_hidden:
            return self._prefill(ids, mask, ("final",), cache, keep_hidden, C)
        past0 = cache.get_seq_length() if cache is not None else 0
        R = min(getattr(sk, "tail", 0), ids.shape[1] - 1)
        head = ids.shape[1] - R
        with skipping(self.lm, sk):
            _, cache = self._prefill(ids[:, :head], mask[:, :head], ("final",), cache, False, C, sk)
        if R:
            pm = mask[:, :head]
            if past0:
                pm = torch.cat([torch.ones(ids.shape[0], past0, dtype=torch.bool, device=mask.device), pm], 1)
            _, cache = self._prefill(ids[:, head:], mask[:, head:], ("final",), cache, False, C, pmask0=pm)
        return None, cache

    @torch.no_grad()
    def prefix_cache_batch(self, seqs: Sequence[Sequence[int]], return_hidden: bool = False, quantize=None):
        """Pass over sequences LEFT-padded to one length (every row ends at its last real token), through prefill():
        exact unless skip_state is set (the fitters clear it; their hooks read the layers' inputs). quantize is
        accepted for signature compatibility and ignored (the HF cache is returned as is).
        Returns (cache, mask) or (cache, mask, last hidden state)."""
        lens = [len(x) for x in seqs]
        T = max(lens)
        ids = torch.tensor([[self.pad_id] * (T - len(x)) + list(x) for x in seqs], dtype=torch.long,
                           device=self.device)
        mask = torch.arange(T, device=self.device)[None] >= (T - torch.tensor(lens, device=self.device))[:, None]
        f, cache = self.prefill(ids, mask, ("final",), None, keep_hidden=return_hidden)
        return (cache, mask, f) if return_hidden else (cache, mask)

    def head_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Final-norm hidden states -> vocabulary logits (float32)."""
        if self.lm_head is None:                    # tied head
            return (h @ self.lm.embed_tokens.weight.T).float()
        return self.lm_head(h).float()

    logits = head_logits


def cache_tensors(cache) -> dict:
    """Every tensor a cache holds, keyed (layer, field[, sub-key]): an HF DynamicCache (attention keys / values, DeltaNet
    conv / recurrent states, any custom layer's tensor fields), or a packed cache exposing .tensors as is."""
    if hasattr(cache, "tensors"):
        return dict(cache.tensors)
    out = {}
    for i, layer in enumerate(cache.layers):
        for k, v in vars(layer).items():
            if torch.is_tensor(v):
                out[(i, k)] = v
            elif isinstance(v, dict):
                out.update({(i, k, kk): t for kk, t in v.items() if torch.is_tensor(t)})
    return out


_KERNEL_FUNCS = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn",
                 "causal_conv1d_update")
_saved_kernels: dict = {}


def use_reference_kernels(enable: bool = True) -> None:
    """Swap Qwen3.5's fla / causal-conv1d kernels for transformers' pure-torch references (CPU, tests); False restores."""
    import transformers.models.qwen3_5.modeling_qwen3_5 as mq
    for name in _KERNEL_FUNCS:
        if enable:
            ref = _saved_kernels.setdefault(name, getattr(mq, name))
            while hasattr(ref, "__wrapped__"):
                ref = ref.__wrapped__
            setattr(mq, name, ref)
        elif name in _saved_kernels:
            setattr(mq, name, _saved_kernels[name])
