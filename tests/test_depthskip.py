"""Exactness of the skip paths on a tiny random Qwen3.5: with maps that predict the true inputs (upper layers made
pass-through), every skipped form must write the same cache as the exact pass."""
import types

import pytest
import torch

from conftest import seqs, tiny_model
from depthskip import (DepthPlan, MLPSurrogates, PartSkip, SkipMaps, cache_tensors, depth_cache, fit_mlp_surrogates,
                     fit_skip_maps, seg_lengths, segment_cache, skipped_share)
from depthskip.maps import ema_scan


def _passthrough_above(model, cut):
    """Layers >= cut add nothing to the residual stream (output projections and MLP zeroed): every upper layer's input
    equals the cut's input, so the identity map predicts it exactly."""
    for layer in model.lm.layers[cut:]:
        mixer = layer.linear_attn.out_proj if hasattr(layer, "linear_attn") else layer.self_attn.o_proj
        mixer.weight.data.zero_()
        layer.mlp.down_proj.weight.data.zero_()


def _cache_tensors(cache):
    return {k: v.float() for k, v in cache_tensors(cache).items()}


def _batch(model, ss):
    """LEFT-padded batch (every row ends at its last real token) through model.prefill (skip_state applies)."""
    T = max(map(len, ss))
    ids = torch.tensor([[0] * (T - len(s)) + s for s in ss])
    mask = torch.arange(T)[None] >= (T - torch.tensor([len(s) for s in ss]))[:, None]
    _, cache = model.prefill(ids, mask)
    return _cache_tensors(cache)


def _same(a, b, atol=1e-5):
    assert set(a) == set(b)
    for k in a:
        assert torch.allclose(a[k], b[k], atol=atol), k


@pytest.mark.parametrize("cut", [3, 6])                     # layer 3 is attention, layer 6 a DeltaNet layer
@pytest.mark.parametrize("chunk", [0, 7])                   # whole pass, and continuation chunks (incl. 1 token)
def test_cache_only_layers_match_full_layers(cut, chunk):
    model = tiny_model(prefill_chunk=chunk)
    _passthrough_above(model, cut)
    ss = seqs(2, [41, 9])
    exact = _batch(model, ss)
    D, U = model.hidden_size, len(model.lm.layers) - cut
    model.skip_state = SkipMaps(cut, "input", torch.eye(D).repeat(U, 1, 1), torch.zeros(U, D))
    skip = _batch(model, ss)
    _same(exact, skip)
    assert all(type(l).__name__ != "CacheOnlyLayer" for l in model.lm.layers)    # layers restored


def test_fit_recovers_passthrough_and_share():
    model = tiny_model()
    _passthrough_above(model, 4)
    texts = seqs(12, seed=3)
    for mode in ("input", "normed"):
        maps = fit_skip_maps(model, texts[:10], texts[10:], 4, mode, lams=(1e-6,), log=lambda *_: None)
        assert maps.W.shape == (4, model.hidden_size, model.hidden_size)
        assert max(maps.meta["held_rel_err"].values()) < 0.05, maps.meta
    s = skipped_share(model.lm.config, 4)
    assert 0 < s["skipped"] < 1


def test_generic_mixer_path_matches():
    """Mixers with a swapped forward (as ported ones) run whole on the predicted input: same cache."""
    model = tiny_model()
    _passthrough_above(model, 3)
    for layer in model.lm.layers[3:]:
        m = layer.linear_attn if hasattr(layer, "linear_attn") else layer.self_attn
        m.forward = types.MethodType(type(m).forward, m)
    ss = seqs(2, [41, 9])
    exact = _batch(model, ss)
    D, U = model.hidden_size, len(model.lm.layers) - 3
    model.skip_state = SkipMaps(3, "input", torch.eye(D).repeat(U, 1, 1), torch.zeros(U, D))
    _same(exact, _batch(model, ss))


def test_maps_save_load(tmp_path):
    D, U = 8, 3
    m = SkipMaps(5, "normed", torch.randn(U, D, D).bfloat16(), torch.randn(U, D).bfloat16(), 0.01, (0.9,))
    m.meta = {"x": 1}
    m.save(tmp_path / "m.pt")
    r = SkipMaps.load(tmp_path / "m.pt", "cpu", tail=7)
    assert (r.cut, r.mode, r.lam, r.alphas, r.tail, r.meta) == (5, "normed", 0.01, (0.9,), 7, {"x": 1})
    assert torch.equal(r.W, m.W) and torch.equal(r.b, m.b)


def test_ema_scan_matches_loop():
    x = torch.randn(2, 300, 4)
    y, c = ema_scan(x, 0.9, block=64)
    ref, prev = torch.zeros_like(x), torch.zeros(2, 4)
    for t in range(300):
        prev = 0.9 * prev + 0.1 * x[:, t]
        ref[:, t] = prev
    assert torch.allclose(y, ref, atol=1e-5) and torch.allclose(c, ref[:, -1], atol=1e-5)
    y2, _ = ema_scan(x[:, 100:], 0.9, ref[:, 99], block=64)          # carry continues a split sequence
    assert torch.allclose(y2, ref[:, 100:], atol=1e-5)


@pytest.mark.parametrize("chunk", [0, 7])
def test_tail_and_context_features_keep_exactness(chunk):
    """Identity on the cut's input, zero weight on its EMAs, last 5 tokens through every layer: still exact."""
    model = tiny_model(prefill_chunk=chunk)
    _passthrough_above(model, 3)
    ss = seqs(2, [41, 9])
    exact = _batch(model, ss)
    D, U = model.hidden_size, len(model.lm.layers) - 3
    W = torch.cat([torch.eye(D), torch.zeros(2 * D, D)]).repeat(U, 1, 1)
    model.skip_state = SkipMaps(3, "input", W, torch.zeros(U, D), alphas=(0.5, 0.9), tail=5)
    _same(exact, _batch(model, ss))


def test_fit_with_context_features():
    model = tiny_model()
    _passthrough_above(model, 4)
    texts = seqs(12, seed=3)
    maps = fit_skip_maps(model, texts[:10], texts[10:], 4, "normed", lams=(1e-6,), log=lambda *_: None, alphas=(0.9,))
    assert maps.W.shape == (4, 2 * model.hidden_size, model.hidden_size) and maps.alphas == (0.9,)
    assert max(maps.meta["held_rel_err"].values()) < 0.05, maps.meta


@pytest.mark.parametrize("tail", [0, 4])
def test_mlp_surrogates_exact_when_mlp_is_zero(tail):
    """Upper MLPs that output zero (down_proj zeroed): the zero surrogate is exact, mixers run as before."""
    model = tiny_model()
    for layer in model.lm.layers[3:]:
        layer.mlp.down_proj.weight.data.zero_()
    ss = seqs(2, [41, 9])
    exact = _batch(model, ss)
    D, U = model.hidden_size, len(model.lm.layers) - 3
    model.skip_state = MLPSurrogates(3, torch.zeros(U, D, D), torch.zeros(U, D), tail=tail)
    _same(exact, _batch(model, ss))
    assert all(type(l.mlp).__name__ != "_LinearMLP" for l in model.lm.layers)
    model.skip_state = None
    texts = seqs(8, seed=3)
    sur = fit_mlp_surrogates(model, texts[:6], texts[6:], 3, lams=(1e-6,), log=lambda *_: None)
    assert sur.W.shape == (U, D, D) and float(sur.W.float().abs().max()) < 1e-3


def test_seg_lengths_head_and_tail_windows():
    m = object()
    assert seg_lengths(1000, [(-1, m), (480, m), (32, None)]) == [488, 480, 32]
    assert seg_lengths(1000, [(64, None), (-1, m), (480, m), (32, None)]) == [64, 424, 480, 32]
    assert seg_lengths(300, [(64, None), (-1, m), (480, m), (32, None)]) == [0, 0, 268, 32]   # tail windows first
    assert seg_lengths(100, [(64, None), (-1, m), (32, None)]) == [64, 4, 32]
    assert sum(seg_lengths(777, [(4, None), (-1, m), (480, m), (32, None)])) == 777


def test_segment_cache_matches_per_token_depth():
    """Recency plans: the fast segment form writes the cache of the per-token-depth reference (depth_cache)."""
    model = tiny_model()
    D, L = model.hidden_size, len(model.lm.layers)
    g = torch.Generator().manual_seed(1)
    maps = {c: SkipMaps(c, "normed", torch.eye(D).repeat(L - c, 1, 1) + 0.05 * torch.randn(L - c, D, D, generator=g),
                        0.01 * torch.randn(L - c, D, generator=g)) for c in (4, 6)}
    ids = seqs(1, [21])[0]
    ref = _cache_tensors(depth_cache(model, ids, DepthPlan(maps, L, [(10, 6), (5, None)])))
    seg = _cache_tensors(segment_cache(model, ids, [(-1, maps[4]), (5, maps[6]), (5, None)]))
    exact = _cache_tensors(segment_cache(model, ids, [(-1, None)]))
    assert set(ref) == set(seg) and any(k[0] == 7 for k in ref)
    for k in ref:
        assert torch.allclose(ref[k], seg[k], atol=1e-4), k
    assert torch.allclose(seg[(3, "keys")], exact[(3, "keys")], atol=1e-5)   # below the cut: exact
    assert not torch.allclose(seg[(7, "keys")], exact[(7, "keys")], atol=1e-3)   # above: predicted inputs


def test_first_n_window_runs_every_layer():
    """A first-N window: those tokens' upper caches are exact (they ran every layer, before anything was skipped)."""
    model = tiny_model()
    D, L = model.hidden_size, len(model.lm.layers)
    g = torch.Generator().manual_seed(2)
    m4 = SkipMaps(4, "normed", torch.eye(D).repeat(L - 4, 1, 1) + 0.05 * torch.randn(L - 4, D, D, generator=g),
                  torch.zeros(L - 4, D))
    ids = seqs(1, [30])[0]
    seg = _cache_tensors(segment_cache(model, ids, [(6, None), (-1, m4), (4, None)]))
    exact = _cache_tensors(segment_cache(model, ids, [(-1, None)]))
    assert torch.allclose(seg[(7, "keys")][:, :, :6], exact[(7, "keys")][:, :, :6], atol=1e-5)
    assert not torch.allclose(seg[(7, "keys")][:, :, 6:26], exact[(7, "keys")][:, :, 6:26], atol=1e-3)


@pytest.mark.parametrize("mlp,attn", [(True, False), (False, True), (True, True)])
def test_part_skip_exact_when_parts_are_zero(mlp, attn):
    """Upper MLPs and attention outputs zeroed: zero surrogates are exact, and the skipped attention still writes its
    cache from the true input, so every cache tensor equals the exact pass's."""
    model = tiny_model()
    _passthrough_above(model, 4)
    D, L = model.hidden_size, len(model.lm.layers)
    ps = PartSkip(4, 4, (torch.zeros(L - 4, D, D), torch.zeros(L - 4, D)) if mlp else None,
                  {7: (torch.zeros(D, D), torch.zeros(D))} if attn else None)
    ids = seqs(1, [21])[0]
    exact = _cache_tensors(segment_cache(model, ids, [(-1, None)]))
    _same(exact, _cache_tensors(segment_cache(model, ids, [(-1, ps)])))
    assert all(type(l.mlp).__name__ != "_LinearMLP" for l in model.lm.layers)
    assert all(type(getattr(l, "self_attn", None)).__name__ != "_AttnSurrogate" for l in model.lm.layers)


def test_fit_parts_recovers_zero_parts():
    from depthskip import fit_parts, part_skip
    model = tiny_model()
    _passthrough_above(model, 4)
    texts = seqs(8, seed=3)
    fit = fit_parts(model, texts[:6], texts[6:], 4, lams=(1e-6,), log=lambda *_: None)
    ps = part_skip(fit, 4, mlp=True, attn=True, device="cpu")
    assert ps.cut == 4 and set(ps.attn) == {7}
    assert float(fit["mlp_W"].float().abs().max()) < 1e-3
