# Terrace

Training-free layer skipping for the **prompt pass** of hybrid DeltaNet + attention language models (Qwen3.5 family,
and Mercurius's MLA / GDN-2 variants by duck typing).

Prompt tokens run the lower layers as usual and stop at a **cut**. Every layer above the cut still writes its cache for
them (K/V in the attention layers, conv and recurrent state in the DeltaNet layers), from a **closed-form ridge map** of
the cut's input, and skips everything else (queries, attention, output projections, MLP). Tokens after the prompt run
every layer and read that cache, so decoding is unchanged and only the prompt pass gets cheaper. On top of that:

- **graded depth**: depth per token, by position windows in one pass. The most recent tokens run every layer, the
  ones before them are cut late, the rest early; optionally the first N tokens run every layer too (attention sinks);
- **part skips**: above the cut, keep the mixers and replace only the MLPs and/or the attention outputs by ridge maps.
  An attention-skipped layer still writes its cache from its true input, so its cache stays exact.

Nothing is trained. The maps are fitted from one pass of the frozen model over calibration text: 200 documents (48k
tokens) take 2 minutes for Mercurius-1-4B at cut 24 and 5 minutes for the 9B at cut 16 on the Orin.

The name: the prompt is cut into terraces of different depth, deep near the end and shallow further back.

## Install

```bash
pip install -e .              # torch, transformers>=5.2, numpy
pip install -e ".[experiments,test]"
python -m pytest -q           # CPU, tiny random Qwen3.5 (20 tests)
```

## Use

```python
import torch
from terrace import Model, SkipMaps, fit_skip_maps, segment_prefill

model = Model.from_pretrained("Qwen/Qwen3.5-9B", dtype="bfloat16", device="cuda", prefill_chunk=4096)

# fit (or load) one ridge map set per cut: 2-5 minutes on 200 calibration documents of 256 tokens (Orin)
calib, held = texts[:200], texts[200:210]          # strings or token-id lists
m16 = fit_skip_maps(model, calib, held, cut=16, mode="normed")
m24 = fit_skip_maps(model, calib, held, cut=24, mode="normed")
m16.save("maps_cut16.pt")                          # SkipMaps.load(path, device) later

# graded-depth prompt pass (g512): last 32 tokens all layers, previous 480 cut 24, the rest cut 16
ids = model.tokenize([prompt])[0]
cache = segment_prefill(model, ids, [(-1, m16), (480, m24), (32, None)])
# first-N variant: [(64, None), (-1, m16), (480, m24), (32, None)]

# `cache` is a transformers DynamicCache: decode as usual, every new token runs all layers
out = model.lm(input_ids=torch.tensor([[next_id]], device=model.device), past_key_values=cache, use_cache=True)
```

A single cut for the whole prompt: `model.skip_state = SkipMaps.load(path, "cuda", tail=32)`, then
`model.prefill(ids)`. With `tail=R`, the last R prompt tokens run every layer in a second call.

### API

| Name | What it does |
|---|---|
| `Model.from_pretrained(path, dtype="bfloat16", device="cuda", prefill_chunk=4096)`, `Model.wrap(causal_lm, tokenizer)` | Wrapper over a HF causal LM: `tokenize`, `prefill(ids, mask=None, layers, cache=None, keep_hidden=False) -> (feats, cache)`, `forward`, `head_logits`, `prefix_cache_batch` |
| `SkipMaps(cut, mode, W, b, lam, alphas, tail)`, `.save(path)`, `SkipMaps.load(path, device, tail=0)` | One ridge map per layer from the cut up (`mode`: `normed` or `input`) |
| `fit_skip_maps(model, texts, held_texts, cut, mode="normed", lams=LAMS, log=print, alphas=())` | Closed-form fit; ridge strength by held-out error; `alphas`: causal-EMA context features |
| `skipping(lm, maps)` | Context manager: layers from the cut up become cache-only (or MLPs become linear, for `MLPSurrogates`) |
| `fit_mlp_surrogates(model, texts, held_texts, cut)`, `MLPSurrogates` | MLP-only skip: mixers run, MLPs become linear maps |
| `fit_parts(model, texts, held_texts, base)`, `part_skip(fit, cut, mlp, attn)`, `PartSkip` | Part skips: MLP and/or attention-output surrogates, attention caches written from true inputs |
| `seg_lengths(T, segs)`, `segment_cache(model, ids, segs)` (= `segment_prefill`) | Graded depth: `segs = [(n or -1, SkipMaps / PartSkip / None), ...]` in prompt order; windows after the `-1` segment fill from the end, windows before it from the start |
| `DepthPlan(maps, n_layers, windows, select)`, `depth_cache(model, ids, plan)` | Per-token-depth reference: same cache as `segment_cache` for recency plans; `select` adds tokens chosen by an attention error score |
| `skipped_share(config, cut, whole_mixer=False)`, `part_share(...)`, `plan_skipped(...)` | Share of the prompt pass's weight work that is skipped |
| `cache_tensors(cache)`, `use_reference_kernels(True)` | Tensors in a cache (for checks); pure-torch DeltaNet / conv kernels (CPU) |

Anything with the same attributes as `Model` works in its place (`lm`, `tokenize`, `device`, `hidden_size`,
`cfg.prefill_chunk`, `skip_state`, `prefix_cache_batch`, `_prefill`). OpenDecider's `Backbone` is one; its decision
model uses Terrace for its state pass.

## Results

All measured on a Jetson AGX Orin (MODE_50W). Models:
- **Mercurius-1-4B**: 32 layers, MLA + GDN-2, 4-bit weights and cache;
- **the 9B**: Qwen3.5-9B, 32 layers, full attention at layers 3, 7, ..., 31.

Both run as the backbones of OpenDecider checkpoints (`--ckpt`). Maps are normed, fitted on 200 FineWeb-Edu documents
(256 tokens each). Raw outputs are in `results/`.

### Quality: generation and multiple choice

Generation quality: 30-60 FineWeb-Edu documents, a 1,024-token prompt, the next 128 tokens teacher-forced:
- **KL**: KL(exact ‖ skipped) of the next-token distribution;
- **loss Δ**: change in continuation NLL;
- **greedy identical**: share of 32-token greedy continuations identical to exact.

Multiple choice: OpenBookQA / CommonsenseQA (lettered, chat template), 300 items (4B) and 200 items (9B).

| arm | 4B KL | 4B loss Δ | 4B OBQA / CSQA | 9B KL | 9B OBQA / CSQA |
|---|---|---|---|---|---|
| exact | 0 | 0 | 0.797 / 0.723 | 0 | 0.885 / 0.815 |
| model's own rounding floor (prompt in other slices) | 0.0040 | 0.000 | 0.803 / 0.713 | – | – |
| cut 24 | 0.0084 | +0.008 | 0.813 / 0.773 | 0.0058 | 0.900 / 0.820 |
| cut 16 | 0.085 | +0.082 | 0.403 / 0.283 | 0.094 | 0.315 / 0.200 |
| cut 16 + exact tail 32 (second call) | 0.068 | +0.075 | 0.710 / 0.630 | 0.075 | 0.835 / 0.775 |
| **g512** (last 32 all layers, 480 cut 24, rest cut 16) | 0.023 | +0.027 | 0.800 / 0.717 | 0.024 | 0.890 / 0.815 |
| g256 | 0.036 | +0.038 | 0.800 / 0.717 | 0.039 | 0.890 / 0.815 |
| attention-only skip from 16 | 0.0047 | +0.003 | 0.807 / 0.743 | 0.0025 | 0.840 / 0.780 |
| attention-only 16 + tail 32 | 0.0050 | +0.004 | 0.790 / 0.703 | 0.0021 | 0.890 / 0.815 |
| MLP-only skip from 16 | 0.050 | +0.051 | 0.437 / 0.257 | 0.046 | 0.795 / 0.755 |
| MLP + attention from 16 | 0.053 | +0.056 | 0.450 / 0.257 | 0.052 | 0.765 / 0.695 |
| MLP-only 24 | 0.0068 | +0.007 | 0.793 / 0.740 | 0.0045 | 0.890 / 0.815 |
| attention-only 24 | 0.0015 | +0.000 | 0.810 / 0.740 | 0.0007 | 0.890 / 0.815 |

**Cut 24 and attention-only skips**
- Cut 24 is close to the model's own rounding noise (KL 2x the floor, the same top-1 agreement: 96.3%) and loses no
  task accuracy.
- Attention-only skips are within cut 24's error at either cut.

**Cut 16 and how to fix it**
- Cut 16 alone collapses on lettered multiple choice: it picks early letters.
- The cause is the most recent prompt tokens, so depth by recency fixes it. **g512** keeps both models at exact
  accuracy and cuts cut 16's KL by about 4x, in one pass.
- Choosing extra deep tokens by an attention error score is no better than the same number of recent tokens.

**Which caches carry the error**
- The few full-attention layers carry most of it. Taking the attention caches from an exact pass cuts cut 16's KL
  from 0.085 to 0.016, while the DeltaNet caches cut it only to 0.048 (4B).
- This holds for stock attention (9B) and for MLA (Mercurius) alike.
- Temperature and output-side calibration do not remove it.

**Other tasks (Mercurius-1-4B, 300 items each)**
- PubMedQA (yes/no over an abstract) is robust: exact 0.830, cut 24 0.830, cut 16 0.820.
- Cut 24's extra answer changes on CommonsenseQA (12.7% against a 4.3% floor) are miscalibration: option-logit
  scaling fitted on half the items halves them on the other half.

### Time

Prompt pass, serving kernels, 4,096-token slices; speedup over exact. The 262,144-token row is a quadratic fit, not a
measurement.

| prompt tokens | 9B exact | 9B cut 24 | 9B cut 16 | Mercurius exact | Mercurius cut 24 | Mercurius cut 16 |
|---|---|---|---|---|---|---|
| 1,024 | 1.9 s | 1.25x | 1.69x | 1.0 s | 1.19x | 1.49x |
| 4,096 | 5.6 s | 1.25x | 1.67x | 3.6 s | 1.18x | 1.46x |
| 16,384 | 28.5 s | 1.26x | 1.69x | 17.7 s | 1.19x | 1.48x |
| 65,536 | 180 s | 1.28x | 1.78x | 120 s | 1.24x | 1.65x |
| 131,072 | 546 s | 1.31x | 1.86x | 391 s | 1.28x | 1.77x |
| 262,144 (fit) | ~1,840 s | ~1.35x | ~1.94x | ~1,400 s | ~1.31x | ~1.87x |

The speedup rises with length because cache-only layers skip their attention scores too. The limit is the share of
full-attention layers skipped: cut 16 skips 4 of 8 (limit 2x), cut 24 skips 2 of 8 (limit 1.33x).

**More timings** (eager attention, seconds):

| arm | 4B at 15k / 32k tokens | 9B at 15k / 32k tokens |
|---|---|---|
| exact | 16.5 / 41.9 | 21.5 / 47.0 |
| cut 24 | 13.9 / 34.8 | 17.2 / 37.7 |
| cut 16 | 11.2 / 27.4 | 13.0 / 28.3 |
| g512 | 12.3 / – | 15.0 / – |
| attention-only 16 | 14.6 / 34.5 | 20.6 / 44.4 |
| MLP + attention 16 | 12.2 / 29.5 | 14.9 / 32.6 |

- **The exact tail is not a fixed overhead.** It is a second forward call and costs about one decode step: 4B
  0.43-0.51 s, 9B 0.95-0.97 s. On the 9B, the 24 DeltaNet layers take 0.72 s of it, the 8 attention layers
  0.23-0.25 s, everything else under 10 ms. Graded depth avoids the call.
- **Memory is unchanged.** Every layer still holds its cache. The maps add 0.10 / 0.20 GB on Mercurius (cut 24 / 16)
  and 0.25 / 0.50 GB on the 9B.

### Decision models (OpenDecider)

As the state pass of a decision model, which reads the prompt only through a softmax over a few options, cut 16 with
ridge maps changes no answer on Banking77 or prompt injections. The 9B's typed decisions stay 94% identical, with
accuracy within noise. The state pass is 1.6x faster: 15k tokens 18.9 s to 11.8 s. See OpenDecider's RESULTS
section 21.

### First-N windows

Coming soon. These runs are still going: `f4/f64/f256_g512` and `f64_tail32` on both models, through
`experiments/depth.py`.

## Reproduce

Each script takes `--model <HF path or id>` (plain model through `terrace.Model`) or `--ckpt <OpenDecider checkpoint>`
(that checkpoint's backbone; how the tables above were measured; needs `opendecider`). Calibration and evaluation text
comes from a FineWeb-Edu parquet (`TERRACE_DATA`). On a shared GPU, set `TERRACE_GPU_LOCK=<lock file>` to serialize
jobs with flock.

| Script | Measures |
|---|---|
| `experiments/lm.py` | Fits maps; generation KL / loss / top-1 / greedy for cuts and variants (`tail<R>`, `ema`, `mlp`, `attnfit`); exact-cache mixes (`--mix exact_attn exact_linear`) |
| `experiments/tasks.py` | OpenBookQA / CommonsenseQA / PubMedQA / GSM8K answer flips against exact and the rounding floor |
| `experiments/depth.py` | Graded depth, first-N windows, error-score selection, part skips (`--parts`): generation + multiple choice + prompt time |
| `experiments/bench.py` | Prompt-pass time, time to first token, peak memory, cache size per variant and length |
| `experiments/overhead.py` | Where the exact tail's time goes (per layer type, CUDA events) |
| `experiments/prefill_sweep.py` | Trunk-only prompt pass from 1k up to 256k+ tokens (preallocated cache, chunked) |
| `experiments/check_exact.py` | Cache-only paths write the exact cache when the map predicts the true input (real checkpoint) |
| `experiments/attnfit.py` | Attention-fitted maps (Mercurius MLA): the maps fine-tuned on what attention reads |

Calibration of option logits and of ranked next-token candidates (`skipkv_calib.py`, `skipkv_rankcal.py`) and the
decision-model evaluations stay in OpenDecider. They use its calibrators and decision heads.

## Related work and what is new

- **SwiftKV** (Qiao et al., arXiv 2410.03960) computes the upper layers' prompt KV from one earlier layer's output and
  adapts the model by distillation. Terrace uses the same split with no training: closed-form ridge maps, cache-only
  paths for DeltaNet recurrent and conv state as well as attention, and one depth for all tokens or a depth per token.
- **KV Prediction** (Horton et al., arXiv 2410.08391) predicts the base model's cache with a small trained auxiliary
  model.
- **FastKV** (arXiv 2502.01068) runs the full context up to a token-selective propagation layer and only selected
  tokens beyond it. **LazyLLM** (arXiv 2407.14057) and **GemFilter** (arXiv 2409.17422) prune or select prompt tokens
  by layer. These are the closest to graded depth, which picks deep tokens by position instead and keeps every token's
  cache in every layer.
- **L2A** (arXiv 2603.17484) learns per token whether to use global attention. That is trained; Terrace is not.

The parts that look new:
- the training-free fit;
- attention-only skips that keep the cache exact while removing the upper layers' prompt attention;
- measurements on hybrid DeltaNet models: the few attention layers carry most of the error, and an exact tail costs a
  full recurrent-layer call;
- the negative result that recency beats error-score token selection.

This is a set of measurements and a working implementation, not a new idea class.

## Limitations

- Measured on two models (Mercurius-1-4B and Qwen3.5-9B) on one device. Other model families need their mixers'
  cache-only paths (stock Qwen3.5 attention and Gated DeltaNet are covered; unknown mixers run whole, MLP skipped).
- Cut 16 alone is not usable for free generation or lettered multiple choice. Use g512 or part skips.
- No cache-memory saving.

## License

Apache-2.0 (see LICENSE and NOTICE).
