<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/depthskip-dark.png">
  <img alt="DepthSkip" src="docs/assets/depthskip-light.png" width="420">
</picture>

### Training-free layer skipping for the prompt pass of hybrid LLMs

Prompt tokens stop at a cut layer. The layers above it only write their caches, from closed-form ridge maps.<br>
The model stays frozen and decoding is unchanged: only the prompt (prefill) pass gets cheaper.

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-e07b39.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-e07b39.svg)](pyproject.toml)
[![PyTorch 2.4+](https://img.shields.io/badge/pytorch-2.4%2B-e07b39.svg)](pyproject.toml)
[![transformers 5.2+](https://img.shields.io/badge/transformers-5.2%2B-e07b39.svg)](pyproject.toml)
[![Tests: 20 (CPU)](https://img.shields.io/badge/tests-20%20passing%20(CPU)-1baf7a.svg)](tests/test_depthskip.py)
[![Used by: HgRT](https://img.shields.io/badge/used%20by-HgRT-2a78d6.svg)](https://github.com/Minerva-Laboratories/HgRT)

[**How it works**](#how-it-works) ·
[**Quickstart**](#quickstart) ·
[**Results**](#results) ·
[**Reproduce**](#reproduce) ·
[**Related work**](#related-work) ·
[**HgRT**](https://github.com/Minerva-Laboratories/HgRT) ·
[**OpenDecider**](https://github.com/Minerva-Laboratories/OpenDecider)

</div>

---

DepthSkip makes the prompt pass of hybrid Gated DeltaNet + attention language models cheaper without training. It
works on the Qwen3.5 family and, by duck typing, on Mercurius's MLA and GDN-2 layers.

Prompt tokens run the lower layers as usual and stop at a **cut**. Every layer above the cut still writes its cache for
them: K/V in the attention layers, conv and recurrent state in the DeltaNet layers. The input for that cache comes from
a **closed-form ridge map** of the cut's input. Everything else in those layers is skipped: queries, the attention
itself, output projections and the MLP. Tokens after the prompt run every layer and read that cache, so decoding is
unchanged. On top of this:

- **graded depth** sets the depth per token by position windows. The most recent tokens run every layer, the ones
  before them are cut late, and the rest are cut early. Optionally the first N tokens also run every layer (attention
  sinks);
- **part skips** keep the mixers above the cut and replace only the MLPs and/or the attention outputs by ridge maps. A
  layer whose attention is skipped still writes its cache from its true input, so that cache stays exact.

Nothing is trained. The maps are fitted from one pass of the frozen model over calibration text: 200 documents (48k
tokens) take 2 minutes for Mercurius-1-4B at cut 24 and 5 minutes for the 9B at cut 16 on a Jetson AGX Orin.

## Highlights

<table>
<tr>
<td width="50%" valign="top">

**Training-free, frozen model.**
One ridge map per upper layer, fitted in closed form (normal equations, ridge strength picked by held-out error). No
gradient step touches the model. The maps add 0.10-0.20 GB (Mercurius-1-4B) or 0.25-0.50 GB (9B).

</td>
<td width="50%" valign="top">

**g512 keeps lettered multiple choice at the exact model's accuracy.**
Plain cut 16 collapses on OpenBookQA / CommonsenseQA (9B: 0.315 / 0.200 against 0.885 / 0.815). The graded plan g512
scores 0.890 / 0.815 and cuts cut 16's generation KL by about 4x. At 15k tokens its prompt pass is 1.34x (4B) /
1.43x (9B) faster than exact.

</td>
</tr>
<tr>
<td valign="top">

**The speedup grows with prompt length.**
Cache-only layers also skip their attention scores. On the 9B, cut 16 goes from 1.69x at 1k tokens to 1.86x at 131k,
and cut 24 from 1.25x to 1.31x.

</td>
<td valign="top">

**Drop-in for Hugging Face models.**
The result is an ordinary `transformers` `DynamicCache`, and decoding runs on it as usual. 20 CPU tests check that the
cache-only paths write exactly the full layers' cache when the map predicts the true input.

</td>
</tr>
</table>

The honest part: cut 16 alone changes free-text generations and breaks lettered multiple choice. An exact tail repairs
multiple choice but not generation. Cut 24 stays near the model's own rounding noise. There is no memory saving. See
[Trade-offs](#trade-offs).

## How it works

### Cache-only upper layers

For a cut `c`, prompt tokens run layers `0 .. c-1` normally. Each layer `j >= c` is replaced for the prompt pass by a
cache-only stand-in (`CacheOnlyLayer`). It takes `h`, the input of layer `c`, predicts layer `j`'s input from it, and
runs only what the cache needs:

| Layer type | Runs for prompt tokens above the cut | Skipped |
|---|---|---|
| full attention (stock Qwen3.5) | `k_proj`, `v_proj`, `k_norm`, RoPE, cache write | queries and gate, attention, `o_proj`, MLP |
| Gated DeltaNet (stock Qwen3.5) | `in_proj_qkv`, `in_proj_a`, `in_proj_b`, conv, delta-rule state update | `z`, gated norm, `out_proj`, MLP |
| Mercurius MLA (`mercurius_cache._compressed_forward`) | this call's latent / RoPE-key codes | the rest of the mixer, MLP |
| Mercurius GDN-2 (`mercurius_port._forward`) | up to the state update | `z`, gated norm, `out_proj`, MLP |
| any other mixer | the whole mixer, on the predicted input | MLP only |

The stand-in returns its input unchanged, so every layer above the cut sees the cut's input `h`.

### Skip maps

Each layer `j >= c` gets one ridge map from the cut's input (`maps.py`):

- `normed` (default): `n(h) W_j + b_j` goes straight to layer `j`'s normed input, where `n` is RMS normalization
  without gain. It is linear up to the projections, so a map could later be folded into them;
- `input`: `h W_j + b_j`, then layer `j`'s own input norm.

`fit_skip_maps` accumulates the normal equations in float64 over exact passes of the calibration text. It solves them
for every ridge strength in `LAMS = (1e-4, 1e-3, 1e-2, 1e-1)` (relative to the mean diagonal) and keeps the one with
the lowest held-out error for all layers. With `alphas=(0.9, 0.99)`, the map input also gets causal exponential moving
averages of `n(h)` as context features.

### Plans

A plan is a list of prompt segments `[(n_tokens or -1, maps or None), ...]` in prompt order (`segment_prefill`).
Segments after the `-1` segment are filled from the end of the prompt, and segments before it from the start. `-1`
takes the rest, and `None` means every layer. Each segment runs on the cache that the previous segments wrote. These are
the plans used in the experiments (`experiments/depth.py`; `m16`, `m24` are the cut-16 / cut-24 maps):

| Plan | Segments | Meaning |
|---|---|---|
| `cut24`, `cut16` | `[(-1, m24)]`, `[(-1, m16)]` | every prompt token cut at 24 / 16 |
| `cut16_tail32` | `[(-1, m16), (32, None)]` | the exact tail: the last 32 prompt tokens run every layer |
| `g256` | `[(-1, m16), (224, m24), (32, None)]` | graded: the last 32 run every layer, the 224 before them are cut at 24, the rest at 16 |
| **`g512`** | `[(-1, m16), (480, m24), (32, None)]` | graded: the last 32 run every layer, the 480 before them are cut at 24, the rest at 16 |
| `f4_g512`, `f64_g512`, `f256_g512` | `[(N, None), (-1, m16), (480, m24), (32, None)]` | g512, and the first N = 4 / 64 / 256 tokens also run every layer |
| `f64_tail32` | `[(64, None), (-1, m16), (32, None)]` | cut 16 with the first 64 and the last 32 tokens at full depth |
| `mlp16`, `attn16`, `mlpattn16` (and `*24`) | `[(-1, part_skip(...))]` | part skips: above the cut, only the MLPs, only the full-attention mixers, or both become ridge maps. DeltaNet mixers always run |

`DepthPlan` + `depth_cache` is a per-token-depth reference implementation. It writes the same cache as
`segment_prefill` for recency plans, and it can also send extra tokens deeper, chosen by an attention-error score
(`error_score`: `sum p_qi ||v_i - o_q||` over the last W queries, at the stock attention layer right below the base cut).
It runs every layer for every token, so its compute is not representative. `skipped_share`, `part_share` and
`plan_skipped` give the analytic share of the prompt pass's weight work that is skipped.

A single cut can also run through `Model.prefill` with `model.skip_state = maps`. Then `SkipMaps.tail = R` runs the
last R prompt tokens through every layer, in a second call.

## Quickstart

```bash
git clone https://github.com/Minerva-Laboratories/DepthSkip && cd DepthSkip
pip install -e .                        # torch>=2.4, transformers>=5.2, numpy
pip install -e ".[experiments,test]"    # + pyarrow, datasets, pytest
python -m pytest -q                     # CPU, tiny random Qwen3.5 (20 tests)
```

```python
import torch
from depthskip import Model, SkipMaps, fit_skip_maps, segment_prefill

model = Model.from_pretrained("Qwen/Qwen3.5-9B", dtype="bfloat16", device="cuda", prefill_chunk=4096)

# Fit (or load) one map set per cut: 200 calibration documents of 256 tokens took 2-5 minutes on the Orin.
calib, held = texts[:200], texts[200:210]           # strings or token-id lists
m16 = fit_skip_maps(model, calib, held, cut=16, mode="normed")
m24 = fit_skip_maps(model, calib, held, cut=24, mode="normed")
m16.save("maps_cut16.pt")                           # later: SkipMaps.load("maps_cut16.pt", "cuda")

# g512 prompt pass over all but the last prompt token
ids = model.tokenize([prompt])[0]
cache = segment_prefill(model, ids[:-1], [(-1, m16), (480, m24), (32, None)])
# first-N variant: [(64, None), (-1, m16), (480, m24), (32, None)]

# `cache` is a transformers DynamicCache. Decode as usual: every new token runs all layers.
x = torch.tensor([ids[-1:]], device=model.device)
h, cache = model.forward(x, torch.ones_like(x, dtype=torch.bool), past_key_values=cache)
next_id = int(model.head_logits(h[:, -1]).argmax(-1))
```

One cut for the whole prompt, through `Model.prefill`:

```python
model.skip_state = SkipMaps.load("maps_cut16.pt", "cuda", tail=32)   # tail=R: last R tokens run every layer (2nd call)
_, cache = model.prefill(torch.tensor(ids[:-1]))
model.skip_state = None                                               # back to exact prompt passes
```

Part skips (one fit from the lowest cut serves every cut above it):

```python
from depthskip import fit_parts, part_skip
fit = fit_parts(model, calib, held, base=16)
attn16 = part_skip(fit, cut=16, mlp=False, attn=True, device=model.device)
cache = segment_prefill(model, ids[:-1], [(-1, attn16), (32, None)])       # attention-only 16 + tail 32
```

### API

| Name | What it does |
|---|---|
| `Model.from_pretrained(path, dtype="bfloat16", device="cuda", prefill_chunk=4096, **kw)`, `Model.wrap(causal_lm, tokenizer=None, prefill_chunk=0)` | Wrapper over a HF causal LM: `tokenize(texts)`, `prefill(ids, mask=None, layers=("final",), cache=None, keep_hidden=False) -> (feats, cache)`, `forward(ids, mask, ..., past_key_values=None) -> (hidden, cache)`, `head_logits(h)`, `prefix_cache_batch(seqs)` |
| `SkipMaps(cut, mode, W, b, lam=0.0, alphas=(), tail=0)`, `.save(path, meta=None)`, `SkipMaps.load(path, device="cuda", tail=0)` | One ridge map per layer from the cut up (`mode`: `"normed"` or `"input"`) |
| `fit_skip_maps(model, texts, held_texts, cut, mode="normed", lams=LAMS, log=print, alphas=())` | Closed-form fit; ridge strength by held-out error; `alphas`: causal-EMA context features |
| `skipping(lm, maps)` | Context manager: layers from the cut up become cache-only (`SkipMaps`), or their MLPs become linear (`MLPSurrogates`) |
| `fit_mlp_surrogates(model, texts, held_texts, cut, lams=LAMS, log=print)`, `MLPSurrogates` | MLP-only skip: mixers run, MLPs become linear maps |
| `fit_parts(model, texts, held_texts, base, lams=LAMS, log=print)`, `part_skip(fit, cut, mlp, attn, device="cuda")`, `PartSkip` | Part skips: MLP and/or attention-output surrogates; attention caches are written from the true inputs |
| `seg_lengths(T, segs)`, `segment_cache(model, ids, segs)` (alias `segment_prefill`) | Graded depth: `segs = [(n or -1, SkipMaps / PartSkip / None), ...]` in prompt order |
| `DepthPlan(maps, n_layers, windows=(), select=None)`, `depth_cache(model, ids, plan)`, `depth_skipping`, `error_score` | Per-token-depth reference; `select = (fraction, cut, W)` adds tokens by the attention-error score |
| `skipped_share(cfg, cut, whole_mixer=False)`, `part_share(cfg, cut, mlp, attn)`, `plan_skipped(cfg, ct, L, whole_mixer)` | Analytic share of the prompt pass's weight-multiply work that is skipped (attention scores not counted) |
| `cache_tensors(cache)`, `use_reference_kernels(enable=True)` | Every tensor in a cache (for checks); pure-torch DeltaNet / conv kernels (CPU, tests) |

Any object with the same attributes as `Model` works in its place: `lm`, `tokenize`, `device`, `hidden_size`,
`cfg.prefill_chunk`, `skip_state`, `prefix_cache_batch`, `_prefill`. OpenDecider's `Backbone` is one, and its decision
model can use DepthSkip for its state pass.

## Results

All measurements: **Jetson AGX Orin, `MODE_50W`**. Two models, both run as the backbones of OpenDecider checkpoints
(`--ckpt`):

- **Mercurius-1-4B** ("4B"): 32 layers, MLA + GDN-2, 4-bit weights and cache;
- **the 9B**: Qwen3.5-9B, 32 layers, full attention at layers 3, 7, ..., 31.

Maps are `normed`, fitted on 200 FineWeb-Edu documents (256 tokens each). Raw outputs are in [`results/`](results/)
(index: [`results/README.md`](results/README.md)). **The raw files predate the rename to DepthSkip:** they were
written by OpenDecider's `scripts/skipkv_*.py` (now `experiments/*.py`), and their `ckpt` fields are OpenDecider run
paths (`runs/x4b-mercurius-lora` = Mercurius-1-4B, `runs/x9b-clean3` = the 9B).

### Generation

Prompts: 30 FineWeb-Edu documents cut to 1,024 tokens, with the next 128 tokens teacher-forced.

- **skipped**: analytic share of the prompt pass's weight work that is skipped (4B / 9B). For Mercurius it counts only
  the skipped MLPs, so the 4B shares of the cut arms are lower bounds;
- **KL**: KL(exact ‖ skipped) of the next-token distribution;
- **loss Δ**: change in continuation NLL;
- **same top-1**: share of positions with the same top-1 token as exact;
- **greedy identical**: share of 32-token greedy continuations identical to exact.

| arm | skipped | 4B KL | 9B KL | 4B loss Δ | 9B loss Δ | same top-1 (4B / 9B) | greedy identical (4B / 9B) |
|---|---|---|---|---|---|---|---|
| exact | 0 | 0 | 0 | 0 | 0 | 100% | 100% |
| model's own rounding floor (prompt pass in other slices) | – | 0.0040 | – | −0.000 | – | 96.3% / – | 53% / – |
| cut 24 | 14% / 20% | 0.0084 | 0.0058 | +0.008 | +0.006 | 96.3% / 96.5% | 27% / 27% |
| cut 16 | 29% / 40% | 0.085 | 0.094 | +0.082 | +0.085 | 89.1% / 89.2% | 0% / 0% |
| cut 16 + exact tail 32 | 28% / 39% | 0.068 | 0.075 | +0.075 | +0.075 | 89.7% / 90.8% | 3% / 0% |
| g256 | 25% / 34% | 0.036 | 0.039 | +0.038 | +0.039 | 92.6% / 92.9% | 10% / 10% |
| **g512** | 21% / 29% | 0.023 | 0.024 | +0.027 | +0.027 | 93.5% / 94.2% | 17% / 23% |
| attention-only 16 | 2.8% / 1.9% | 0.0047 | 0.0025 | +0.003 | +0.003 | 96.8% / 97.9% | 33% / 50% |
| attention-only 16 + tail 32 | 2.7% / 1.9% | 0.0050 | 0.0021 | +0.004 | +0.002 | 96.5% / 97.9% | 37% / 53% |
| MLP-only 16 | 29% / 31% | 0.050 | 0.046 | +0.051 | +0.043 | 91.4% / 92.3% | 3% / 3% |
| MLP + attention 16 | 32% / 33% | 0.053 | 0.052 | +0.056 | +0.050 | 90.9% / 92.1% | 3% / 0% |
| MLP-only 24 | 14% / 16% | 0.0068 | 0.0045 | +0.007 | +0.005 | 96.6% / 97.1% | 23% / 43% |
| attention-only 24 | 1.4% / 1.0% | 0.0015 | 0.0007 | +0.000 | +0.001 | 98.0% / 98.9% | 60% / 63% |

<sub>Sources: `results/depth_{4b,9b}.json` (cut, tail, g256, g512), `results/parts_{4b,9b}.json` (part skips),
`results/lm_4b_calib24.json` (floor: `cut24_floor`, the exact model with its prompt pass in other slices, same
math with other rounding). Files written before the rename (see above).</sub>

### Multiple choice

OpenBookQA / CommonsenseQA, lettered, with the chat template: 300 items (4B) and 200 items (9B). Accuracy, with the
share of answers that differ from exact in parentheses.

| arm | 4B OpenBookQA | 4B CommonsenseQA | 9B OpenBookQA | 9B CommonsenseQA |
|---|---|---|---|---|
| exact | 0.797 | 0.723 | 0.885 | 0.815 |
| rounding floor | 0.803 (2.3%) | 0.713 (4.3%) | – | – |
| cut 24 | 0.813 (6.0%) | 0.773 (12.7%) | 0.900 (2.5%) | 0.820 (2.0%) |
| cut 16 | 0.403 (63.0%) | 0.283 (80.3%) | 0.315 (71.5%) | 0.200 (84.5%) |
| cut 16 + exact tail 32 | 0.710 (14.3%) | 0.630 (19.0%) | 0.835 (7.0%) | 0.775 (10.0%) |
| g256 | 0.800 (3.7%) | 0.717 (4.3%) | 0.890 (1.0%) | 0.815 (0.0%) |
| **g512** | 0.800 (3.7%) | 0.717 (4.3%) | 0.890 (1.0%) | 0.815 (0.0%) |
| attention-only 16 | 0.807 (15.3%) | 0.743 (21.7%) | 0.840 (9.0%) | 0.780 (10.0%) |
| attention-only 16 + tail 32 | 0.790 (3.7%) | 0.703 (6.0%) | 0.890 (1.0%) | 0.815 (1.5%) |
| MLP-only 16 | 0.437 (58.7%) | 0.257 (81.7%) | 0.795 (16.0%) | 0.755 (22.0%) |
| MLP + attention 16 | 0.450 (57.7%) | 0.257 (82.7%) | 0.765 (20.0%) | 0.695 (30.0%) |
| MLP-only 24 | 0.793 (4.3%) | 0.740 (8.0%) | 0.890 (2.5%) | 0.815 (4.0%) |
| attention-only 24 | 0.810 (6.0%) | 0.740 (5.3%) | 0.890 (0.5%) | 0.815 (2.0%) |

<sub>Sources: `results/depth_{4b,9b}.json`, `results/parts_{4b,9b}.json`; floor from `results/tasks_4b.json`
(exact model, prompt pass in 32-token slices).</sub>

### Trade-offs

**Cut 24 is close to the model's own rounding noise.**
- Its KL is 2x the floor, with the same top-1 agreement (96.3%). It loses no multiple-choice accuracy on either model.
- Greedy continuations match exact less often than the floor does (27% against 53%). The 4-bit Mercurius is
  greedy-identical to itself only about half the time under a different prompt slicing, so greedy identity overstates
  the damage.
- Its extra answer changes on CommonsenseQA (12.7% against a 4.3% floor) are miscalibration. Option-logit vector scaling
  fitted on half the items halves them on the other half (6.3%; `results/tasks_4b_calib.json`).

**Cut 16 alone changes free-text generations.**
- It never gives a greedy continuation identical to exact (0% on both models), and its KL is about 20x the floor.
- On lettered multiple choice it collapses: it picks early letters.
- An exact tail (the last 32 tokens through every layer) restores most of the multiple-choice accuracy (9B 0.835 /
  0.775), but **not** free generation (greedy identical 3% / 0%).
- The cause is mostly the most recent prompt tokens, so depth by recency fixes it. **g512** keeps both models at exact
  accuracy and cuts cut 16's KL about 4x. Its greedy identity (17% / 23%) approaches cut 24's (27%).

**Attention-only skips have the lowest KL, but not the lowest answer changes.**
- Attention-only 16 is within cut 24's KL on both models.
- On multiple choice it changes 15-22% of the 4B's answers (9-10% on the 9B) at the same accuracy. With an exact tail
  it drops to 1-6%.
- It skips little weight work (1-3%). Its gain is the upper layers' prompt attention, which grows with length (see
  [Timings](#timings)).

**MLP-only skips** halve cut 16's KL but keep most of its cost. On the 9B they keep multiple choice much better than on
the 4B (0.795 against 0.437 on OpenBookQA).

**No memory saving.** Every layer still holds its cache (later tokens read it). See [Memory](#memory).

### First-N windows

The first N prompt tokens also run every layer, on top of g512 (sink-like). Generation as above. Multiple choice uses
200 items on **both** models here, so the 4B's exact accuracy differs from the table above (0.785 / 0.720).

| plan | skipped (4B / 9B) | KL (4B / 9B) | greedy identical (4B / 9B) | 4B OBQA / CSQA | 9B OBQA / CSQA | prompt time 1k (4B / 9B) | 15k | 32k |
|---|---|---|---|---|---|---|---|---|
| exact | 0 | 0 | 100% | 0.785 / 0.720 | 0.885 / 0.815 | 1.01 / 2.16 s | 16.46 / 21.63 s | 42.09 / 47.23 s |
| cut 16 | 29% / 40% | 0.085 / 0.094 | 0% / 0% | 0.405 / 0.240 | 0.315 / 0.200 | 0.68 / 1.25 s | 11.16 / 13.06 s | 27.51 / 28.29 s |
| cut 16 + tail 32 | 28% / 39% | 0.068 / 0.075 | 3% / 0% | 0.700 / 0.635 | 0.835 / 0.775 | 1.10 / 2.18 s | 11.82 / 14.03 s | 28.45 / 29.44 s |
| g512 | 21% / 29% | 0.023 / 0.024 | 17% / 23% | 0.780 / 0.715 | 0.890 / 0.815 | 1.37 / 2.95 s | 12.25 / 14.99 s | 29.36 / 30.73 s |
| f4_g512 | 21% / 29% | 0.023 / 0.024 | 13% / 17% | 0.780 / 0.715 | 0.890 / 0.815 | 1.80 / 3.91 s | 13.00 / 22.53 s | 30.32 / 100.34 s |
| f64_g512 | 19% / 27% | 0.020 / 0.018 | 17% / 17% | 0.780 / 0.715 | 0.890 / 0.815 | 1.83 / 3.91 s | 12.97 / 22.47 s | 30.06 / 102.07 s |
| f256_g512 | 14% / 19% | 0.014 / 0.0092 | 23% / 30% | 0.780 / 0.715 | 0.890 / 0.815 | 1.74 / 3.94 s | 12.81 / 21.93 s | 29.95 / 98.46 s |
| f64_tail32 | 26% / 36% | 0.063 / 0.063 | 7% / 0% | 0.770 / 0.730 | 0.890 / 0.815 | 1.61 / 3.19 s | 12.53 / 22.12 s | 29.44 / 78.65 s |

<sub>Source: `results/firstn_{4b,9b}.json` (copied from the same OpenDecider run directory as the other files; written
before the rename).</sub>

- First-N windows lower the KL further: f256_g512 brings the 9B to 0.0092, between cut 24 and g512. Part of this is
  that more of a 1,024-token prompt runs at full depth (19% skipped instead of 29%). Multiple choice does not change:
  g512 already matches exact there.
- **Their timings are not settled.** On the 4B, a first-N window costs about one more segment call (+0.4-1.0 s over
  g512). On the 9B at 15k and 32k, the first-N plans were *slower than exact* (32k: 98-102 s against 47 s). The cause
  is not established. One difference: in these plans the long cut-16 segment is the first to run on a non-empty
  cache, which in `transformers` takes the masked attention path instead of the causal one. Use g512 until this is
  resolved.

### Where the error comes from

- **The few full-attention layers carry most of it.** Taking the attention caches from an exact pass cuts cut 16's KL
  from 0.085 to 0.016 (4B) and from 0.094 to 0.016 (9B). Taking the DeltaNet caches exact cuts it only to 0.048 /
  0.050. This holds for stock attention (9B) and for MLA (Mercurius) alike (`results/lm_{4b,9b}_mix.json`).
- Within Mercurius's MLA cache, the error is in the latent: an exact latent gives KL 0.056, an exact RoPE key 0.084
  (no change) (`results/lm_4b_mlaparts.json`).
- **Recency beats error-score selection.** On the 9B, sending the last 32 tokens plus 5% of tokens chosen by the
  attention-error score through every layer (`d_sel5`) gives KL 0.059. The same number of most recent tokens
  (`d_tail83`) gives 0.058. On top of g512, 5% by score moves KL from 0.024 to 0.022 (`d_g512_sel5`,
  `results/depth_9b.json`).
- **Temperature and output-side calibration do not remove it.** The best single temperature is 1.0
  (`results/lm_4b_calib24.json`). A ridge map on the final hidden state does not help either (OpenDecider RESULTS
  §21.2).

### Cut 16 mitigations (Mercurius-1-4B)

| cut 16 variant | KL | loss Δ | greedy identical | OpenBookQA (exact 0.797) | CommonsenseQA (exact 0.723) |
|---|---|---|---|---|---|
| closed-form maps | 0.085 | +0.082 | 0% | 0.403 | 0.283 |
| + exact tail 32 | 0.068 | +0.075 | 3% | 0.710 | 0.630 |
| + exact tail 128 | 0.043 | +0.052 | 13% | – | – |
| causal-EMA context features (`alphas=(0.9, 0.99)`) | 0.079 | +0.076 | 3% | 0.410 | 0.287 |
| EMA features + tail 32 | 0.063 | +0.067 | 3% | 0.723 | 0.613 |
| attention-fitted maps (`experiments/attnfit.py`) | 0.078 | +0.076 | 0% | 0.407 | 0.283 |
| attention-fitted + tail 32 | 0.062 | +0.069 | 3% | 0.723 | 0.640 |
| **g512** (for comparison) | 0.023 | +0.027 | 17% | 0.800 | 0.717 |

<sub>Sources: `results/lm_4b_improve.json`, `results/lm_4b_fix16.json`, `results/tasks_4b_improve.json`,
`results/tasks_4b_fix16.json`, `results/depth_4b.json`.</sub>

None of these brings cut 16 near cut 24. Graded depth does the most, in one plan.

### Other tasks (Mercurius-1-4B)

PubMedQA (yes/no over an abstract, 300 items) is robust: exact 0.830, cut 24 0.830, cut 16 0.820 (5.7% of answers
changed, floor 2.3%). GSM8K (20 problems, 200-token cap) was not informative: replies were cut at the cap (exact 5%
accuracy), and final answers changed in 55% (floor) and 50% (cut 24) of problems alike (`results/tasks_4b.json`).

### Prompt-pass speed

Prompt pass on OpenDecider's serving kernels, in 4,096-token slices; speedup over exact. The 262,144-token row is a
quadratic fit, not a measurement: the exact pass ran below the 12 GB free-memory guard and was stopped.

| prompt tokens | 9B exact | 9B cut 24 | 9B cut 16 | Mercurius exact | Mercurius cut 24 | Mercurius cut 16 |
|---|---|---|---|---|---|---|
| 1,024 | 1.9 s | 1.25x | 1.69x | 1.0 s | 1.19x | 1.49x |
| 4,096 | 5.6 s | 1.25x | 1.67x | 3.6 s | 1.18x | 1.46x |
| 16,384 | 28.5 s | 1.26x | 1.69x | 17.7 s | 1.19x | 1.48x |
| 65,536 | 180 s | 1.28x | 1.78x | 120 s | 1.24x | 1.65x |
| 131,072 | 546 s | 1.31x | 1.86x | 391 s | 1.28x | 1.77x |
| 262,144 (fit) | ~1,840 s | ~1.35x | ~1.94x | ~1,400 s | ~1.31x | ~1.87x |

<sub>Sources: `results/scale_{x9b-clean3,x4b-mercurius-lora}.json` (exact, under `partial`),
`results/scale2_{x9b-clean3,x4b-mercurius-lora}.json` (cut 24 / 16). The 262k row is from OpenDecider RESULTS §21.3
(the fit is not in `results/`).</sub>

The speedup rises with length because cache-only layers also skip their attention scores. The limit is the share of
full-attention layers skipped: cut 16 skips 4 of 8 (limit 2x), cut 24 skips 2 of 8 (limit 1.33x).

### Timings

Eager attention, OpenDecider's serving loader, seconds:

| arm | 4B 1k / 4k / 15k / 32k | 9B 1k / 4k / 15k / 32k |
|---|---|---|
| exact | 0.97 / 3.58 / 16.5 / 41.9 | 2.18 / 5.84 / 21.5 / 47.0 |
| cut 24 | 0.82 / 3.03 / 13.9 / 34.8 | 1.72 / 4.62 / 17.2 / 37.7 |
| cut 16 | 0.66 / 2.45 / 11.2 / 27.4 | 1.26 / 3.40 / 13.0 / 28.3 |
| cut 16 + tail 32 | 1.07 / 2.91 / 11.8 / 28.3 | 2.19 / 4.37 / 14.0 / 29.4 |
| **g512** | 1.33 / 3.26 / 12.3 / – | 2.95 / 5.48 / 15.0 / – |
| attention-only 16 | 0.92 / 3.36 / 14.6 / 34.5 | 2.13 / 5.67 / 20.6 / 44.4 |
| MLP + attention 16 | 0.74 / 2.72 / 12.2 / 29.5 | 1.43 / 3.95 / 14.9 / 32.6 |

<sub>Sources: `results/parts_{4b,9b}.json` (`prompt_seconds`), g512 from `results/depth_{4b,9b}.json` (timed to
15k only).</sub>

- **Graded plans pay only on long prompts.** g512 runs as one call per segment, and each extra call costs about one
  decode step (below). At 1k tokens g512 is *slower* than exact (0.73x on both models). At 4k it is 1.10x (4B) /
  1.06x (9B), and at 15k 1.34x / 1.43x.
- **The exact tail is a second forward call** and costs about one decode step: 4B 0.43-0.51 s, 9B 0.95-0.98 s on 1k /
  4k prompts. On the 9B, the 24 DeltaNet layers take 0.72 s of it, the 8 attention layers 0.23-0.25 s, and everything
  else under 10 ms. It is the per-call cost of the recurrent layers, not overhead that a fused kernel would remove
  (`results/overhead_{4b,9b}.json`).
- **Attention-only skips** save 4-18% at 15k-32k (more on the 4B). Their gain grows with length, and at 1k they save
  little.

### Memory

Every layer keeps its cache, so the cache and the peak memory are those of the exact model. The maps add their own size.

| | 9B exact | 9B cut 24 | 9B cut 16 | Mercurius exact | Mercurius cut 24 | Mercurius cut 16 |
|---|---|---|---|---|---|---|
| cache at 131k tokens | 4.05 GB | 4.05 GB | 4.05 GB | 0.35 GB | 0.35 GB | 0.35 GB |
| peak over weights at 131k | 8.74 GB | 8.77 GB | 8.74 GB | 3.94 GB | 3.96 GB | 3.95 GB |
| maps | – | 0.25 GB | 0.50 GB | – | 0.10 GB | 0.20 GB |

<sub>Sources: `results/scale_*.json`, `results/scale2_*.json` (`cache_gb_131072`, `peak_over_weights_gb_131072`,
`maps_gb`).</sub>

### Decision models (OpenDecider)

A decision model reads the prompt only through a softmax over a few options. Used as its state pass, cut 16 with ridge
maps changes no answer on Banking77 or prompt injections. The 9B's typed decisions stay 94% identical, with accuracy
within noise. The state pass is 1.6x faster: 15k tokens go from 18.9 s to 11.8 s. Source: OpenDecider
`docs/RESULTS.md` §21.1 (not in `results/` here).

## Reproduce

Each script takes `--model <HF path or id>` (a plain model through `depthskip.Model`) or `--ckpt <OpenDecider
checkpoint>` (that checkpoint's backbone; needs `opendecider`). The tables above were measured with `--ckpt`.
Calibration and evaluation text comes from a FineWeb-Edu parquet (`DEPTHSKIP_DATA`). Multiple-choice items are read
from `data/public/<task>.jsonl` relative to the working directory (OpenDecider's data layout).

| Script | Measures |
|---|---|
| `experiments/lm.py` | Fits maps; generation KL / loss / top-1 / greedy for cuts and variants (`tail<R>`, `ema`, `mlp`, `attnfit`); exact-cache mixes (`--mix exact_attn exact_linear ...`) |
| `experiments/tasks.py` | OpenBookQA / CommonsenseQA / PubMedQA / GSM8K answer flips against exact and the rounding floor |
| `experiments/depth.py` | Graded depth, first-N windows, error-score selection, part skips (`--parts`): generation, multiple choice and prompt time |
| `experiments/bench.py` | Prompt-pass time, time to first token, peak memory and cache size per variant and length (`--serving`: OpenDecider's serving kernels) |
| `experiments/overhead.py` | Where the exact tail's time goes (per layer type, CUDA events) |
| `experiments/prefill_sweep.py` | Trunk-only prompt pass from 1k up to 256k+ tokens (preallocated cache, chunked) |
| `experiments/check_exact.py` | Cache-only paths write the exact cache when the map predicts the true input (real checkpoint, GPU) |
| `experiments/attnfit.py` | Attention-fitted maps (Mercurius MLA): the maps fine-tuned on what attention reads |

```bash
python experiments/depth.py --ckpt runs/x4b-mercurius-lora/model.pt --out results/maps/depth_4b.json
python experiments/depth.py --model Qwen/Qwen3.5-9B --prefill-chunk 4096 --out results/maps/depth_9b.json
```

Calibration of option logits and of ranked next-token candidates (`skipkv_calib.py`, `skipkv_rankcal.py`) and the
decision-model evaluations stay in OpenDecider, because they use its calibrators and decision heads.

### Environment variables

The library itself reads none. The experiment scripts read:

| Variable | Default | Meaning |
|---|---|---|
| `DEPTHSKIP_DATA` | a local FineWeb-Edu sample parquet | Calibration / evaluation text: any parquet with a `text` column |
| `DEPTHSKIP_GPU_LOCK` | unset (no lock) | Lock file: serialize GPU jobs on a shared machine with `flock` (falls back to `OPENDECIDER_GPU_LOCK`) |
| `DEPTHSKIP_GPU_LOCK_HELD` | set by the lock | Marks a job already running under the lock (re-entrant; `OPENDECIDER_GPU_LOCK_HELD` is honored too) |
| `DEPTHSKIP_HF_CACHE` | HF default | `datasets` cache directory for GSM8K (`experiments/tasks.py`) |
| `OPENDECIDER_STATE_CACHE_MB` | set to `0` by the scripts | Disables OpenDecider's state-memory cache for `--ckpt` backbones |
| `TITLE_CHROME` | Playwright's headless Chromium | Browser used by `scripts/make_title.py` to render the title PNGs |

## Repository layout

```
src/depthskip/
  maps.py         skip maps: SkipMaps, fit_skip_maps, CacheOnlyLayer, skipping, MLPSurrogates, skipped_share
  depth.py        graded depth: segment_cache / segment_prefill, seg_lengths, DepthPlan, depth_cache, error_score
  parts.py        part skips: fit_parts, part_skip, PartSkip, part_share
  model.py        Model (HF causal-LM wrapper), cache_tensors, use_reference_kernels
tests/            CPU tests on a tiny random Qwen3.5 (3 DeltaNet : 1 attention, 8 layers)
experiments/      measurement scripts (GPU); common.py: model loading, GPU lock, timing
results/          raw JSON outputs of the runs above (index: results/README.md)
scripts/          make_title.py: the README title artwork
docs/assets/      title artwork (the bars under the wordmark are g512's depth profile)
```

## Testing

```bash
python -m pytest -q      # 20 tests, CPU only, about 20 s
```

The tests build a tiny random Qwen3.5-architecture model (8 layers, 3 DeltaNet : 1 attention) and use the pure-torch
reference kernels (`use_reference_kernels`). They check that:

- cache-only layers write the full layers' cache when the map predicts the true input (several cuts, prompt slicing,
  the generic mixer path);
- the fit recovers a pass-through map, and save / load round-trips;
- the causal EMA scan matches a loop, and the exact tail and the EMA context features keep the cache exact;
- MLP surrogates and part skips are exact when the replaced parts are zero, and `fit_parts` recovers zero parts;
- `seg_lengths` fills head and tail windows correctly, `segment_cache` matches the per-token-depth reference, and a
  first-N window runs every layer.

`experiments/check_exact.py` runs the same exactness check on a real checkpoint (GPU).

## Related work

- **SwiftKV** (Qiao et al., arXiv 2410.03960) computes the upper layers' prompt KV from one earlier layer's output and
  adapts the model by distillation. DepthSkip uses the same split with no training: closed-form ridge maps, cache-only
  paths for DeltaNet recurrent and conv state as well as attention, and one depth for all tokens or a depth per token.
- **KV Prediction** (Horton et al., arXiv 2410.08391) predicts the base model's cache with a small trained auxiliary
  model.
- **FastKV** (arXiv 2502.01068) runs the full context up to a token-selective propagation layer and only selected
  tokens beyond it. **LazyLLM** (arXiv 2407.14057) and **GemFilter** (arXiv 2409.17422) prune or select prompt tokens
  by layer. These are the closest to graded depth, which instead picks deep tokens by position and keeps every token's
  cache in every layer.
- **L2A** (arXiv 2603.17484) learns per token whether to use global attention. That is trained; DepthSkip is not.

The parts that look new:
- the training-free fit;
- attention-only skips that keep the cache exact while removing the upper layers' prompt attention;
- measurements on hybrid DeltaNet models: the few attention layers carry most of the error, and an exact tail costs a
  full recurrent-layer call;
- the negative result that recency beats error-score token selection.

This is a set of measurements and a working implementation, not a new idea class.

## Part of a stack

- [**HgRT**](https://github.com/Minerva-Laboratories/HgRT): the long-context stack as one system. DepthSkip is its
  prompt pass, handed off to a decode loop with FoveaKV pages and SparsePage selection.
- [**OpenDecider**](https://github.com/Minerva-Laboratories/OpenDecider): decision models on frozen LMs. DepthSkip is an
  optional state pass (`pip install opendecider[depthskip]`, maps file in `OPENDECIDER_SKIPKV`).
- [**FoveaKV**](https://github.com/Minerva-Laboratories/FoveaKV): 2+2-bit KV cache for long-context decoding.
- [**SparsePage**](https://github.com/Minerva-Laboratories/SparsePage): summary-scored page selection for decode
  attention.

## Limitations

- Measured on two models (Mercurius-1-4B and Qwen3.5-9B) on one device (Jetson AGX Orin, `MODE_50W`). Other model
  families need cache-only paths for their mixers. Stock Qwen3.5 attention and Gated DeltaNet are covered. Unknown
  mixers run whole on the predicted input, and only their MLP is skipped.
- Cut 16 alone is not usable for free generation or lettered multiple choice. Use g512 or part skips, or cut 24 when
  outputs must stay within the model's own noise.
- No cache-memory saving: the maps add memory.
- Graded plans run one call per segment, so they are slower than exact on short prompts (1k tokens). The first-N plans'
  long-prompt timings on the 9B are unexplained (see [First-N windows](#first-n-windows)).
- Maps are per model and per cut, and they are fitted on generic web text (FineWeb-Edu). Other prompt distributions
  were not studied.
- `DepthPlan` / `depth_cache` and error-score selection are quality references, not fast paths. Selection needs a stock
  attention layer right below the base cut.

## License

Apache-2.0: see [LICENSE](LICENSE) and [NOTICE](NOTICE). Model weights (Qwen3.5 and derived checkpoints) are not
included. They are downloaded from their original repositories and keep their own licenses.
