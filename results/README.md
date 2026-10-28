# Results

Raw outputs behind the tables in the [main README](../README.md). All runs: Jetson AGX Orin, `MODE_50W`.

These files were written before the project was split out of OpenDecider and renamed DepthSkip. They are kept
exactly as written, so they still carry the old context:

- they were produced by OpenDecider's `scripts/skipkv_*.py`, the scripts that became `experiments/*.py` here. Some
  field names differ from what `experiments/` writes now (for example `ckpt` where `bench.py` now writes `model`);
- `ckpt` paths are OpenDecider runs: `runs/x4b-mercurius-lora/model.pt` is **Mercurius-1-4B**,
  `runs/x9b-clean3/model.pt` is the **Qwen3.5-9B** backbone of OpenDecider's 9B decision model;
- `tasks_4b_calib.json` is keyed by the paths of the logit files it read (`/mnt/ext/opendecider/skipkv/...`).

Write-up of the same runs: OpenDecider `docs/RESULTS.md` §21.2-21.4.

| File | Producer (now) | What it holds | Used in |
|---|---|---|---|
| `depth_4b.json`, `depth_9b.json` | `experiments/depth.py` | Graded depth (`g256`, `g512`), exact tail, per-token reference (`d_*`), error-score selection (`d_sel*`, 9B): generation, OpenBookQA / CommonsenseQA, prompt time 1k-15k | Quality, Per-token selection, Timings |
| `parts_4b.json`, `parts_9b.json` | `experiments/depth.py --parts` | MLP-only / attention-only / both, from 16 and 24, with and without tail; prompt time 1k-32k | Quality, Timings |
| `firstn_4b.json`, `firstn_9b.json` | `experiments/depth.py --arms ...` | First-N windows (`f4/f64/f256_g512`, `f64_tail32`). 200 multiple-choice items on both models | First-N windows |
| `lm_4b_mix.json`, `lm_9b_mix.json` | `experiments/lm.py --mix` | Cut 24 / 16 with the attention or the DeltaNet caches taken from an exact pass | Which caches carry the error |
| `lm_4b_mlaparts.json` | `experiments/lm.py` | Cut 16 with one part of the MLA cache exact (latent, RoPE key, key scale) | Which caches carry the error |
| `lm_4b_calib24.json` | `experiments/lm.py` | Cut 24 against the rounding floor (`cut24_floor`: exact model, prompt pass in other slices) | Quality (floor row) |
| `lm_4b_improve.json`, `lm_4b_fix16.json` | `experiments/lm.py` | Exact tails 32 / 128, causal-EMA features, attention-fitted maps, MLP-only skip | Cut 16 mitigations |
| `lm_4b_mercurius.json`, `lm_4b_mercurius_cacheonly_timing.json` | `experiments/lm.py` | First generation test on Mercurius (60 docs, whole mixers above the cut) and a 2-doc timing run with the cache-only paths | OpenDecider RESULTS §21.2 |
| `lm_4b_rank.json`, `lm_4b_rank_top16_rankcal.json` | `experiments/lm.py`, OpenDecider `scripts/skipkv_rankcal.py` | Next-token candidates and their calibration (ODIR over the top 16) | OpenDecider RESULTS §21.2 |
| `tasks_4b.json`, `tasks_4b_improve.json`, `tasks_4b_fix16.json` | `experiments/tasks.py` | OpenBookQA / CommonsenseQA / PubMedQA / GSM8K accuracy and answer flips (Mercurius-1-4B, 300 items; GSM8K 20) | Other tasks |
| `tasks_4b_calib.json` | OpenDecider `scripts/skipkv_calib.py` | Option-logit calibrators fitted on half the items, scored on the other half | Other tasks |
| `x4b-mercurius-lora_attnfit_cut16.json` | `experiments/attnfit.py` | Held-out attention error per layer before / after the attention fit | Cut 16 mitigations |
| `bench_4b.json`, `bench_9b.json`, `bench_4b_fix16.json` | `experiments/bench.py` | Prompt time, time to first token, peak memory, cache size (eager loader), 1k / 4k / 15k | OpenDecider RESULTS §21.3 |
| `bench_4b_serving.json`, `bench_9b_serving.json` | `experiments/bench.py --serving` | Same on OpenDecider's serving kernels | OpenDecider RESULTS §21.3 |
| `scale_*.json` (exact arm, under `partial`), `scale2_*.json` (cut 24 / 16) | `experiments/bench.py --serving` | 1k to 131k tokens, 4,096-token slices | Prompt-pass speed, Memory |
| `overhead_4b.json`, `overhead_9b.json` | `experiments/overhead.py` | Where the exact tail's time goes (per layer type, CUDA events) | Timings |
| `prefill/smoke_9b.json` | `experiments/prefill_sweep.py` | Smoke run of the trunk-only sweep (lean path, 1k-16k) | not used |
