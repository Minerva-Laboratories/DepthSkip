"""Shared pieces of the experiment scripts: model loading, the optional GPU lock, timing, the evaluation text.

Models: --model <HF path or hub id> loads a plain causal LM through depthskip.Model. --ckpt <OpenDecider checkpoint> loads
that checkpoint's backbone instead (needs the opendecider package; how the RESULTS numbers for the Mercurius-1-4B and
OpenDecider 9B backbones were measured). Both expose the same calls (lm, tok, tokenize, prefill, forward, head_logits,
layer_types, cfg.prefill_chunk, skip_state).
"""
from __future__ import annotations

import contextlib
import fcntl
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
os.environ.setdefault("OPENDECIDER_STATE_CACHE_MB", "0")          # OpenDecider backbones: no state-memory cache

# FineWeb-Edu sample (ODC-BY) used for calibration and evaluation text; any parquet with a "text" column works
DATA = os.environ.get("DEPTHSKIP_DATA", "/mnt/ext/mercurius/data/fineweb-edu/sample/10BT/000_00000.parquet")


def add_model_args(ap):
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", help="Hugging Face causal LM (local path or hub id), loaded with depthskip.Model")
    g.add_argument("--ckpt", help="OpenDecider checkpoint: run on its backbone (needs the opendecider package)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--prefill-chunk", type=int, default=0, help="--model only: prompt pass in slices of this size")


def load_backbone(a, serving: bool = False):
    """The model the experiments drive. serving (--ckpt only): OpenDecider's serving kernels first."""
    if a.ckpt:
        from opendecider.checkpoint import load_model
        model, _ = load_model(a.ckpt)
        if serving:
            from opendecider.decider import prepare_inference_
            prepare_inference_(model)
        return model.backbone
    from depthskip import Model
    return Model.from_pretrained(a.model, a.dtype, "cuda", a.prefill_chunk)


def model_name(a) -> str:
    """Tag for saved maps: the run directory of an OpenDecider checkpoint, else the model's basename."""
    return os.path.basename(os.path.dirname(a.ckpt)) if a.ckpt else os.path.basename(a.model.rstrip("/"))


def model_ref(a) -> str:
    return a.ckpt or a.model


@contextlib.contextmanager
def gpu_lock(tag: str):
    """Serialize GPU jobs on a shared machine: flock on DEPTHSKIP_GPU_LOCK (or OPENDECIDER_GPU_LOCK) if set, else a no-op.
    Re-entrant when DEPTHSKIP_GPU_LOCK_HELD / OPENDECIDER_GPU_LOCK_HELD is set (job launched under the lock)."""
    path = os.environ.get("DEPTHSKIP_GPU_LOCK") or os.environ.get("OPENDECIDER_GPU_LOCK")
    if not path or os.environ.get("DEPTHSKIP_GPU_LOCK_HELD") or os.environ.get("OPENDECIDER_GPU_LOCK_HELD"):
        yield
        return
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"[gpu_lock] {tag}: GPU busy, waiting", file=sys.stderr, flush=True)
            fcntl.flock(fd, fcntl.LOCK_EX)
        os.environ["DEPTHSKIP_GPU_LOCK_HELD"] = "1"
        try:
            yield
        finally:
            os.environ.pop("DEPTHSKIP_GPU_LOCK_HELD", None)
    finally:
        os.close(fd)


def timed(f, reps=3):
    f(); torch.cuda.synchronize()
    lat = []
    for _ in range(reps):
        t = time.perf_counter(); f(); torch.cuda.synchronize(); lat.append(time.perf_counter() - t)
    return float(np.median(lat))
