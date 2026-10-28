"""GPU check of skipkv's cache-only layer paths on a real backbone: with the map predicting the true input (cut at the
last layer, or at the one before, comparing only that layer), the skipped state pass must write the same cache as the
exact pass. Covers whatever mixers the checkpoint has (stock Qwen3.5, Mercurius MLA + GDN-2).

    python experiments/check_exact.py --ckpt runs/x4b-mercurius-lora/model.pt
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import add_model_args, gpu_lock, load_backbone  # noqa: E402
from depthskip.model import cache_tensors  # noqa: E402
from depthskip.maps import SkipMaps, _merc, _stock  # noqa: E402

TEXT = ("The customer wrote on Monday that the blender arrived damaged and asked for a refund. An agent replied the "
        "same day with a return label. On Wednesday the customer said the label did not work. ") * 6


def main():
    ap = argparse.ArgumentParser()
    add_model_args(ap)
    a = ap.parse_args()
    with gpu_lock("skipkv_check"):
        bb = load_backbone(a)
        L, D = len(bb.lm.layers), bb.hidden_size
        seqs = bb.tokenize([TEXT, TEXT[:300]])
        exact, _ = bb.prefix_cache_batch(seqs, quantize=False)
        ok = True
        for cut in (L - 1, L - 2):
            layer = bb.lm.layers[cut]
            mixer = layer.linear_attn if hasattr(layer, "linear_attn") else layer.self_attn
            path = ("stock" if _stock(mixer) else "mercurius" if (_merc(mixer, "mercurius_port", "_forward") or
                    _merc(mixer, "mercurius_cache", "_compressed_forward")) else "generic")
            U = L - cut
            bb.skip_state = SkipMaps(cut, "input", torch.eye(D, device=bb.device, dtype=torch.bfloat16).repeat(U, 1, 1),
                                     torch.zeros(U, D, device=bb.device, dtype=torch.bfloat16))
            try:
                skip, _ = bb.prefix_cache_batch(seqs, quantize=False)
            finally:
                bb.skip_state = None
            for k, t in cache_tensors(exact).items():
                if k[0] != cut:
                    continue
                u = cache_tensors(skip)[k]
                if t.dtype in (torch.uint8, torch.int8, torch.int32, torch.int64):
                    d = (t != u).float().mean().item()
                    print(f"layer {cut} ({type(mixer).__name__}, {path}) {k[1]}: share of codes differing {d:.2e}")
                    ok &= d < 1e-3
                else:
                    d = ((t.float() - u.float()).norm() / t.float().norm().clamp(min=1e-12)).item()
                    print(f"layer {cut} ({type(mixer).__name__}, {path}) {k[1]}: relative difference {d:.2e}")
                    ok &= d < 1e-2
        print("CHECK", "OK" if ok else "FAILED", flush=True)


if __name__ == "__main__":
    main()
