"""A/B the two patch->volume schemes of the FM prior, on the prior-ONLY ODE.

`predict_x1_patched` implements two families (see fm3d/prior_patch.py):

    hann     FAMILY A (ours)  -- OVERLAPPING patch^3 tiles at stride patch/2, combined with a
                                 raised-Hann window normalized to 1. The window kills the seam
                                 directly. At 256^3 / patch 32 that is 15^3 = 3375 tiles per grid.
    uniform  FAMILY B         -- the archive paper's scheme (arXiv:2512.18161 / DiffusionBlend):
                                 NON-overlapping tiles (stride = patch) at K FULLY RANDOM phases,
                                 averaged with a box window. Each pass lays its seams somewhere
                                 different and the average over K tilings IS the blend. 8^3 = 512
                                 tiles per grid, so K=2 costs 1024 -- 3.3x cheaper than family A.
                                 Meaningless at K=1 (a single non-overlapping pass, seams intact),
                                 which is exactly why it is included here as the control.

The score is the SAME quantity the training validation reports (val_fm3d.run_validation): cold
start = FDK of a motion-corrupted scan, 50-step prior-only Euler ODE, no data term, no TV, then
rigid-align-then-metric against the static FDK. Everything except `blend`/`n_offsets` is held
fixed -- same patients, same motion seeds (1000+i), same checkpoint, ONE process so the cold
starts come out of the shared _VAL_CACHE and are bit-identical across configs.

    CUDA_VISIBLE_DEVICES=0 python scripts/exp_blend_compare.py \
        --ckpt logs/fm3d_cq500_leap/ckpt_iter500000.pth
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.dataset_cq500 import CQ500Generator
from fm3d.geometry_3d import ConeBeam3DConfig, measured_region_mask
from fm3d.rigid_motion import amp_from_run_args
from fm3d.unet_3d import UNet3D
from val_fm3d import run_validation

# (tag, blend, n_offsets, note) -- CHEAPEST FIRST, so the core answer lands before the long run.
CONFIGS = [
    ("uniB_K1", "uniform", 1, "family B, single non-overlapping pass (control: seams intact)"),
    ("uniB_K2", "uniform", 2, "family B, K=2 random tilings -- the archive paper's setting"),
    ("hannA_K1", "hann", 1, "family A, overlapping stride patch/2 + Hann (OUR CURRENT DEFAULT)"),
    ("hannA_K2", "hann", 2, "family A + one extra jittered grid"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="logs/fm3d_cq500_leap/ckpt_iter500000.pth")
    ap.add_argument("--root", default=None, help="default: the checkpoint's --root")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="data/blend_cmp")
    ap.add_argument("--patients", type=int, default=3)
    ap.add_argument("--ode_steps", type=int, default=50)
    ap.add_argument("--tile_batch", type=int, default=64)
    ap.add_argument("--only", default=None, help="comma-separated subset of the config tags")
    args = ap.parse_args()

    dev = "cuda"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    ca = ck["args"]
    it = ck.get("iter", 0)

    cfg = ConeBeam3DConfig.thies(n_views=ca["views"])
    gen = CQ500Generator(args.root or ca["root"], cfg=cfg, device=dev, split=args.split,
                         shape=tuple(ca["shape"]), voxel_mm=1.0, verbose=True)
    meas = measured_region_mask(gen.shape, (gen.dz, gen.dy, gen.dx), cfg, device=dev)

    in_ch = int(ck["ema"]["in_conv.weight"].shape[1])
    model = UNet3D(in_ch=in_ch, base=ca["base"]).to(dev).eval()
    model.load_state_dict(ck["ema"])
    for q in model.parameters():
        q.requires_grad_(False)
    print(f"ckpt {args.ckpt} @ iter {it} | in_ch {in_ch} base {ca['base']} patch {ca['patch']} | "
          f"{args.patients} patients x {args.ode_steps} ODE steps", flush=True)

    keep = set(args.only.split(",")) if args.only else None
    results = {}
    for tag, blend, k, note in CONFIGS:
        if keep and tag not in keep:
            continue
        out = os.path.join(args.out, tag)
        os.makedirs(out, exist_ok=True)
        print(f"\n=== {tag}: blend={blend} n_offsets={k} | {note}", flush=True)
        t0 = time.time()
        _, mean, rows = run_validation(
            model, gen, meas, out, it=it, patients=args.patients, patch=ca["patch"],
            ode_steps=args.ode_steps, anchor=ca.get("anchor", "static"),
            trans_mm=(amp_from_run_args(ca)[0] or 10.0), rot_deg=(amp_from_run_args(ca)[1] or 10.0),
            blend=blend, n_offsets=k, tile_batch=args.tile_batch, dev=dev)
        dt = time.time() - t0
        results[tag] = {
            "blend": blend, "n_offsets": k, "note": note, "seconds": dt, "rows": rows,
            "psnr": mean,
            "ssim": float(np.mean([r["ode_ssim_aligned"] for r in rows])),
            "psnr_raw": float(np.mean([r["ode_psnr_raw"] for r in rows])),
        }
        print(f"=== {tag}: {mean:.3f} dB / SSIM {results[tag]['ssim']:.4f}  "
              f"[{dt / 60:.1f} min]", flush=True)
        with open(os.path.join(args.out, "results.json"), "w") as f:
            json.dump({"ckpt": args.ckpt, "iter": it, "results": results}, f, indent=2)

    cold_p = float(np.mean([r["cold_psnr_aligned"] for r in rows]))
    cold_s = float(np.mean([r["cold_ssim_aligned"] for r in rows]))
    print(f"\n{'config':10s} {'blend':8s} {'K':>2s} {'tiles/step':>10s} "
          f"{'PSNR':>7s} {'SSIM':>7s} {'min':>6s}")
    print(f"{'cold FDK':10s} {'-':8s} {'-':>2s} {'-':>10s} {cold_p:7.2f} {cold_s:7.4f} {'-':>6s}")
    p = ca["patch"]
    D = gen.shape[0]
    for tag, r in results.items():
        n = (len(range(0, D - p + 1, p)) ** 3 if r["blend"] == "uniform"
             else (len(range(0, D - p + 1, p // 2)) + 1) ** 3) * r["n_offsets"]
        print(f"{tag:10s} {r['blend']:8s} {r['n_offsets']:2d} {n:10d} "
              f"{r['psnr']:7.2f} {r['ssim']:7.4f} {r['seconds'] / 60:6.1f}")
    print(f"\n-> {args.out}/results.json  (montages in {args.out}/<tag>/)")


if __name__ == "__main__":
    main()
