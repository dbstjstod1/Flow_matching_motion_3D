"""Is the residual damage under motion an FDK artefact, or is the information gone?

THE OBSERVATION THAT PROMPTS THIS (user, 2026-07-27). Even with PERFECT motion knowledge, FDK does
not reproduce the motion-free reconstruction: FDK(y_motion, P(theta_true)) scores 0.7935 SSIM /
36.27 dB against the static FDK, and its montage panel visibly carries streaks the static FDK does
not have. The hypothesis: FDK is an ANALYTIC inverse derived for a circular, equiangular orbit,
and arbitrary per-view rigid motion moves the source off that orbit RELATIVE TO THE OBJECT. No
angular re-weighting can undo that -- but an operator-based solve, which only ever uses the actual
A_theta, has no such assumption and should recover what FDK cannot.

(The strongest form of the mechanism -- patient rotation about the gantry axis running BACKWARDS,
so the effective view angle is non-monotonic and FDK's angular partition is undefined -- does NOT
occur at akima 5 mm / 5 deg: the effective spacing modulates between 0.69 and 1.22 deg/view but
never reverses. So if the hypothesis holds here it is the general non-circularity, not a gap.)

THE 2x2 THAT SEPARATES THE TWO EXPLANATIONS. Scoring FDK(theta_true) against CG(theta_true) alone
would confound "CG is a better reconstructor" with "motion hurts FDK specifically", so the same
pair is also run on the MOTION-FREE data:

                        FDK                         CG (same iterations)
    static data     the reference itself        does CG beat FDK with no motion at all?
    motion data     the oracle FDK ceiling      does CG recover what FDK lost?

If motion costs FDK much more than it costs CG, the damage is the operator's, not the data's.
All four are scored against the GT volume, which is the only reference neutral between them.

    python scripts/exp_fdk_vs_cg_oracle.py --cg_iters 60
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.dataset_cq500 import CQ500Generator
from fm3d.geometry_3d import ConeBeam3DConfig, measured_region_mask
from fm3d.reg_metric import aligned_metrics
from fm3d.rigid_motion import make_motion, params_to_Pmot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_posterior3d import cg_dc_step                                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="logs/fm3d_cq500_leap/ckpt_iter500000.pth")
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--cg_iters", type=int, default=60)
    args = ap.parse_args()

    dev = "cuda"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    ca = ck["args"]
    cfg = ConeBeam3DConfig.thies(n_views=ca["views"])
    gen = CQ500Generator(ca["root"], cfg=cfg, device=dev, split="val",
                         shape=tuple(ca["shape"]), voxel_mm=1.0)
    sp = (gen.dz, gen.dy, gen.dx)
    meas = measured_region_mask(gen.shape, sp, cfg, device=dev)
    gt = gen.volume(args.run)
    gt3 = gt[0, 0]
    th = make_motion("akima", cfg.n_views, device=dev, seed=args.seed, trans_mm=10.0, rot_deg=10.0)
    zero = torch.zeros_like(th)

    with torch.no_grad():
        y_mot = gen.simulate(args.run, params_to_Pmot(th, gen.P_nom)[None])[0]
        y_sta = gen.simulate(args.run, gen.P_nom[None])[0]

    print(f"val {args.run} seed {args.seed} | akima 5 mm / 5 deg | CG {args.cg_iters} iters, "
          f"warm-started at the matching FDK\n")
    print(f"{'':14} {'FDK':>16} {'CG':>16}   (aligned, vs GT)")
    out = {}
    for tag, yy, tt in (("static data", y_sta, zero), ("motion data", y_mot, th)):
        with torch.no_grad():
            fdk = gen.fdk(yy[None], params_to_Pmot(tt, gen.P_nom)[None])[0]
            cg = cg_dc_step(fdk.clone(), tt, yy, gen, iters=args.cg_iters, lam=0.0)
        mf = aligned_metrics(fdk, gt3, sp, mask=meas, iters=300)
        mc = aligned_metrics(cg, gt3, sp, mask=meas, iters=300)
        out[tag] = (mf, mc)
        print(f"{tag:14} {mf['psnr_aligned']:6.2f}/{mf['ssim_aligned']:.4f} "
              f"{mc['psnr_aligned']:6.2f}/{mc['ssim_aligned']:.4f}", flush=True)

    (sf, sc), (mf, mc) = out["static data"], out["motion data"]
    print(f"\nWHAT MOTION COSTS EACH OPERATOR (static -> motion, with PERFECT theta):")
    print(f"  FDK  {sf['psnr_aligned'] - mf['psnr_aligned']:+6.2f} dB  "
          f"{mf['ssim_aligned'] - sf['ssim_aligned']:+.4f} SSIM")
    print(f"  CG   {sc['psnr_aligned'] - mc['psnr_aligned']:+6.2f} dB  "
          f"{mc['ssim_aligned'] - sc['ssim_aligned']:+.4f} SSIM")
    print("A large FDK loss beside a small CG loss means the damage is the OPERATOR's assumption "
          "of a circular orbit,\nnot missing information -- i.e. the deliverable should not be an "
          "FDK.")


if __name__ == "__main__":
    main()
