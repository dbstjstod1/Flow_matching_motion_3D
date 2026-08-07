"""How much theta accuracy does Thies' SSIM 0.94 actually require?

The akima 5/5 front rests on a claim -- "the estimator is the bottleneck" -- that is so far only
supported by a correlation: theta got 11x worse and the output SSIM fell 0.086. This script
measures the TRANSFER FUNCTION itself, which is what turns that claim into a target.

METHOD. Take a finished run's residual e = theta_hat - theta_true and reconstruct with

    theta(a) = theta_true + a * e ,   a = 0, 0.25, 0.5, 0.75, 1

so a=0 is the ORACLE FDK(theta_true) and a=1 reproduces the run's own output. Every point uses
the SAME measured sinogram and the SAME operator, so the only thing varying is theta -- and the
error keeps its REALISTIC SHAPE (our residual is not white: it is a smooth low-frequency drift
plus a global pose, see scripts/rpe_report.py), which a synthetic perturbation would not.

WHAT IT SETTLES.
  * The CEILING. a=0 is the best any motion estimator could do with this operator. If oracle
    FDK(theta_true) vs the static FDK is already below 0.94, then no estimator work reaches
    Thies' number and the gap is in the reconstruction protocol, not the motion.
  * The TARGET. Reading SSIM(a) back through rot(a) says what rotation error buys 0.94 -- i.e.
    whether we need 0.4 deg or 0.05 deg, which is the difference between tuning and redesigning.
  * The SHAPE. If SSIM is flat near a=0 and collapses late, there is slack; if it is steep
    everywhere, every 0.1 deg counts.

Scored on all four cells (output vs GT and vs static FDK) plus RPE, in one table.

    python scripts/exp_theta_transfer.py --from data/runs/akima55/thies_v0/result.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.dataset_cq500 import CQ500Generator
from fm3d.geometry_3d import ConeBeam3DConfig, measured_region_mask
from fm3d.reg_metric import aligned_metrics
from fm3d.rigid_motion import make_motion, motion_error, params_to_Pmot, reprojection_error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="logs/fm3d_cq500_leap/ckpt_iter500000.pth")
    ap.add_argument("--from", dest="src", default="data/runs/akima55/thies_v0/result.pt",
                    help="a finished run: supplies theta_hat (its residual is the error SHAPE)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--motion_kind", default="akima")
    ap.add_argument("--trans_mm", type=float, default=10.0)  # PEAK-TO-PEAK
    ap.add_argument("--rot_deg", type=float, default=10.0)   # PEAK-TO-PEAK
    ap.add_argument("--alphas", default="0,0.1,0.25,0.5,0.75,1.0")
    ap.add_argument("--out", default="data/runs/akima55/theta_transfer")
    args = ap.parse_args()

    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    ca = ck["args"]
    cfg = ConeBeam3DConfig.thies(n_views=ca["views"])
    gen = CQ500Generator(ca["root"], cfg=cfg, device=dev, split=args.split,
                         shape=tuple(ca["shape"]), voxel_mm=1.0)
    spacing = (gen.dz, gen.dy, gen.dx)
    meas = measured_region_mask(gen.shape, spacing, cfg, device=dev)

    gt = gen.volume(args.run)
    gt3 = gt[0, 0]
    th_true = make_motion(args.motion_kind, cfg.n_views, device=dev, seed=args.seed,
                          trans_mm=args.trans_mm, rot_deg=args.rot_deg)
    with torch.no_grad():
        y = gen.simulate(args.run, params_to_Pmot(th_true, gen.P_nom)[None])[0]
        # Thies' reference: FDK of the MOTION-FREE scan through the nominal orbit.
        static_fdk = gen.fdk(gen.simulate(args.run, gen.P_nom[None]), gen.P_nom[None])[0]

    r = torch.load(args.src, map_location=dev, weights_only=False)
    th_hat = r["theta"].to(dev).float()
    e = th_hat - th_true

    ms = aligned_metrics(static_fdk, gt3, spacing, mask=meas, iters=300)
    print(f"static FDK vs GT (the operator's own ceiling): {ms['psnr_aligned']:.2f} dB / "
          f"SSIM {ms['ssim_aligned']:.4f}\n")
    print(f"{'alpha':>6} {'rot deg':>8} {'obs mm':>7} {'RPE raw':>8} {'RPE gau':>8} | "
          f"{'OUT vs sFDK':>13} | {'OUT vs GT':>13}")

    rows = []
    for a in [float(x) for x in args.alphas.split(",")]:
        th_a = th_true + a * e
        with torch.no_grad():
            x_fdk = gen.fdk(y[None], params_to_Pmot(th_a, gen.P_nom)[None])[0]
        m_gt = aligned_metrics(x_fdk, gt3, spacing, mask=meas, iters=300)
        m_sf = aligned_metrics(x_fdk, static_fdk, spacing, mask=meas, iters=300)
        me = motion_error(th_a, th_true, cfg=cfg)
        rp = reprojection_error(th_a, th_true, gen.P_nom)
        print(f"{a:6.2f} {me['rot_rmse_deg']:8.3f} {me['trans_obs_mm']:7.3f} "
              f"{rp['rpe_mm']:8.3f} {rp['rpe_mm_gauged']:8.3f} | "
              f"{m_sf['psnr_aligned']:6.2f}/{m_sf['ssim_aligned']:.4f} | "
              f"{m_gt['psnr_aligned']:6.2f}/{m_gt['ssim_aligned']:.4f}", flush=True)
        rows.append({"alpha": a, **me, **rp, "out_vs_sfdk": m_sf, "out_vs_gt": m_gt})
        with open(os.path.join(args.out, f"transfer_run{args.run}.json"), "w") as f:
            json.dump({"static_fdk_vs_gt": ms, "rows": rows}, f, indent=1)
    print("\nThies: SSIM 0.94 vs the static FDK, RPE 0.61 mm. alpha=0 is the ceiling this "
          "operator can reach with PERFECT motion.")


if __name__ == "__main__":
    main()
