"""How many iterations does hash-MLP + LNCC need to nail the motion?

Oracle setting: the estimator is handed the TRUE image, so this measures the estimator alone, with
no coupling to a converging reconstruction. It is the ceiling any blind loop inherits.

THE QUESTION BEHIND THE QUESTION. A convergence curve that flattens tells you nothing on its own --
you cannot tell an OPTIMIZATION floor (needs more iterations, or a better lr) from a REPRESENTATION
floor (the band-limited hash grid simply cannot express this trajectory, so no number of iterations
will help). So the sweep runs two motions:

    sinusoid   smooth, and comfortably inside the encoder's bandwidth
    mixed      a different profile per DoF, including a STEP and a JERK -- deliberately outside it

If both plateau at the same error, it is optimization. If `mixed` plateaus higher, that gap IS the
band limit, and the fix is bandwidth (or a different parameterization), not patience.

A FITTED-BASELINE CONTROL. `fit_ceiling` reports the best the model class can do at all: the same
network fitted DIRECTLY to the true theta by regression, no projections involved. That separates
"the encoder cannot represent this" from "the projection loss cannot see it".

    python scripts/exp_est_convergence.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.dataset_slab import AAPMSlabGenerator
from fm3d.geometry_3d import ConeBeam3DConfig, measured_region_mask
from fm3d.motion_estimation import make_estimator
from fm3d.motion_net import MotionNet6DoF
from fm3d.reg_metric import aligned_metrics
from fm3d.rigid_motion import make_motion, motion_error, params_to_Pmot

DATA = "/home/mirlab/Desktop/Flow_matching_motion/data/AAPM_head_data"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "est_conv")


def fit_ceiling(theta_true, n_views, device, *, iters=3000, lr=1e-2, **net_kw):
    """Best the MotionNet can do on this trajectory when handed it directly. Representation floor.

    No projector, no data term -- plain regression of the network onto theta_true. Whatever error
    survives here is the encoder's band limit, and the projection-domain estimator cannot beat it.
    """
    net = MotionNet6DoF(n_views, **net_kw).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        # scale the two blocks to comparable magnitudes so neither dominates the regression
        p = net.all_params(device=device)
        loss = ((p[:, :3] - theta_true[:, :3]) ** 2).mean() + \
               100.0 * ((p[:, 3:] - theta_true[:, 3:]) ** 2).mean()
        loss.backward()
        opt.step()
    return motion_error(net.all_params(device=device).detach(), theta_true)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--max_iters", type=int, default=4000)
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--views_per_iter", type=int, default=24)
    ap.add_argument("--n_samples", type=int, default=384)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--kinds", default="sinusoid,mixed")
    ap.add_argument("--losses", default="lncc,l2si")
    args = ap.parse_args()

    dev = "cuda"
    os.makedirs(OUT, exist_ok=True)
    cfg = ConeBeam3DConfig(det_bin=2, n_views=360)
    gen = AAPMSlabGenerator(args.data, cfg, device=dev, slab=64, in_plane=256)
    spacing = (gen.dz, gen.dy, gen.dx)
    meas = measured_region_mask(gen.shape, spacing, cfg, device=dev)
    gt = gen.volume(0, 0)
    gt3 = gt[0, 0]
    print(f"grid {gen.shape} @ {spacing} mm | {cfg.n_views} views | "
          f"{args.views_per_iter} views/iter, n_samples {args.n_samples}, lr {args.lr}")

    hist = {}
    for kind in args.kinds.split(","):
        torch.manual_seed(args.seed)
        theta_true = make_motion(kind, cfg.n_views, device=dev, seed=args.seed)
        with torch.no_grad():
            y = gen.project(gt, params_to_Pmot(theta_true, gen.P_nom)[None])
            x_oracle = gen.fdk(y, params_to_Pmot(theta_true, gen.P_nom)[None])[0]
        m_or = aligned_metrics(x_oracle, gt3, spacing, mask=meas, iters=200)

        torch.manual_seed(args.seed)
        ceil = fit_ceiling(theta_true, cfg.n_views, dev)
        print(f"\n=== motion '{kind}' | true-theta FDK {m_or['psnr_aligned']:.2f} dB / "
              f"SSIM {m_or['ssim_aligned']:.3f}")
        print(f"    representation floor (net regressed straight onto theta_true): "
              f"rot {ceil['rot_rmse_deg']:.3f} deg, trans {ceil['trans_rmse_mm']:.3f} mm")

        for loss in args.losses.split(","):
            torch.manual_seed(args.seed)
            est = make_estimator("net", cfg, gen.P_nom, gen.u_coords, gen.v_coords, dev,
                                 dx=gen.dx, dy=gen.dy, dz=gen.dz, loss=loss,
                                 n_samples=args.n_samples,
                                 views_per_iter=args.views_per_iter, lr=args.lr)
            print(f"\n    {kind} / {loss}")
            print(f"    {'iters':>6} {'fit':>10} {'rot deg':>8} {'t_obs mm':>9} "
                  f"{'t_depth':>8} {'FDK dB':>7} {'SSIM':>6} {'s':>6}")
            rows = []
            t0 = time.time()
            done = 0
            while done < args.max_iters:
                n = min(args.every, args.max_iters - done)
                fit = est.refine_global(gt3, y[0], iters=n)     # warm-starts; params persist
                done += n
                th = est.current_params()
                me = motion_error(th, theta_true, cfg=cfg)
                with torch.no_grad():
                    xh = gen.fdk(y, params_to_Pmot(th, gen.P_nom)[None])[0]
                mm = aligned_metrics(xh, gt3, spacing, mask=meas, iters=120)
                el = time.time() - t0
                rows.append({"iters": done, "fit": fit, **me, **mm, "sec": el})
                print(f"    {done:6d} {fit:10.5f} {me['rot_rmse_deg']:8.3f} "
                      f"{me['trans_obs_mm']:9.3f} {me['trans_depth_mm']:8.3f} "
                      f"{mm['psnr_aligned']:7.2f} {mm['ssim_aligned']:6.3f} {el:6.0f}", flush=True)
            hist[f"{kind}/{loss}"] = rows
        hist[f"{kind}/_oracle"] = [{**m_or, **{"ceil_" + k: v for k, v in ceil.items()}}]

    with open(os.path.join(OUT, "conv.json"), "w") as f:
        json.dump(hist, f, indent=1)
    print(f"\nsaved -> {OUT}/conv.json")


if __name__ == "__main__":
    main()
