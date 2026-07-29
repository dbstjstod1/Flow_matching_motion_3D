"""Smoke test on a real AAPM slab: does the whole 3D chain work before anything is trained?

Runs on virtual slab volumes stacked from contiguous AAPM head slices (see fm3d/dataset_slab.py
for why that stacking is not trivial -- the archive interleaves slices from other levels).

Three things, in order of what they would cost you to discover late:

  S1  THE SLAB IS A VALID OBJECT. Static FDK of a real head slab, not a phantom. If the FDK scale,
      the measured-region barrel or the spacing are wrong, this is where it shows.

  S2  MOTION ESTIMATION WORKS AT ALL -- given the GROUND-TRUTH image. This is an oracle: the
      posterior loop never has the true image. But it is the right thing to test first, because if
      the estimator cannot find the motion when handed a perfect image, no prior will rescue it,
      and every hour spent training would be wasted. It also compares the estimators and the data
      terms (l2si vs the projection-domain LNCC that AI_Geocal uses) on equal footing.

  S3  THE ORACLE CEILING. FDK under the recovered theta, scored the only way that means anything
      here -- rigidly aligned first, because the SE(3) gauge makes raw PSNR rank reconstructions
      wrong (fm3d/reg_metric.py). This number is the ceiling the blind loop is trying to reach.

    python scripts/smoke_slab.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.dataset_slab import AAPMSlabGenerator
from fm3d.geometry_3d import ConeBeam3DConfig, measured_region_mask
from fm3d.motion_estimation import make_estimator
from fm3d.reg_metric import aligned_metrics
from fm3d.rigid_motion import make_motion, motion_error, params_to_Pmot

DATA = "/home/mirlab/Desktop/Flow_matching_motion/data/AAPM_head_data"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "smoke")


def montage(path, panels, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D, H, _ = panels[0][1].shape
    zc, yc = D // 2, H // 2
    lo, hi = 0.0, 1.4 * 0.02
    n = len(panels)
    fig, ax = plt.subplots(2, n, figsize=(3.1 * n, 6.4))
    for c, (name, v) in enumerate(panels):
        ax[0, c].imshow(v[zc].cpu(), cmap="gray", vmin=lo, vmax=hi)
        ax[0, c].set_title(name, fontsize=9)
        ax[1, c].imshow(v[:, yc].cpu(), cmap="gray", vmin=lo, vmax=hi, aspect="auto")
        for r in range(2):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    ax[0, 0].set_ylabel("axial", fontsize=9)
    ax[1, 0].set_ylabel("coronal", fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--slab", type=int, default=64)
    ap.add_argument("--in_plane", type=int, default=256)
    ap.add_argument("--views", type=int, default=360)
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--views_per_iter", type=int, default=24)
    ap.add_argument("--motion_kind", default="mixed")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    dev = "cuda"
    os.makedirs(OUT, exist_ok=True)
    # Seed everything: the estimators draw random view subsets and the hash encoder is randomly
    # initialized, so without this the run-to-run spread is comparable to the differences the
    # table is trying to show.
    torch.manual_seed(args.seed)
    cfg = ConeBeam3DConfig(det_bin=2, n_views=args.views)

    t0 = time.time()
    gen = AAPMSlabGenerator(args.data, cfg, device=dev, slab=args.slab, in_plane=args.in_plane)
    print(f"S1  dataset ({time.time() - t0:.0f}s)")
    print(f"    runs {len(gen.runs)} (lengths {[int(b - a) for a, b in gen.runs]}) -> {gen.n_slabs} slabs")
    print(f"    grid {gen.shape} @ ({gen.dz}, {gen.dy}, {gen.dx}) mm | self-normalized FDK")
    print(f"    detector {cfg.nv}x{cfg.nu} @ {cfg.du:.3f} mm | FOV {cfg.fov_diameter_mm():.0f} mm"
          f" | axial {cfg.axial_coverage_mm():.0f} mm")

    spacing = (gen.dz, gen.dy, gen.dx)
    meas = measured_region_mask(gen.shape, spacing, cfg, device=dev)
    gt = gen.volume(args.run, 0)
    gt3 = gt[0, 0]

    theta_true = make_motion(args.motion_kind, cfg.n_views, device=dev, seed=args.seed)
    tr = theta_true[:, :3].abs().max().item()
    ro = torch.rad2deg(theta_true[:, 3:].norm(dim=-1).max()).item()
    print(f"    motion: |t| <= {tr:.1f} mm, |rot| <= {ro:.2f} deg")

    with torch.no_grad():
        y_static = gen.project(gt, gen.P_nom[None])
        y = gen.project(gt, params_to_Pmot(theta_true, gen.P_nom)[None])
        x_static = gen.fdk(y_static, gen.P_nom[None])[0]
        x_uncorr = gen.fdk(y, gen.P_nom[None])[0]
        x_oracle = gen.fdk(y, params_to_Pmot(theta_true, gen.P_nom)[None])[0]

    for name, v in [("static FDK", x_static), ("uncorrected", x_uncorr), ("true theta", x_oracle)]:
        m = aligned_metrics(v, gt3, spacing, mask=meas, iters=200)
        print(f"    {name:14s} aligned {m['psnr_aligned']:5.2f} dB / SSIM {m['ssim_aligned']:.3f}"
              f"   (raw {m['psnr_raw']:5.2f} dB, gauge {m['gauge_shift_mm']:.2f} mm)")

    # ---- S2: can the estimator find the motion, given the true image?
    print(f"\nS2  motion estimation from the GT image (oracle) -- {args.iters} iters, "
          f"{args.views_per_iter} views/iter")
    # READ `direct` WITH CARE -- the comparison is CONFOUNDED, and in its disfavour. With
    # views_per_iter=24 of 360, a free per-view parameter only receives a gradient on the ~1/15 of
    # iterations where its view is drawn, so after `iters` steps each of its parameters has been
    # updated ~iters/15 times; `net` and `basis` share parameters across views and update ALL of
    # them every single iteration. `direct` scoring badly here says it is starved, not that free
    # per-view parameterization is hopeless. (It is still the wrong choice for a different and
    # sounder reason -- V*6 unconstrained DoF against V projections -- but this table does not
    # show that.) Give it views_per_iter=None to compare it honestly.
    #
    # `trans` has the SE(3) GAUGE FITTED OUT (`rigid_motion.motion_error`), not merely
    # mean-subtracted: blind motion correction cannot observe a global rigid pose, so the raw
    # translation error measures that pose and not the estimator.
    print(f"    {'estimator':>10s} {'loss':>6s} | {'fit':>9s} | {'rot RMSE':>9s} "
          f"{'trans (gauge-fit)':>18s} | {'time':>6s}")
    results = {}
    for est_name, loss in [("net", "l2si"), ("net", "lncc"), ("basis", "l2si"), ("direct", "l2si")]:
        t1 = time.time()
        est = make_estimator(est_name, cfg, gen.P_nom, gen.u_coords, gen.v_coords, dev,
                             dx=gen.dx, dy=gen.dy, dz=gen.dz, loss=loss,
                             views_per_iter=args.views_per_iter,
                             lr=1e-2 if est_name == "net" else 0.3)
        fit = est.refine_global(gt3, y[0], iters=args.iters)
        th = est.current_params()
        me = motion_error(th, theta_true)
        results[f"{est_name}/{loss}"] = (th, me)
        print(f"    {est_name:>10s} {loss:>6s} | {fit:9.5f} | {me['rot_rmse_deg']:6.2f} deg "
              f"{me['trans_rmse_mm']:13.2f} mm | {time.time() - t1:5.0f}s", flush=True)

    # ---- S3: the ceiling a blind loop is chasing
    best = min(results, key=lambda k: results[k][1]["trans_rmse_mm"])
    th = results[best][0]
    with torch.no_grad():
        x_est = gen.fdk(y, params_to_Pmot(th, gen.P_nom)[None])[0]
    m = aligned_metrics(x_est, gt3, spacing, mask=meas, iters=300)
    print(f"\nS3  oracle ceiling, best estimator = {best}")
    print(f"    FDK(theta_hat) aligned {m['psnr_aligned']:5.2f} dB / SSIM {m['ssim_aligned']:.3f}")

    p = os.path.join(OUT, "smoke.png")
    montage(p, [("ground truth", gt3), ("static FDK", x_static), ("uncorrected", x_uncorr),
                ("true theta", x_oracle), (f"est ({best})", x_est)],
            f"AAPM slab {gen.shape} | oracle motion estimation")
    print(f"\n    montage -> {p}   (judge by eye)")


if __name__ == "__main__":
    main()
