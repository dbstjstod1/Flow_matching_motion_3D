"""Phase 1: tune the motion estimator ALONE, at EQUAL COST, outside the posterior loop.

WHY THIS CAN BE DONE SEPARATELY. Four of the knobs we need to set -- `estimator`, `per`,
`views_per_iter`, `lr` (and `loss`) -- affect exactly one thing: given a reference image, how
accurate is theta after some amount of work. None of them touches the prior, the data step or TV.
So they do not need the 70-minute loop; hand the estimator a FIXED reference image and measure
rot(work). That is ~20 minutes per config instead of 70, and it turns a 3^k grid into something
affordable.

AND `per` COMES OUT FREE. The estimator warm-starts across ODE steps, so the only quantity that
matters is the TOTAL iteration count N*per -- which is the x-axis of the curve this script draws.
There is nothing to search: read `per` off the knee.

ISO-COST IS THE WHOLE POINT. A cone-beam iteration costs `views_per_iter` view-projections, so
"more iterations" and "more views per iteration" are the same currency. Comparing configs at equal
ITERATIONS would just rediscover that more views is better. The budget here is therefore fixed in
VIEW-EVALUATIONS (default 60,000 = the deployed 24 views x 2500 iters = N50 x PER50), and each
config gets iters = budget // views_per_iter. A config with 4 views/iter gets 6x the iterations.

THE TWO ANCHORS this sweep bridges:

                       views/iter   lr      total iters   loss
    ours (deployed)        24       1e-2       2,500       l2si
    AI_Geocal              4        1e-3      12,000       lncc, kernel 31

AI_Geocal is where MotionNet6DoF comes from, so its settings are a real reference and not a guess
-- but it solves a DIFFERENT problem (a static calibration phantom, fitted once, offline), which
is why it can afford 12,000 iterations. The question this script answers is which of its choices
survive being moved inside an ODE loop at a fixed budget. NOTE its ENCODER settings deliberately
do NOT transfer: AI_Geocal uses stock Instant-NGP (n_levels 16, base_res 16, scale 1.5), and this
project's 2D sibling MEASURED that that is excess bandwidth for a few-hundred-view trajectory and
jitters -- `hashbl` (4 / 2 / 2.0) is the fix and stays.

REFERENCE IMAGE. `--ref` picks what the estimator is shown, because the answer may depend on it:
    cold   the uncorrected FDK (~23 dB) -- what step 0 of the loop hands it
    xt     the carried x_t from a finished run (~36 dB) -- what the late loop hands it
    gt     the ground truth -- the ORACLE ceiling, i.e. the best any loop could inherit
Run `cold` and `xt` at minimum: a config that only wins on a clean image is no use at step 0.

    python scripts/exp_est_sweep.py --ref xt --xt_from data/runs/akima55/thies_v0/result.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn.functional as F

from fm3d.dataset_cq500 import CQ500Generator
from fm3d.geometry_3d import ConeBeam3DConfig, detector_coords_3d
from fm3d.motion_estimation import make_estimator
from fm3d.rigid_motion import (make_motion, motion_error, params_to_Pmot,
                               trajectory_roughness)

# Each config: label, estimator, views_per_iter, lr, loss, lncc_win, extra estimator kwargs.
# Two suites, selectable with --suite:
#
# `oracle` (the user's Phase, 2026-07-24): l2si only, GT reference, FIXED iterations (2500 =
# the deployed N50 x PER50), because the question is "does PER=50 converge inside the loop's own
# budget, and is the deployed point SAFE" -- a loop-realistic reading, not an efficiency
# frontier. Axes: views UP from 24 (does averaging more rays per step buy accuracy?), lr around
# 1e-2 (incl. AI_Geocal's 1e-3 and a hotter 3e-2), and the encoder-bandwidth control the user
# asked for: `fullband` = AI_Geocal's stock Instant-NGP (n_levels 16, base_res 16, scale 1.5,
# the setting 2D measured as jittery excess bandwidth for a few-hundred-view trajectory) vs our
# band-limited hashbl (4/2/2.0) -- at BOTH lrs, so bandwidth is not confounded with lr.
#
# `isocost`: the original efficiency sweep (fewer views buys proportionally more iterations,
# lncc/basis controls included) -- kept for the follow-up question of where to SPEND.
FULLBAND = dict(n_levels=16, base_resolution=16, per_level_scale=1.5)   # stock NGP, AI_Geocal
# The 2D band-limited preset. Was the CONSTRUCTOR DEFAULT until 2026-08-07 (rows passing {}
# meant hashbl); the deployed encoder is now fullband, so hashbl rows must say so explicitly.
HASHBL = dict(n_levels=4, base_resolution=2, per_level_scale=2.0)
# Headroom on the net's tanh output bounds. The default is trans 15 mm / rot 8 deg, chosen when
# the simulated motion peaked at 3 mm / 2 deg (|tanh arg| ~ 0.25, gradient factor 1-tanh^2 ~ 0.94
# -- effectively linear). At the CORRECT akima 5/5 setting the measured peaks are 5.23 mm /
# 5.50 deg, so rotation sits at 5.5/8 = 0.69 and the gradient is attenuated to ~0.53: the
# estimator's effective lr is HALVED exactly where the trajectory is largest. `HEADROOM` restores
# the linear regime; it is a one-line structural suspect for the 11x theta collapse, so it is
# swept as its own axis rather than folded into lr.
HEADROOM = dict(trans_max_mm=25.0, rot_max_deg=15.0)
SUITES = {
    # ---- the akima 5mm/5deg re-tune (2026-07-26). Every other suite here was run at the old
    # `mixed` 3mm/2deg and is NOT evidence at this setting. Diagnosis this suite must settle:
    # the in-loop theta trace at 5/5 is still DESCENDING at the last ODE step (val0 3.51 -> 0.46
    # deg, val2 4.51 -> 1.34, both monotone), whereas at 3mm/2deg it was flat from step ~35 at
    # 0.08 deg. So the estimator is BUDGET-limited, not stuck -- which means the axes that matter
    # are the ones that buy convergence RATE (lr, headroom, views), and a GT-reference ceiling
    # tells us whether 2500 iterations can reach a good theta AT ALL.
    "akima55": [
        ("deployed",      "net", 24, 1e-3, "l2si", 9, FULLBAND),          # the baseline row
        ("hashbl_lr1e-2", "net", 24, 1e-2, "l2si", 9, HASHBL),                # the other anchor
        # --- lr on the DEPLOYED band (fullband); earlier lr sweeps only varied it on hashbl ---
        ("fb_lr3e-3",     "net", 24, 3e-3, "l2si", 9, FULLBAND),
        ("fb_lr1e-2",     "net", 24, 1e-2, "l2si", 9, FULLBAND),
        # --- tanh headroom, alone and combined with the hotter lr ---
        ("fb_hr",         "net", 24, 1e-3, "l2si", 9, {**FULLBAND, **HEADROOM}),
        ("fb_hr_lr3e-3",  "net", 24, 3e-3, "l2si", 9, {**FULLBAND, **HEADROOM}),
        # --- views per iteration (at FIXED iters these cost 2x / 0.5x; read with that in mind) --
        ("fb_views48",    "net", 48, 1e-3, "l2si", 9, FULLBAND),
        ("fb_views12",    "net", 12, 1e-3, "l2si", 9, FULLBAND),
        # --- loss, WITH ITS OWN LR (a fixed-lr lncc ablation froze theta once; see memory) ------
        ("fb_lncc_lr1e-3", "net", 24, 1e-3, "lncc", 9, FULLBAND),
        ("fb_lncc_lr1e-2", "net", 24, 1e-2, "lncc", 9, FULLBAND),
    ],
    # ---- IS THIES' ESTIMATOR BETTER THAN OURS? (user, 2026-07-27) -----------------------------
    # "Thies' way" differs from ours in TWO things at once -- the MODEL (an Akima/B-spline with 30
    # control points = 180 dof) and the OPTIMIZER (plain GD, step s0 = 100, exponential decay
    # 0.97, TMI 2025 II-C) -- so a head-to-head cannot attribute the result. This is the 2x2 that
    # can, plus two controls.
    #
    # THE HYPOTHESIS BEING TESTED (user's): a spline + GD REGULARIZES toward smoother motion, so
    # the FDK comes out smoother and better. `rot_rmse` cannot see that, which is why every row
    # here is also scored with `trajectory_roughness` -- our encoder carries ~7000 cells of
    # bandwidth across 360 views while the simulated trajectory is a 10-node spline, so we may
    # well be buying amplitude accuracy with jitter.
    #
    # NOTE ON BUDGET. Thies runs 100 GD iterations total against a FROZEN quality net. Our loop
    # grants 400/step x 50 steps. Comparing at HIS budget would answer a question nobody asked;
    # every row here gets OUR budget, and each candidate gets its own lr swept, because a step
    # size that is right for Adam is meaningless for GD (and vice versa).
    "struct": [
        # ours
        ("mlp_fb_adam",   "net", 24, 3e-3, "l2si", 9, dict(FULLBAND)),
        # our model, THEIR optimizer. Plain GD needs a step size 4-5 ORDERS larger than Adam's
        # here: the head is zero-initialized and the hash features are ~1e-4, so the raw gradient
        # is tiny and Adam's per-parameter normalization is what escapes it. At lr 0.1 and 1 the
        # net did not move AT ALL (rot stuck at its init 3.510, roughness exactly 0).
        ("mlp_fb_gd1e2",  "net", 24, 1e2, "l2si", 9, dict(FULLBAND, opt="gd", decay_reset="global")),
        ("mlp_fb_gd1e3",  "net", 24, 1e3, "l2si", 9, dict(FULLBAND, opt="gd", decay_reset="global")),
        ("mlp_fb_gd1e4",  "net", 24, 1e4, "l2si", 9, dict(FULLBAND, opt="gd", decay_reset="global")),
        # THEIR model (30 control points = Thies' 180 dof), our optimizer
        ("bspl30_adam.01", "basis", 24, 1e-2, "l2si", 9, dict(n_ctrl=30)),
        ("bspl30_adam.03", "basis", 24, 3e-2, "l2si", 9, dict(n_ctrl=30)),
        ("bspl30_adam.06", "basis", 24, 6e-2, "l2si", 9, dict(n_ctrl=30)),
        # THEIR model AND their optimizer. `decay` is BUDGET-MATCHED, not copied: Thies decays
        # 0.97 per iteration over 100 iterations = a 21x reduction end-to-end. Applying 0.97 to
        # our 2500 would kill the step size by iteration ~200, so the same 21x is spread over the
        # whole budget (0.047 ** (1/2500) = 0.99878). decay_reset="global" because this script
        # calls refine_global in chunks, and "call" would restart the schedule every chunk.
        ("bspl30_gd1",     "basis", 24, 1.0, "l2si", 9, dict(n_ctrl=30, opt="gd", decay_reset="global")),
        ("bspl30_gd3",     "basis", 24, 3.0, "l2si", 9, dict(n_ctrl=30, opt="gd", decay_reset="global")),
        ("bspl30_gd3_dec", "basis", 24, 3.0, "l2si", 9,
         dict(n_ctrl=30, opt="gd", decay=0.99878, decay_reset="global")),
        # controls on the BAND LIMIT: the simulator's own 10 nodes, twice Thies' dof, and none
        ("bspl10_adam",   "basis", 24, 3e-2, "l2si", 9, dict(n_ctrl=10)),
        ("bspl60_adam",   "basis", 24, 3e-2, "l2si", 9, dict(n_ctrl=60)),
        ("direct_adam",   "direct", 24, 3e-2, "l2si", 9, {}),
    ],
    # ---- Q1a: WHERE TO SPEND, at constant cost. Run with --budget (view-evaluations), NOT
    # --iters_per_config: a cone-beam iteration costs `views` view-projections, so views and
    # iterations are the same currency and only an iso-cost comparison can rank them. At 60,000
    # view-evals views=8 gets 7500 iterations and views=96 gets 625. lr is held at the value the
    # akima55 sweep chose (3e-3) because it was tuned at views=24; if the winner is far from 24
    # its lr needs re-checking before it is deployed.
    "views": [
        ("views8",   "net",  8, 3e-3, "l2si", 9, FULLBAND),
        ("views12",  "net", 12, 3e-3, "l2si", 9, FULLBAND),
        ("views24",  "net", 24, 3e-3, "l2si", 9, FULLBAND),
        ("views48",  "net", 48, 3e-3, "l2si", 9, FULLBAND),
        ("views96",  "net", 96, 3e-3, "l2si", 9, FULLBAND),
    ],
    # ---- IS `l2si` DOING ANYTHING? (user, 2026-07-28) -----------------------------------------
    # `l2si` came from the 2D project, where the flow-matching push and the data-residual update
    # disagreed about the image's overall brightness and a plain L2 sinogram term charged that
    # scale oscillation to the motion parameters. `scripts/exp_loss_l2_geom.py` measures whether
    # that failure mode exists here at all: the optimal scale c = <p,y>/<p,p> is 1.0007 +- 0.0001
    # at EVERY reference image the loop hands the estimator, and cos(grad_l2, grad_l2si) >= 0.997
    # with a norm ratio of 1.00. So the prediction is a DEAD TIE, and this suite is the empirical
    # check of that prediction -- with lr swept on BOTH losses, because a loss compared at one
    # fixed lr is a confound (see the lncc freeze in the memory).
    "loss": [
        ("l2si_lr1e-3",   "net", 24, 1e-3, "l2si", 9, FULLBAND),
        ("l2si_lr3e-3",   "net", 24, 3e-3, "l2si", 9, FULLBAND),   # the deployed point
        ("l2si_lr1e-2",   "net", 24, 1e-2, "l2si", 9, FULLBAND),
        ("l2_lr1e-3",     "net", 24, 1e-3, "l2",   9, FULLBAND),
        ("l2_lr3e-3",     "net", 24, 3e-3, "l2",   9, FULLBAND),
        ("l2_lr1e-2",     "net", 24, 1e-2, "l2",   9, FULLBAND),
    ],
    "oracle": [
        ("base24",        "net", 24, 1e-2, "l2si", 9, HASHBL),   # the deployed point
        # --- views axis, UP ---
        ("views48",       "net", 48, 1e-2, "l2si", 9, HASHBL),
        ("views96",       "net", 96, 1e-2, "l2si", 9, HASHBL),
        # --- lr axis ---
        ("lr3e-3",        "net", 24, 3e-3, "l2si", 9, HASHBL),
        ("lr3e-2",        "net", 24, 3e-2, "l2si", 9, HASHBL),
        ("lr1e-3",        "net", 24, 1e-3, "l2si", 9, HASHBL),
        # --- encoder bandwidth (user #4): band-UNlimited, at our lr and AI_Geocal's ---
        ("fullband",      "net", 24, 1e-2, "l2si", 9, FULLBAND),
        ("fullband_lr1e-3", "net", 24, 1e-3, "l2si", 9, FULLBAND),
    ],
    "isocost": [
        ("ours",          "net", 24, 1e-2, "l2si", 9, HASHBL),
        ("aigeocal",      "net",  4, 1e-3, "lncc", 31, HASHBL),
        ("views8",        "net",  8, 1e-2, "l2si", 9, HASHBL),
        ("views4",        "net",  4, 1e-2, "l2si", 9, HASHBL),
        ("lncc9",         "net", 24, 1e-2, "lncc", 9, HASHBL),
        ("lncc31",        "net", 24, 1e-2, "lncc", 31, HASHBL),
        ("views4_lr3e-3", "net",  4, 3e-3, "l2si", 9, HASHBL),
        ("basis",       "basis", 24,  0.3, "l2si", 9, {}),
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="logs/fm3d_cq500_leap/ckpt_iter500000.pth",
                    help="read ONLY for the geometry/dataset/scale -- the prior net is not used")
    ap.add_argument("--split", default="val")
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--seed", type=int, default=3)
    # akima 5/5 by default, matching train / val / inference / Thies since 2026-07-26. The old
    # `mixed` 3mm/2° default is what every pre-2026-07-26 sweep in this file ran at, and those
    # results are NOT evidence at the real setting -- theta degrades ~11x between the two.
    ap.add_argument("--motion_kind", default="akima")
    ap.add_argument("--trans_mm", type=float, default=10.0)  # PEAK-TO-PEAK
    ap.add_argument("--rot_deg", type=float, default=10.0)   # PEAK-TO-PEAK
    ap.add_argument("--ref", default="xt", choices=["cold", "xt", "gt"])
    ap.add_argument("--xt_from", default="data/runs/akima55/thies_v0/result.pt",
                    help="a finished run's result.pt, for --ref xt. Defaults to the CURRENT-"
                         "SETTING baseline (akima 5mm/5°); a legacy 3mm/2° x_t would hand the "
                         "estimator an easier reference than the loop will.")
    ap.add_argument("--suite", default="oracle", choices=list(SUITES))
    ap.add_argument("--budget", type=int, default=60_000,
                    help="VIEW-EVALUATIONS per config (24*2500 = the deployed N50 x PER50 cost); "
                         "used when --iters_per_config is 0")
    ap.add_argument("--iters_per_config", type=int, default=0,
                    help="if >0, EVERY config runs this many iterations regardless of its "
                         "views_per_iter (loop-realistic: the loop grants iterations, not "
                         "view-evals -- a views48 config then genuinely costs 2x). The oracle "
                         "suite wants 2500 = N50 x PER50.")
    ap.add_argument("--coarse", type=int, default=1,
                    help="estimate on a 1/N grid, as Thies does (he fits motion on 128^3 @ 2 mm "
                         "and only RECONSTRUCTS at 256^3 @ 1 mm). N halves nothing that matters "
                         "to a rigid trajectory but cuts the per-iteration cost ~N^3: the volume "
                         "is avg-pooled N (same 256 mm box, N mm voxels), the detector is binned "
                         "N (N^2 fewer rays), and the ray sampling is decimated N (the volume can "
                         "no longer resolve more). theta is reported against the FINE geometry "
                         "either way, so the numbers stay comparable to every other table.")
    ap.add_argument("--chunk", type=int, default=25, help="iterations between measurements")
    ap.add_argument("--only", default=None, help="comma-separated config labels to run")
    ap.add_argument("--out", default="data/runs/akima55/est_sweep")
    args = ap.parse_args()

    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    ca = ck["args"]
    cfg = ConeBeam3DConfig.thies(n_views=ca["views"])
    gen = CQ500Generator(ca["root"], cfg=cfg, device=dev, split=args.split,
                         shape=tuple(ca["shape"]), voxel_mm=1.0)

    gt = gen.volume(args.run)
    theta_true = make_motion(args.motion_kind, cfg.n_views, device=dev, seed=args.seed,
                             trans_mm=args.trans_mm, rot_deg=args.rot_deg)
    with torch.no_grad():
        y = gen.simulate(args.run, params_to_Pmot(theta_true, gen.P_nom)[None])[0]
    gt3 = gt[0, 0]

    if args.ref == "gt":
        ref = gt3
    elif args.ref == "cold":
        with torch.no_grad():
            ref = gen.fdk(y[None], gen.P_nom[None])[0]
    else:
        r = torch.load(args.xt_from, map_location="cpu", weights_only=False)
        if "x_t" not in r:
            raise SystemExit(f"{args.xt_from} has no saved x_t (run predates the volume-saving "
                             f"change) -- use --ref cold/gt, or point at a newer run")
        ref = r["x_t"].to(dev).float()
    # ---- the grid the ESTIMATOR works on. `--coarse 1` is our historical behaviour (everything
    # at 256^3 @ 1 mm); `--coarse 2` is Thies' (motion on 128^3 @ 2 mm). Reporting is unchanged:
    # theta is compared against theta_true through the FINE cfg in both cases.
    c = args.coarse
    if c > 1:
        est_cfg = ConeBeam3DConfig.thies(n_views=cfg.n_views, det_bin=cfg.det_bin * c)
        eu, ev = detector_coords_3d(est_cfg, device=dev)
        y_est = F.avg_pool2d(y[None], c)[0]                       # (V,nv,nu) -> binned panel
        ref_est = F.avg_pool3d(ref[None, None], c)[0, 0]          # 256^3 @1mm -> 128^3 @2mm
        vox = float(c) * gen.dx
    else:
        est_cfg, eu, ev = cfg, gen.u_coords, gen.v_coords
        y_est, ref_est = y, ref
        vox = gen.dx
    cost = (f"{args.iters_per_config} iters/config (loop-realistic)"
            if args.iters_per_config else f"budget {args.budget} view-evals (iso-cost)")
    print(f"suite = {args.suite} | reference = {args.ref} | {cost} | val {args.run} seed "
          f"{args.seed} | {cfg.n_views} views | estimation grid {tuple(ref_est.shape)} @ "
          f"{vox:g} mm, panel {est_cfg.nv}x{est_cfg.nu} @ {est_cfg.du:g} mm",
          flush=True)

    tag = f"{args.suite}_{args.ref}" + (f"_c{c}" if c > 1 else "")
    want = set(args.only.split(",")) if args.only else None
    rows = {}
    for label, est_name, views, lr, loss, win, net_kw in SUITES[args.suite]:
        if want and label not in want:
            continue
        iters = args.iters_per_config or max(1, args.budget // views)
        torch.manual_seed(args.seed)          # same init for every config -- see run_posterior3d
        est = make_estimator(est_name, est_cfg, gen.P_nom, eu, ev, dev,
                             dx=vox, dy=vox, dz=vox, loss=loss, lncc_win=win,
                             views_per_iter=views, lr=lr, **net_kw)
        curve, done, t0 = [], 0, time.time()
        while done < iters:
            n = min(args.chunk, iters - done)
            fit = est.refine_global(ref_est, y_est, iters=n)
            done += n
            th_hat = est.current_params()
            me = motion_error(th_hat, theta_true, cfg=cfg)
            # roughness is the axis rot_rmse is blind to -- see trajectory_roughness
            curve.append({"iters": done, "fit": fit, **me,
                          **trajectory_roughness(th_hat, theta_true)})
        el = time.time() - t0
        best = min(curve, key=lambda c: c["rot_rmse_deg"])
        rows[label] = {"config": {"estimator": est_name, "views_per_iter": views, "lr": lr,
                                  "loss": loss, "lncc_win": win, "iters": iters, **net_kw},
                       "curve": curve, "secs": el}
        c = curve[-1]
        print(f"{label:16} lr {lr:<6g} | {iters:6d} it | FINAL rot {c['rot_rmse_deg']:.3f} deg "
              f"obs {c['trans_obs_mm']:.3f} mm | BEST rot {best['rot_rmse_deg']:.3f}"
              f"@{best['iters']} | rough rot {c['rough_rot_deg']:.4f} "
              f"(x{c['rough_rot_ratio']:.1f} truth) hf {100*c['hf_frac']:.1f}% "
              f"| {el/60:.1f} min", flush=True)
        with open(os.path.join(args.out, f"sweep_{tag}.json"), "w") as f:
            json.dump(rows, f, indent=1)

    # ---- one figure, and EVERY curve labelled with its own final number (a legend of bare
    # config names cannot be read against the curves at this density)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for label, r in rows.items():
        c = r["curve"]
        it = [q["iters"] for q in c]
        lab = f"{label} ({c[-1]['rot_rmse_deg']:.3f} deg)"
        ax[0].plot(it, [q["rot_rmse_deg"] for q in c], label=lab, lw=1.4)
        ax[1].plot(it, [q["trans_obs_mm"] for q in c],
                   label=f"{label} ({c[-1]['trans_obs_mm']:.3f} mm)", lw=1.4)
    xlab = (f"estimator iterations (FIXED {args.iters_per_config}/config; views48/96 cost 2x/4x)"
            if args.iters_per_config else
            f"estimator iterations (ISO-COST: all configs spent {args.budget} view-evals)")
    for a, t, u in ((ax[0], "rotation RMSE", "deg"), (ax[1], "observable translation", "mm")):
        a.set_xlabel(xlab, fontsize=8)
        a.set_ylabel(u); a.set_title(f"{t} -- suite {args.suite}, reference = {args.ref}",
                                     fontsize=10)
        a.set_xscale("log"); a.set_yscale("log"); a.grid(alpha=.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(args.out, f"sweep_{tag}.png")
    fig.savefig(p, dpi=140)
    print(f"-> {p}")


if __name__ == "__main__":
    main()
