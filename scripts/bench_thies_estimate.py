"""Thies' motion estimation: 100 plain gradient-descent steps on the frozen VIF* network.

    STAGE 2 OF 2. Stage 1 (`scripts/bench_thies_train_qm.py`) trains the quality-metric network
    this script freezes. It will refuse to run without one.

THE ALGORITHM, WHOLE (TMI II-C, Eq. 6, L347-380)
------------------------------------------------
    x^(0) = 0                                       # the identity geometry = the uncompensated scan
    repeat n = 0 .. 99:
        theta   = Akima(x, 30 nodes)                # motion model p
        P*      = P_nom @ T(theta)                  # L213-215
        I       = BACKPROJECT(g_filtered, P*)       # r, at 128^3 / 2 mm  (L505-507)
        f       = mean( QMNet(I) )                  # q, the frozen VIF* regressor
        x      -= s0 * t^n * df/dx                  # s0 = 100, t = 0.97
    reconstruct once more at 256^3 / 1 mm with the final x           # L507-508

There is NO outer loop, NO re-training, NO alternation: a single 100-step descent on a fixed
objective. (Our own posterior loop refreshes its target every step, which is why the fair unit of
comparison is our per-step estimator iterations against their 100 total -- see the Thies-method
note in the repo docs.)

THE PROBLEM INSTANCE IS *OURS*, BIT FOR BIT
-------------------------------------------
`run_posterior3d.build_world` is called with the same `--ckpt / --run / --seed / --trans_mm /
--rot_deg` our own loop uses, so this baseline is handed the identical patient, the identical
geometry, and the identical natively-simulated sinogram `y`. The checkpoint is read ONLY to
recover that world (dataset, grid, view count, simulation grid); the flow-matching prior inside
it is never evaluated here.

AMPLITUDE: 10 mm / 10 deg peak-to-peak by default = **2x the paper's own 5 mm / 5 deg
evaluation**, matching our deployed evaluation. `--thies_amp` restores 5/5 if you want to check
the reimplementation against the published RPE of 0.61 mm. bench/thies/PROVENANCE.md §3.

WHAT IS REPORTED, AND AGAINST WHICH REFERENCE
---------------------------------------------
Both, because they rank differently and only one of them is comparable to the paper:
  * vs the GT VOLUME          -- this repo's house metric
  * vs the motion-free THIES RECONSTRUCTION -- Thies' own reference ("we rigidly register all
    motion-compensated reconstructions to their respective ground truth reconstruction",
    L508-510), i.e. the number that is on the same footing as his SSIM 0.94
  * RPE, raw and gauge-quotiented (`fm3d.rigid_motion.reprojection_error`). Only `rpe_mm` (raw)
    is comparable to his 0.61 mm.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.thies.data import EVAL_AMP, THIES_EVAL_AMP                       # noqa: E402
from bench.thies.motion import ThiesSplineMotion                            # noqa: E402
from bench.thies.qmnet import QualityMetricUNet3D                           # noqa: E402
from bench.thies.recon import ThiesConeRecon, VolumeGrid, to_unit           # noqa: E402
from fm3d.reg_metric import aligned_metrics                                 # noqa: E402
from fm3d.rigid_motion import (params_to_Pmot, reprojection_error,          # noqa: E402
                               zero_centre_gauge)
from scripts.run_posterior3d import build_world                             # noqa: E402

DEFAULT_CKPT = "logs/fm3d_cq500_leap/ckpt_iter500000.pth"


def build_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # -- the frozen objective ---------------------------------------------------------------
    ap.add_argument("--qm", required=True,
                    help="quality-metric checkpoint from scripts/bench_thies_train_qm.py")

    # -- the world (identical flags to scripts/run_posterior3d.py) ---------------------------
    ap.add_argument("--ckpt", default=DEFAULT_CKPT,
                    help="our prior's checkpoint -- read ONLY to rebuild the same world "
                         "(dataset/grid/views/sim grid). The prior itself is not used here.")
    ap.add_argument("--root", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--run", type=int, default=0, help="patient index within the split")
    ap.add_argument("--seed", type=int, default=3, help="the motion draw's seed")
    ap.add_argument("--motion_kind", default="akima")
    ap.add_argument("--trans_mm", type=float, default=EVAL_AMP["trans_mm"])
    ap.add_argument("--rot_deg", type=float, default=EVAL_AMP["rot_deg"])
    ap.add_argument("--thies_amp", action="store_true",
                    help="use the PAPER's evaluation amplitude (5 mm / 5 deg p2p) instead of ours")

    # -- the reconstruction operator ----------------------------------------------------------
    ap.add_argument("--ramp", default="ramlak")
    ap.add_argument("--distance_weight", action="store_true",
                    help="A/B ONLY -- their Eq. 3 and their released kernel have no 1/w^2 term")
    ap.add_argument("--est_shape", type=int, default=128, help="TMI L505-507")
    ap.add_argument("--est_voxel_mm", type=float, default=2.0, help="TMI L505-507")
    ap.add_argument("--out_shape", type=int, default=256, help="TMI L507-508")
    ap.add_argument("--out_voxel_mm", type=float, default=1.0, help="TMI L507-508")

    # -- the optimizer, verbatim (Eq. 6 and L376-380) -----------------------------------------
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--s0", type=float, default=100.0)
    ap.add_argument("--decay", type=float, default=0.97)
    ap.add_argument("--est_nodes", type=int, default=30,
                    help="TMI L288-290: 30 nodes for the ESTIMATED motion (10 for the simulated)")

    # -- bookkeeping ---------------------------------------------------------------------------
    ap.add_argument("--out", default="data/bench_thies_val0")
    ap.add_argument("--montage_every", type=int, default=10,
                    help="0 disables. The repo's standing rule is that recon quality is judged "
                         "by eye, so this is on by default.")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args(argv)


def montage(path, panels, title, lo=0.0, hi=0.05):
    """Axial + coronal strip. Window is the mu equivalent of Thies' own Fig. 6 window
    (-1200 .. +1500 HU, caption L606), which is also this repo's montage window."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(panels)
    D, H, _ = panels[0][1].shape
    fig, ax = plt.subplots(2, n, figsize=(3.3 * n, 7.0), squeeze=False)
    for c, (name, v) in enumerate(panels):
        ax[0][c].imshow(v[D // 2].detach().cpu(), cmap="gray", vmin=lo, vmax=hi)
        ax[0][c].set_title(name, fontsize=8)
        ax[1][c].imshow(v[:, H // 2].detach().cpu(), cmap="gray", vmin=lo, vmax=hi,
                        aspect="auto")
        for r in range(2):
            ax[r][c].set_xticks([]); ax[r][c].set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    a = build_args(argv)
    if a.thies_amp:
        a.trans_mm, a.rot_deg = THIES_EVAL_AMP["trans_mm"], THIES_EVAL_AMP["rot_deg"]
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)

    # ---- 1. the world: identical to what scripts/run_posterior3d.py is handed ---------------
    W = build_world(ckpt=a.ckpt, dev=str(dev), root=a.root, split=a.split, run=a.run,
                    motion_kind=a.motion_kind, seed=a.seed,
                    trans_mm=a.trans_mm, rot_deg=a.rot_deg)
    cfg, gen, gt, theta_true, y = W["cfg"], W["gen"], W["gt3"], W["theta_true"], W["y"]
    P_nom = gen.P_nom
    print(f"[bench-thies] patient {a.run} of split '{a.split}' | motion "
          f"{a.trans_mm:g} mm / {a.rot_deg:g} deg p2p ({'PAPER' if a.thies_amp else 'OURS, 2x'})")

    # ---- 2. the frozen objective -------------------------------------------------------------
    ck = torch.load(a.qm, map_location=dev, weights_only=False)
    net = QualityMetricUNet3D(ck.get("f_maps"), norm=ck.get("norm", "none")).to(dev)
    net.load_state_dict(ck["model"])
    net.freeze()
    print(f"[bench-thies] quality metric: {a.qm} (iter {ck.get('iter')}) FROZEN")

    # ---- 3. Thies' reconstruction, and the one-time filtering --------------------------------
    recon = ThiesConeRecon(cfg, ramp_window=a.ramp, distance_weight=a.distance_weight)
    est_grid = VolumeGrid.centred(a.est_shape, a.est_voxel_mm)
    out_grid = VolumeGrid.centred(a.out_shape, a.out_voxel_mm)
    g_filt = recon.filter(y)                       # cosine + ramp, ONCE (L207-211)
    print(f"[bench-thies] filtered sinogram {tuple(g_filt.shape)} | est {est_grid.shape} "
          f"@ {a.est_voxel_mm:g} mm -> out {out_grid.shape} @ {a.out_voxel_mm:g} mm")

    # ---- 4. Eq. 6: 100 plain gradient-descent steps -------------------------------------------
    mot = ThiesSplineMotion(cfg.n_views, n_nodes=a.est_nodes, device=dev)
    hist = []
    t0 = time.time()
    for n in range(a.iters):
        vol = recon.backproject(g_filt, mot.Pmot(P_nom), est_grid)
        f = net.score(to_unit(vol)[None, None]).mean()
        f.backward()
        s = a.s0 * (a.decay ** n)
        x_before = mot.x.detach().clone()
        gnorm = mot.gd_step(s)
        if n == 0:
            # STEP-SIZE SANITY. s0 = 100 is calibrated on THEIR objective's scale (L376-380).
            # Our quality net is trained at a different motion amplitude on a different
            # reconstruction, so the gradient magnitude need not match theirs, and a step that
            # moves x by ~1e-4 mm is a 100-iteration no-op that will still finish and still
            # print a plausible-looking RPE. Say so loudly instead.
            dx = float((mot.x.detach() - x_before).abs().max())
            print(f"  [step size] iter 0: |grad| {gnorm:.4g}, s0 {a.s0:g} -> max |dx| "
                  f"{dx:.4g} (mm / deg)")
            if dx < 1e-3:
                print("  [step size] WARNING: the first step moves the motion parameters by "
                      f"{dx:.2g}, which is a no-op at this amplitude. Either the quality net is "
                      "undertrained or --s0 needs recalibrating for this objective's scale.")
            elif dx > 5.0:
                print("  [step size] WARNING: the first step is larger than the whole motion "
                      "amplitude; expect divergence. Lower --s0.")
        hist.append(dict(n=n, f=float(f.detach()), step=s, grad=gnorm))
        if n % 5 == 0 or n == a.iters - 1:
            # NOT under torch.no_grad(): `reprojection_error` fits the SE(3) gauge by gradient
            # descent internally, and its own docstring says so ("The gauge fit is itself an
            # optimization, so it must run OUTSIDE no_grad"). Wrapping it raises
            # "element 0 of tensors does not require grad". theta is detached, so nothing here
            # leaks into the estimator's graph.
            rpe = reprojection_error(mot.theta().detach(), theta_true, P_nom)["rpe_mm"]
            print(f"  it {n:3d}  f {float(f):.5f}  s {s:8.3f}  |grad| {gnorm:.4g}  "
                  f"RPE {rpe:.3f} mm  ({(time.time()-t0)/(n+1):.2f} s/it)", flush=True)
        if a.montage_every and (n % a.montage_every == 0 or n == a.iters - 1):
            montage(os.path.join(a.out, f"est_{n:03d}.png"),
                    [(f"I(x) @ it {n}", vol.detach())], f"Thies GD it {n}  f={float(f):.5f}")
    est_sec = time.time() - t0
    theta_hat = mot.theta().detach()
    print(f"[bench-thies] estimation done in {est_sec:.1f} s ({est_sec/a.iters:.2f} s/it)")

    # ---- 5. the deliverable: one 256^3 @ 1 mm reconstruction with the final estimate ----------
    with torch.no_grad():
        out_vol = recon.backproject(g_filt, params_to_Pmot(theta_hat, P_nom), out_grid)
        init_vol = recon.backproject(g_filt, P_nom, out_grid)                 # x = 0, uncorrected
        # Thies' OWN reference: the motion-free scan through the SAME operator.
        y_static = gen.simulate(a.run, P_nom[None])
        ref_vol = recon(y_static, P_nom, out_grid)
        del y_static

    # ---- 6. metrics ---------------------------------------------------------------------------
    # RPE in fp64: at the paper's 0.61 mm on a 0.64 mm pixel, the raw RPE is a difference of two
    # nearly-equal projected point sets and fp32 cancellation is a percent-level effect on the
    # very digit being compared (same reasoning as cmp_thies_vs_ours, which computes OUR side in
    # fp64 -- the two sides of the paired table must not differ in precision).
    sp = (a.out_voxel_mm,) * 3
    res = {"args": vars(a), "est_seconds": est_sec, "history": hist}
    th64, tt64, P64 = theta_hat.double(), theta_true.double(), P_nom.double()
    res["rpe"] = reprojection_error(th64, tt64, P64)
    res["rpe_zero_centred"] = reprojection_error(zero_centre_gauge(th64), tt64, P64)
    res["rpe_init"] = reprojection_error(torch.zeros_like(tt64), tt64, P64)

    # Per-DoF MAE of the motion parameters, the axes of the paper's Fig. 4 / Table I: mean |error|
    # over views, translations in mm and rotations in DEGREES, split by the plane the source
    # rotates in (our gantry rotates in xy): IN-plane = tx, ty, rz; OUT-of-plane = tz, rx, ry.
    # Raw (no gauge fit), which is the paper's own convention. Caveat for any cross-paper quote:
    # our rotation parameterization is axis-angle where the paper never says which; at <=5 deg the
    # difference from Euler angles is second order.
    err = (th64 - tt64).abs().mean(0)                                 # (6,)
    mae = dict(tx=float(err[0]), ty=float(err[1]), tz=float(err[2]),
               rx=float(torch.rad2deg(err[3])), ry=float(torch.rad2deg(err[4])),
               rz=float(torch.rad2deg(err[5])))
    mae["inplane_t_mm"] = 0.5 * (mae["tx"] + mae["ty"]); mae["inplane_r_deg"] = mae["rz"]
    mae["outplane_t_mm"] = mae["tz"]; mae["outplane_r_deg"] = 0.5 * (mae["rx"] + mae["ry"])
    res["mae"] = mae

    # Image metrics per reference: PSNR/SSIM/RMSE from aligned_metrics, plus VIF computed on the
    # ALIGNED volume (the paper registers before scoring, p.1103) -- with the standing caveat
    # that our VIF-P reads ~0.15 below the paper's VIF at the same state (PROVENANCE 4.10):
    # comparable across OUR methods, never against the paper's absolute VIF column.
    from bench.thies.vif import vif_scalar_3d                       # local: bench-only metric
    for tag, ref in (("vs_gt", gt), ("vs_thies_static", ref_vol)):
        res[tag] = {}
        for name, vol_ in (("output", out_vol), ("input", init_vol)):
            m, al = aligned_metrics(vol_, ref, sp, return_aligned=True)
            m["rmse_hu_raw"] = m["rmse_raw"] / 0.02 * 1000.0        # mu_water = 0.02 1/mm
            m["rmse_hu_aligned"] = m["rmse_aligned"] / 0.02 * 1000.0
            m["vif_aligned"] = float(vif_scalar_3d(to_unit(al)[None, None],
                                                   to_unit(ref)[None, None]))
            res[tag][name] = m
    print(json.dumps({k: v for k, v in res.items() if k not in ("args", "history")},
                     indent=1, default=float))

    # ---- 7. artifacts --------------------------------------------------------------------------
    montage(os.path.join(a.out, "final.png"),
            [("input (x=0, uncorrected)", init_vol),
             ("Thies output (x*)", out_vol),
             ("Thies static (their reference)", ref_vol),
             ("ground truth", gt)],
            f"Thies baseline | patient {a.run} | {a.trans_mm:g}/{a.rot_deg:g} p2p | "
            f"RPE {res['rpe']['rpe_mm']:.3f} mm")
    torch.save(dict(theta_hat=theta_hat.cpu(), theta_true=theta_true.cpu(),
                    x=mot.x.detach().cpu(), out_vol=out_vol.cpu()),
               os.path.join(a.out, "result.pt"))
    with open(os.path.join(a.out, "result.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"[bench-thies] wrote {a.out}/{{final.png,result.json,result.pt}}")


if __name__ == "__main__":
    main()
