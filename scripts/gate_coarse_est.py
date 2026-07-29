"""Gate: the COARSE estimation grid (Thies' 128^3 @ 2 mm) is the same physics as the fine one.

`exp_est_sweep.py --coarse N` (and, if it wins, the posterior loop) hands the estimator a
downsampled world: the volume avg-pooled by N, the detector binned by N, the ray sampling
decimated by N -- while REUSING `gen.P_nom`. That reuse is an assumption: it is only valid if the
projection matrices map the world to PHYSICAL detector millimetres (so binning changes only
`u_coords`/`v_coords`), and not to pixel indices (where binning would silently rescale the
geometry by N and every theta would come out wrong by a factor).

Nothing downstream would fail loudly if that assumption were wrong -- the estimator would simply
fit a mis-scaled trajectory and report a plausible-looking rot error -- so it is gated here.

Checks, on a real CQ500 volume with a real Akima 5 mm / 5 deg trajectory:
  1. FORWARD CONSISTENCY. A(coarse volume, binned panel) ~= bin(A(fine volume, fine panel)).
     Both are line integrals of the same object through the same rays, so they must agree to
     within the resampling error of a 2 mm grid (a few percent of the signal RMS, not 2x).
  2. NO SILENT SCALE. The same check must FAIL loudly if P were pixel-indexed -- verified by
     re-running it with a deliberately mis-scaled panel, which must blow the tolerance.
  3. GRADIENT DIRECTION. One estimator step on the coarse grid must reduce theta error from a
     perturbed start, i.e. the coarse loss still points at the right trajectory.

    python scripts/gate_coarse_est.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.dataset_cq500 import CQ500Generator
from fm3d.geometry_3d import ConeBeam3DConfig, detector_coords_3d
from fm3d.motion_estimation import make_estimator
from fm3d.projector_3d import forward_project_3d_batched
from fm3d.rigid_motion import make_motion, motion_error, params_to_Pmot

CKPT = "logs/fm3d_cq500/ckpt_iter500000.pth"
COARSE = 2
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}", flush=True)


def main():
    dev = "cuda"
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    ca = ck["args"]
    cfg = ConeBeam3DConfig.thies(n_views=ca["views"])
    gen = CQ500Generator(ca["root"], cfg=cfg, device=dev, split="val",
                         shape=tuple(ca["shape"]), voxel_mm=1.0)
    gt = gen.volume(0)
    theta = make_motion("akima", cfg.n_views, device=dev, seed=3, trans_mm=10.0, rot_deg=10.0)
    P = params_to_Pmot(theta, gen.P_nom)[None]

    with torch.no_grad():
        y_fine = gen.project(gt, P)[0]                                    # (V, nv, nu)

    # ---- 1. forward consistency ---------------------------------------------------------
    ccfg = ConeBeam3DConfig.thies(n_views=cfg.n_views, det_bin=cfg.det_bin * COARSE)
    cu, cv = detector_coords_3d(ccfg, device=dev)
    vol_c = F.avg_pool3d(gt, COARSE)                                      # (1,1,128,128,128)
    with torch.no_grad():
        y_coarse = forward_project_3d_batched(
            vol_c, P, cu, cv, dx=COARSE * gen.dx, dy=COARSE * gen.dy, dz=COARSE * gen.dz)[0]
    y_binned = F.avg_pool2d(y_fine[None], COARSE)[0]

    print(f"fine {tuple(y_fine.shape)} -> binned {tuple(y_binned.shape)} | "
          f"coarse projection {tuple(y_coarse.shape)} | volume {tuple(vol_c.shape[2:])} @ "
          f"{COARSE * gen.dx:g} mm")
    check("shapes match", y_coarse.shape == y_binned.shape,
          f"{tuple(y_coarse.shape)} vs {tuple(y_binned.shape)}")
    rms = y_binned.pow(2).mean().sqrt()
    err = (y_coarse - y_binned).pow(2).mean().sqrt() / rms
    scale = (y_coarse * y_binned).sum() / y_binned.pow(2).sum().clamp_min(1e-12)
    check("relative RMS error < 5%", err < 0.05, f"{100 * err:.2f}%")
    check("no scale factor (0.98..1.02)", 0.98 < scale < 1.02, f"best-fit scale {scale:.4f}")

    # ---- 2. the same check must CATCH a mis-scaled panel ---------------------------------
    with torch.no_grad():
        y_bad = forward_project_3d_batched(
            vol_c, P, cu * 1.10, cv * 1.10, dx=COARSE * gen.dx, dy=COARSE * gen.dy,
            dz=COARSE * gen.dz)[0]
    err_bad = (y_bad - y_binned).pow(2).mean().sqrt() / rms
    check("a 10% panel mis-scale is caught", err_bad > 0.05,
          f"{100 * err_bad:.2f}% (vs {100 * err:.2f}% correct)")

    # ---- 3. the coarse loss still points the right way -----------------------------------
    ref_c = vol_c[0, 0]
    y_meas_c = y_binned
    torch.manual_seed(3)
    est = make_estimator("net", ccfg, gen.P_nom, cu, cv, dev,
                         dx=COARSE * gen.dx, dy=COARSE * gen.dy, dz=COARSE * gen.dz,
                         loss="l2si", views_per_iter=24, lr=1e-3,
                         n_levels=16, base_resolution=16, per_level_scale=1.5)
    e0 = motion_error(est.current_params(), theta, cfg=cfg)
    est.refine_global(ref_c, y_meas_c, iters=200)
    e1 = motion_error(est.current_params(), theta, cfg=cfg)
    check("200 coarse iters reduce rot error", e1["rot_rmse_deg"] < e0["rot_rmse_deg"],
          f"{e0['rot_rmse_deg']:.3f} -> {e1['rot_rmse_deg']:.3f} deg")
    check("200 coarse iters reduce observable translation",
          e1["trans_obs_mm"] < e0["trans_obs_mm"],
          f"{e0['trans_obs_mm']:.3f} -> {e1['trans_obs_mm']:.3f} mm")

    print("\nGATE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
