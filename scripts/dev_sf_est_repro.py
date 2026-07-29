"""Standalone repro of the SF-estimator in-loop divergence: oracle-reference fit, fine grid."""
import sys
import time

import torch

sys.path.insert(0, "/home/mirlab/Desktop/Flow_matching_motion_3D")

from fm3d.geometry_3d import ConeBeam3DConfig, build_conebeam_orbit, detector_coords_3d
from fm3d.motion_estimation import make_estimator
from fm3d.phantom import head_phantom
from fm3d.projector_3d import forward_project_3d_batched
from fm3d.rigid_motion import make_motion, motion_error, params_to_Pmot

DEV = "cuda"
torch.manual_seed(0)
cfg = ConeBeam3DConfig.thies(n_views=360)
P_nom = build_conebeam_orbit(cfg, device=DEV)
u, v = detector_coords_3d(cfg, device=DEV)
gt = head_phantom((256, 256, 256), (1.0, 1.0, 1.0), device=DEV)
th_true = make_motion("akima", 360, trans_mm=(10, 10, 10), rot_deg=(10, 10, 10), device=DEV, seed=3)
with torch.no_grad():
    y = forward_project_3d_batched(gt[None, None], params_to_Pmot(th_true, P_nom)[None], u, v,
                                   dx=1, dy=1, dz=1)[0]

# "ray" retired 2026-07-28 (forward_project routes to SF); historical anchor from the last
# two-arm run, same seed/GPU: ray reached loss 0.00197 / rot 3.37 / obs 0.55 mm at iter 300.
for backend in ("leap",):
    torch.manual_seed(0)
    est = make_estimator("net", cfg, P_nom, u, v, DEV, dx=1.0, dy=1.0, dz=1.0,
                         loss="l2", views_per_iter=24, lr=3e-3,
                         n_levels=16, base_resolution=16, per_level_scale=1.5)
    t0 = time.time()
    for it in range(6):
        loss = est.refine_global(gt, y, iters=50)
        me = motion_error(est.current_params(), th_true, cfg=cfg)
        print(f"{backend:4s} iter {(it+1)*50:4d} | loss {loss:.5f} | "
              f"rot {me['rot_rmse_deg']:.3f} deg obs {me['trans_obs_mm']:.3f} mm | "
              f"{time.time()-t0:.1f}s", flush=True)
