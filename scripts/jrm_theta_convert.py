"""Convert JRM-ADM's estimated motion into OUR theta convention (A2d, 2026-08-18).

THEIR representation (rigid_motion_theta.py + physics.RigidWarper): per view k a (3,4) affine
A_k = [R_e | t_raw] consumed by F.affine_grid AFTER the warper rescales the translation column
by 2/(N-1) per axis -- so A_k maps OUTPUT grid_sample coords to INPUT sampling coords,
p_in = R_e p_out + t_norm, with t_raw in VOXELS (= mm at 1 mm) and coords ordered (x, y, z) =
(W, H, D). On a CUBIC grid (our port runs 224^3 -- this is load-bearing) the normalized frame
is an isotropic scaling of centered mm coords, so the same map holds in mm:

    sampling map   S_k(q) = R_e q + t_raw            [centered mm, (x,y,z)]
    patient motion M_k    = S_k^{-1}: q -> R_e^T (q - t_raw)

(a warp that SAMPLES at S moves the content by S^{-1}). NOTE their warper is genuinely rigid
in mm ONLY on cubic grids: at their native 160x192x192 the per-axis normalization makes any
z-mixing "rotation" a shear in mm. Our port sidesteps that; do not reuse this converter for
non-cubic runs.

OUR representation (fm3d.rigid_motion): y = A_{P_nom @ T(theta)}(x) is the static projection
of the patient moved by T(theta) = [[R, t],[0,1]] about the volume centre, theta =
[tx,ty,tz mm | axis-angle]. The operator cross-check (diag_jrm_operator_xcheck, rel 0.0016)
plus the +90 deg angle feed make our view k and their view k the SAME source position and the
same world axes, so the conversion is simply T(theta_k) = M_k:

    R_ours = R_e^T          t_ours = -R_e^T t_raw

Gauge: their recon (and hence their theta-hat) sits at an arbitrary global rigid offset; FDK
built from the converted theta inherits it, and the rigid-aligned metrics absorb it exactly
as they do for every other arm. Validated end-to-end by scripts/gate_jrm_theta_convert.py
(their own data-term projection vs ours under the converted theta).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.rigid_motion import so3_log


def jrm_thetas_to_ours(thetas_est: torch.Tensor) -> torch.Tensor:
    """(V, 3, 4) their affine (raw, voxel translations) -> (V, 6) our [t | axis-angle]."""
    if thetas_est.ndim != 3 or thetas_est.shape[-2:] != (3, 4):
        raise ValueError(f"expected (V,3,4), got {tuple(thetas_est.shape)}")
    R_e = thetas_est[:, :, :3]
    t_raw = thetas_est[:, :, 3]
    R = R_e.transpose(-1, -2)
    t = -torch.einsum("vij,vj->vi", R, t_raw)
    w = so3_log(R)
    return torch.cat([t, w], dim=-1)
