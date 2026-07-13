"""Gauge-aware evaluation: rigidly align, THEN measure.

Blind motion correction has an EXACT SE(3) GAUGE. Move the object by a rigid G and compose every
view's motion with G^-1 and the measurements are unchanged:

    A(G x ; P_nom @ T(theta)) == A(x ; P_nom @ T(theta) @ G)

so the data cannot distinguish `x` from `G x`. The solution is an equivalence class, not a point.
Absolute pose is recoverable only from an external anchor (a couch, a fiducial, a prior scan).

The consequence is practical and it bites: a reconstruction can be perfect up to a 5-pixel rigid
shift and score terribly on raw PSNR, while a blurrier one that happens to sit at the right pose
scores better. Raw PSNR ranks the wrong reconstruction. Every headline number in this project
therefore goes through a rigid alignment first, which quotients the gauge out.

(Distinct from ordinary ill-conditioning: the gauge is an exact, structural invariance, not a
badly-observed direction.)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .rigid_motion import so3_exp


def _affine_from_theta(theta: torch.Tensor, shape, spacing) -> torch.Tensor:
    """(6,) [mm | rad] -> (1,3,4) affine_grid matrix in NORMALIZED coords.

    affine_grid works in [-1,1] on each axis, so the physical rotation has to be conjugated by
    the anisotropic voxel scaling: a rotation in mm is not a rotation in normalized units unless
    the voxels happen to be cubic. `S` carries that.
    """
    D, H, W = shape
    dz, dy, dx = spacing
    R = so3_exp(theta[None, 3:])[0]                                   # (3,3) world (x,y,z)
    t = theta[:3]

    # normalized half-extents, in the (x, y, z) order the rotation is written in
    half = torch.tensor([0.5 * W * dx, 0.5 * H * dy, 0.5 * D * dz], device=theta.device)
    S = torch.diag(half)
    A = torch.linalg.inv(S) @ R @ S                                   # rotation in normalized xyz
    b = t / half

    # grid_sample orders the last axis (x, y, z); affine_grid's matrix rows are (x, y, z) too.
    M = torch.cat([A, b[:, None]], dim=1)                             # (3,4)
    return M[None]


def rigid_align(src: torch.Tensor, ref: torch.Tensor, spacing, *, iters: int = 300,
                lr: float = 0.02, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit the 6-DoF pose that best maps `src` onto `ref`. -> (aligned src, theta).

    Objective is masked NCC, not L2: the two volumes may differ in overall brightness (the whole
    reason `l2si` exists upstream) and the alignment must not chase that.
    """
    D, H, W = src.shape
    th = torch.zeros(6, device=src.device, requires_grad=True)
    opt = torch.optim.Adam([th], lr=lr)
    r = ref[None, None]
    m = None if mask is None else mask[None, None].float()

    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        M = _affine_from_theta(th, (D, H, W), spacing)
        grid = F.affine_grid(M, (1, 1, D, H, W), align_corners=False)
        w = F.grid_sample(src[None, None], grid, align_corners=False, padding_mode="zeros")
        a, b = (w, r) if m is None else (w * m, r * m)
        a = a - a.mean()
        b = b - b.mean()
        loss = 1.0 - (a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12)
        loss.backward()
        opt.step()

    with torch.no_grad():
        M = _affine_from_theta(th, (D, H, W), spacing)
        grid = F.affine_grid(M, (1, 1, D, H, W), align_corners=False)
        out = F.grid_sample(src[None, None], grid, align_corners=False, padding_mode="zeros")
    return out[0, 0], th.detach()


def psnr(a, b, mask=None, peak=None):
    if mask is not None:
        a, b = a[mask], b[mask]
    peak = float(b.max()) if peak is None else peak
    mse = ((a - b) ** 2).mean().clamp_min(1e-20)
    return float(10 * torch.log10(peak ** 2 / mse))


def ssim(a: torch.Tensor, b: torch.Tensor, *, data_range: float, win: int = 7) -> float:
    """3D SSIM with a uniform window (self-contained; no skimage)."""
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    x, y = a[None, None], b[None, None]
    k, p = win, win // 2
    mu_x = F.avg_pool3d(x, k, 1, p, count_include_pad=False)
    mu_y = F.avg_pool3d(y, k, 1, p, count_include_pad=False)
    xx = F.avg_pool3d(x * x, k, 1, p, count_include_pad=False) - mu_x ** 2
    yy = F.avg_pool3d(y * y, k, 1, p, count_include_pad=False) - mu_y ** 2
    xy = F.avg_pool3d(x * y, k, 1, p, count_include_pad=False) - mu_x * mu_y
    s = ((2 * mu_x * mu_y + C1) * (2 * xy + C2)) / \
        ((mu_x ** 2 + mu_y ** 2 + C1) * (xx + yy + C2))
    return float(s.mean())


def aligned_metrics(recon: torch.Tensor, gt: torch.Tensor, spacing, *,
                    mask: torch.Tensor | None = None, iters: int = 300) -> dict:
    """THE headline metric. Rigidly align `recon` to `gt`, then score. See the module docstring.

    Both the raw and the aligned numbers are returned, deliberately: the gap between them IS the
    gauge, and watching it is how you tell a genuinely bad reconstruction from a well-reconstructed
    one sitting at the wrong pose.
    """
    peak = float(gt[mask].max()) if mask is not None else float(gt.max())
    dr = peak
    out = {
        "psnr_raw": psnr(recon, gt, mask, peak),
        "ssim_raw": ssim(recon, gt, data_range=dr),
    }
    al, th = rigid_align(recon, gt, spacing, mask=mask, iters=iters)
    out["psnr_aligned"] = psnr(al, gt, mask, peak)
    out["ssim_aligned"] = ssim(al, gt, data_range=dr)
    out["gauge_shift_mm"] = float(th[:3].norm())
    out["gauge_rot_deg"] = float(torch.rad2deg(th[3:].norm()))
    return out
