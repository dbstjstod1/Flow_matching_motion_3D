"""Sidky-Pan separable (anisotropic) directional-TV denoiser + soft-DC nudge.

Ported verbatim (semantics-identical) from the rigid project's
`scripts/exp_xtsoft_loss.py` so the 4DCT inference loop is self-contained and the
winner PnP corrector (`DENOISER=tv`, Sidky 15 it) reuses the SAME operator the
rigid loop was tuned on. The TV denoiser is x1_hat-FREE (no network eval), hence
OOD-safe: it can clean the carried image without feeding a data-prox'd (off-manifold)
image back into the FM prior.

Convention: images are (1,1,H,W). `wx != wy` makes the TV DIRECTIONAL (heavier
penalty along the chosen streak axis).
"""

from __future__ import annotations

import torch


def soft_nudge(x: torch.Tensor, grad: torch.Tensor, alpha: float) -> torch.Tensor:
    """Normalized soft step: move `alpha` fraction of ||x|| along -unit(grad).

    This is the SOFT data-prox (a gentle, magnitude-controlled descent), NOT a hard
    reconstruction -- the winner keeps the DC as a soft nudge on the FM-predicted image."""
    g_unit = grad / (grad.norm() + 1e-12)
    return (x - alpha * x.norm() * g_unit).detach()


def sidky_dtv_grad(x: torch.Tensor, wx: float = 1.0, wy: float = 1.0,
                   eps: float = 1e-8) -> torch.Tensor:
    """Sidky-Pan SEPARABLE directional-TV gradient.
    Penalty = wx*||D_x x||_1 + wy*||D_y x||_1 (anisotropic TV; D_x,D_y = forward diffs).
    grad = wx*D_x^T sign(D_x x) + wy*D_y^T sign(D_y x), with a Huber-smoothed sign
    s = d/sqrt(d^2+eps) so it is differentiable and streak-selective."""
    dx = x[..., :, 1:] - x[..., :, :-1]          # forward diff along W
    dy = x[..., 1:, :] - x[..., :-1, :]          # forward diff along H
    sx = dx / torch.sqrt(dx * dx + eps)          # smoothed sign(D_x x)
    sy = dy / torch.sqrt(dy * dy + eps)
    gx = torch.zeros_like(x); gy = torch.zeros_like(x)
    gx[..., :, 1:] += sx; gx[..., :, :-1] -= sx  # D_x^T sx  (= -divergence_x)
    gy[..., 1:, :] += sy; gy[..., :-1, :] -= sy  # D_y^T sy
    return wx * gx + wy * gy


def sidky_dtv_denoise(x: torch.Tensor, iters: int, step: float,
                      wx: float = 1.0, wy: float = 1.0) -> torch.Tensor:
    """Denoise (1,1,H,W) with the Sidky separable directional TV: `iters` normalized
    steps of size `step`*||z|| along -unit(grad). Winner: iters=15, step=0.1-ish."""
    z = x.detach().clone()
    for _ in range(iters):
        g = sidky_dtv_grad(z, wx, wy)
        z = (z - step * z.norm() * g / (g.norm() + 1e-12)).detach()
    return z


# =======================================================================================
# 3D (prompt 6). Same separable anisotropic TV with a third (z / SI) forward-difference
# term. `wz` is separate because the cone-beam z axis has different sampling (slice
# thickness) and different streak structure than the in-plane axes.
# =======================================================================================
def sidky_dtv_grad_3d(x: torch.Tensor, wx: float = 1.0, wy: float = 1.0,
                      wz: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    """3D Sidky-Pan separable directional-TV gradient. x: (1,1,D,H,W)."""
    dx = x[..., :, :, 1:] - x[..., :, :, :-1]        # forward diff along W (+x)
    dy = x[..., :, 1:, :] - x[..., :, :-1, :]        # forward diff along H (+y)
    dz = x[..., 1:, :, :] - x[..., :-1, :, :]        # forward diff along D (+z, SI)
    sx = dx / torch.sqrt(dx * dx + eps)
    sy = dy / torch.sqrt(dy * dy + eps)
    sz = dz / torch.sqrt(dz * dz + eps)
    gx = torch.zeros_like(x); gy = torch.zeros_like(x); gz = torch.zeros_like(x)
    gx[..., :, :, 1:] += sx; gx[..., :, :, :-1] -= sx
    gy[..., :, 1:, :] += sy; gy[..., :, :-1, :] -= sy
    gz[..., 1:, :, :] += sz; gz[..., :-1, :, :] -= sz
    return wx * gx + wy * gy + wz * gz


def sidky_dtv_denoise_3d(x: torch.Tensor, iters: int, step: float, wx: float = 1.0,
                         wy: float = 1.0, wz: float = 1.0) -> torch.Tensor:
    """Denoise (1,1,D,H,W) with the 3D Sidky separable directional TV. x1_hat-FREE
    (no network eval) hence OOD-safe, exactly like the 2D twin."""
    z = x.detach().clone()
    for _ in range(iters):
        g = sidky_dtv_grad_3d(z, wx, wy, wz)
        z = (z - step * z.norm() * g / (g.norm() + 1e-12)).detach()
    return z
