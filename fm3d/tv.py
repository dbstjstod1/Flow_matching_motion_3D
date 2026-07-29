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
    steps of size `step`*||z|| along -unit(grad).

    2D BY-EYE SWEET SPOT (do not paraphrase this from memory -- the numbers are small and an
    order-of-magnitude slip here is invisible): step = 0.03 with kappa 0.3, or step = 0.015 with
    kappa 0.5. In 2D the by-eye K=1 pick used iters = 15; on our 3D volume the user judged that TV
    too strong (bone washed out), so 3D runs iters = 5 -- the sharp end of the 2D metric sweep,
    where the ceiling was flat over 4-8 and x_t got sharper as iters fell. Never below 4."""
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


# ---------------------------------------------------------------------------------------------
# The finite-difference operator D and its EXACT adjoint D^T, as separate maps.
#
# `sidky_dtv_grad_3d` above computes D^T s(D x) fused -- the TV gradient -- which is all a
# gradient-descent denoiser needs. ADMM-TV (Boyd et al. 2011, Found. & Trends ML 3(1), Sec. 6.4.1)
# needs the two halves SEPARATELY: the split variable d lives on the gradient field D x, and the
# x-subproblem's operator is A^T A + rho D^T D. See `admm_dc_step` in scripts/run_posterior3d.py.
#
# ADJOINTNESS IS THE WHOLE POINT and it is easy to get subtly wrong at the boundary. With the
# forward difference (D x)_i = x_{i+1} - x_i defined on i = 0 .. N-2, the adjoint is
#     (D^T p)_0     = -p_0
#     (D^T p)_i     =  p_{i-1} - p_i          0 < i < N-1
#     (D^T p)_{N-1} =  p_{N-2}
# which is exactly the "+= on the shifted slice, -= on the unshifted slice" accumulation used
# below (and, not by accident, the same pattern sidky_dtv_grad_3d already uses). Gated by
# scripts/gate_tv_adjoint.py -- <D x, p> == <x, D^T p> to float tolerance on random inputs.
# ---------------------------------------------------------------------------------------------

def grad_forward_3d(x: torch.Tensor):
    """D x for (1,1,D,H,W) -> (dz, dy, dx), each one voxel shorter along its own axis."""
    dz = x[..., 1:, :, :] - x[..., :-1, :, :]        # +z (SI)
    dy = x[..., :, 1:, :] - x[..., :, :-1, :]        # +y
    dx = x[..., :, :, 1:] - x[..., :, :, :-1]        # +x
    return dz, dy, dx


def div_adjoint_3d(dz: torch.Tensor, dy: torch.Tensor, dx: torch.Tensor,
                   shape, device=None, dtype=None) -> torch.Tensor:
    """D^T (dz, dy, dx) -> (1,1,D,H,W). The exact adjoint of `grad_forward_3d` (= -divergence)."""
    g = torch.zeros(shape, device=device if device is not None else dz.device,
                    dtype=dtype if dtype is not None else dz.dtype)
    g[..., 1:, :, :] += dz
    g[..., :-1, :, :] -= dz
    g[..., :, 1:, :] += dy
    g[..., :, :-1, :] -= dy
    g[..., :, :, 1:] += dx
    g[..., :, :, :-1] -= dx
    return g


def shrink(a: torch.Tensor, k: float) -> torch.Tensor:
    """Elementwise soft-threshold S_k(a) = sign(a) * max(|a| - k, 0).

    This is the EXACT prox of k*||.||_1, i.e. the z-update of ADMM for ANISOTROPIC TV (Boyd
    Sec. 6.4.1). Our TV is separable/anisotropic already (see sidky_dtv_grad_3d, which treats the
    three axes independently), so the closed form applies directly and no inner Chambolle/FGP
    loop is needed -- the single strongest practical argument for ADMM over FISTA here.
    Isotropic TV would instead need the BLOCK shrink (Boyd Sec. 6.4.2), grouping the three
    components at each voxel; that is NOT what this does."""
    return torch.sign(a) * torch.clamp(a.abs() - k, min=0.0)
