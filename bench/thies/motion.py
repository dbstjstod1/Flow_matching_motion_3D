"""Thies' motion model `p : R^N -> R^M`: 6-DoF Akima splines -> per-view projection matrices.

TMI II-B.1, L249-292:

    "for each projection j, we seek to estimate rigid transformation matrices T_j in R^{4x4}
     which are multiplied to the initial projection matrices P_j to yield updated projection
     matrices"                                                                  P*_j = P_j T_j(x)

    "we assume that the motion is smooth over time ... this reduces the degrees of freedom from
     6 N_p to 6 N_n ... We use ... Akima splines ... 10 nodes for the motion patterns being
     simulated and 30 nodes for the motion patterns being estimated by our method. The nodes are
     evenly spaced across the temporal direction of the scan with one node at the beginning and
     one node at the end of the scan range."

TWO NODE COUNTS, TWO DIFFERENT JOBS -- do not collapse them:

  * n_nodes = 10  simulates the ground-truth motion. We do NOT use this class for that; the
    ground truth comes from `fm3d.rigid_motion.akima_motion`, which is the same family with the
    same 10 nodes and is what our own pipeline draws, so both methods are handed the identical
    corrupted scan. Gate G2 checks the two Akima implementations agree.
  * n_nodes = 30  is the ESTIMATOR's parameterization. That is this class. It is deliberately
    over-parameterized relative to the truth (30 > 10) -- their choice, kept.

UNITS: translations in mm, rotations in **DEGREES**
--------------------------------------------------
The optimizer takes ONE scalar step size for all 6 DoF (`s0 = 100`, Eq. 6 / L376-380), so the
parameters have to be on a comparable scale. Their released 2D sampler settles what that scale
is: node values are drawn in degrees and converted with `r = r / 180 * pi` inside the spline
motion model (`is_radian=False` in `refs/thies_moco_diff_likelihood/motion_models/`). We do the
same: `x` is (mm, mm, mm, deg, deg, deg) per node, and `theta()` hands radians onward, because
`fm3d.rigid_motion` is radians throughout. Parameterizing the rotations in radians instead would
silently make the rotational half of the search ~57x slower at the same s0.

ROTATION PARAMETERIZATION is ours, not theirs -- the paper says only "three rotational and three
translational components". We use the repo's AXIS-ANGLE convention
(`fm3d.rigid_motion.rigid_motion_matrices`) so that the estimated `theta` is the same object our
own estimator produces and `reprojection_error` / `motion_error` compare like with like.
See PROVENANCE.md section 4.3.
"""

from __future__ import annotations

import math

import torch

from fm3d.rigid_motion import params_to_Pmot

from .vendor_import import akima

__all__ = ["ThiesSplineMotion", "akima_resample"]


def akima_resample(node_values: torch.Tensor, n_views: int) -> torch.Tensor:
    """(n_nodes, C) node values -> (n_views, C), via THEIR torch Akima. Differentiable.

    Nodes are evenly spaced with one at each end (`linspace(0, V-1, n_nodes)`), per L290-292.
    `interpolate_akima_spline` uses its sample points as an index, so they must be an integer
    tensor -- that cast lives here and nowhere else.
    """
    interp = akima()
    dev, dt = node_values.device, node_values.dtype
    tn = torch.linspace(0.0, n_views - 1, node_values.shape[0], device=dev, dtype=dt)
    pts = torch.arange(n_views, device=dev)                     # INTEGER on purpose
    return torch.stack([interp(tn, node_values[:, c], pts) for c in range(node_values.shape[1])],
                       dim=-1)


class ThiesSplineMotion:
    """The optimizer's free parameters `x` and the map `x -> P*`.

    `x` is (n_nodes, 6) = (tx, ty, tz [mm], rx, ry, rz [deg]) and is initialized to ZERO, which
    is Eq. 6's `x^(0) = 0` and also the identity transform, so iteration 0 reconstructs exactly
    the uncompensated scan.
    """

    def __init__(self, n_views: int, *, n_nodes: int = 30, device="cuda",
                 dtype: torch.dtype = torch.float32):
        if n_nodes < 5:
            raise ValueError("their Akima needs >= 5 nodes (it forms m1 from 4 phantom slopes)")
        self.n_views = int(n_views)
        self.n_nodes = int(n_nodes)
        self.x = torch.zeros(self.n_nodes, 6, device=device, dtype=dtype, requires_grad=True)

    # -- the map ------------------------------------------------------------------------
    def theta(self) -> torch.Tensor:
        """(V, 6) per-view motion in the repo's units: mm and RADIANS. Differentiable in `x`."""
        th = akima_resample(self.x, self.n_views)
        return torch.cat([th[:, :3], th[:, 3:] * (math.pi / 180.0)], dim=-1)

    def Pmot(self, P_nom: torch.Tensor) -> torch.Tensor:
        """(V,3,4) `P*_j = P_j T_j(x)` -- the exact composition of L213-215, via the repo's one
        seam so that simulate-side and correct-side motion can never disagree."""
        return params_to_Pmot(self.theta(), P_nom)

    # -- optimizer plumbing --------------------------------------------------------------
    @property
    def parameters(self) -> list[torch.Tensor]:
        return [self.x]

    @torch.no_grad()
    def gd_step(self, step_size: float) -> float:
        """Eq. 6: `x <- x - s(n) * df/dx`. A PLAIN gradient step -- not Adam, not normalized, no
        momentum. Returns the gradient norm so the caller can log it. Clears the grad after."""
        if self.x.grad is None:
            raise RuntimeError("no gradient on x -- did you call .backward()?")
        g = self.x.grad
        norm = float(g.norm())
        self.x -= step_size * g
        self.x.grad = None
        return norm

    @torch.no_grad()
    def load_theta_(self, theta: torch.Tensor) -> None:
        """Seed `x` by SAMPLING a given (V,6) per-view motion at the 30 node positions.

        An Akima spline interpolates its nodes exactly, so this reproduces `theta` at the nodes
        and Akima-interpolates in between; it is not a least-squares fit, and it does not need to
        be, because the map from node values to per-view motion is NOT linear (the slope weights
        depend on |diff(m)|) so a pseudo-inverse would be wrong anyway.

        Used ONLY by the gates (can the 30-node model represent a 10-node truth?) and by
        `--init oracle` diagnostics. The method itself always starts from x = 0.
        """
        if theta.shape != (self.n_views, 6):
            raise ValueError(f"theta must be ({self.n_views},6); got {tuple(theta.shape)}")
        th = torch.cat([theta[:, :3], theta[:, 3:] * (180.0 / math.pi)], dim=-1)
        tn = torch.linspace(0.0, self.n_views - 1, self.n_nodes,
                            device=theta.device, dtype=torch.float64)
        lo = tn.floor().long().clamp(0, self.n_views - 1)
        hi = tn.ceil().long().clamp(0, self.n_views - 1)
        f = (tn - lo.to(tn.dtype))[:, None].to(th.dtype)
        self.x.copy_(((1 - f) * th[lo] + f * th[hi]).to(self.x.dtype))
