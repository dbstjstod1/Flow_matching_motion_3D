"""Common projection-residual fitting for the paper's two spline estimators.

Akima + gradient descent is the main estimator; cubic B-splines + RMSprop
provide the motion-estimator ablation. Both retain spline coefficients across
image updates and the coarse-to-fine grid transition. The objective is ordinary
projection MSE, with no explicit pose penalty or initial-loss normalization.

The projector differentiates P_nom @ T(theta); images are not warped.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .filters import DEFAULT_RAMP_WINDOW, ramp_filter
from .projector_3d import forward_project_3d_batched
from .rigid_motion import params_to_Pmot


# ======================================================================================
# projection-domain data terms
# ======================================================================================

def lncc_loss(pred: torch.Tensor, y: torch.Tensor, win: int = 9, eps: float = 1e-5):
    """1 - mean local NCC over a win x win window on each (nv, nu) projection. (V,nv,nu).

    This is AI_Geocal's objective (it uses MONAI's LocalNormalizedCrossCorrelationLoss with a
    31x31 rectangular kernel, 2D, on each projection). Reimplemented with avg_pool2d so there is
    no MONAI dependency and the window size is a plain argument.

    WHY IT IS THE RIGHT DATA TERM HERE and plain L2 is not: inside the posterior loop the image
    estimate `x` is still converging, so `A(x)` and `y` differ by an intensity/scale mismatch
    that has nothing to do with geometry. L2 charges that mismatch to the motion parameters. LNCC
    normalizes by a LOCAL mean and variance, so it is blind to any local affine intensity change
    and only sees misalignment -- which is exactly the quantity being estimated.

    `l2si` (below) is the cheap version of the same idea: invariant to a single GLOBAL scale.
    LNCC is invariant to a local one, which is stronger and costs a few pooled convolutions.
    """
    p, t = pred[:, None], y[:, None]                                  # (V,1,nv,nu)
    k, pad = win, win // 2
    mu_p = F.avg_pool2d(p, k, 1, pad, count_include_pad=False)
    mu_t = F.avg_pool2d(t, k, 1, pad, count_include_pad=False)
    pp = F.avg_pool2d(p * p, k, 1, pad, count_include_pad=False) - mu_p * mu_p
    tt = F.avg_pool2d(t * t, k, 1, pad, count_include_pad=False) - mu_t * mu_t
    pt = F.avg_pool2d(p * t, k, 1, pad, count_include_pad=False) - mu_p * mu_t
    ncc = pt / torch.sqrt(pp.clamp_min(0) * tt.clamp_min(0) + eps)
    return 1.0 - ncc.mean()


def sinogram_data_loss(pred: torch.Tensor, y: torch.Tensor, kind: str = "l2si", *,
                       lncc_win: int = 9, du: float = 1.0) -> torch.Tensor:
    """Data term between a rendered and a measured cone-beam sinogram, both (V, nv, nu).

    Plain MSE conflates GEOMETRIC MISALIGNMENT with the MASS/scale mismatch of an image estimate
    that is still evolving. Every other option here removes one class of that nuisance:

      l2si   scale-invariant L2: c = <p,y>/<p,p>, then ||c*p - y||^2. Invariant to one global
             multiplicative factor. THE 2D PROJECT'S WORKING DEFAULT, and the cheapest fix.
      lncc   local NCC (see above). AI_Geocal's choice. Invariant to a LOCAL affine.
      ncc    global NCC: invariant to a global affine a*p + b (broader than l2si, weaker
             than lncc).
      ramp   L2 after ramp-filtering along u: deletes the low-frequency band where brightness
             drift lives (measured in 2D: the drift is ~99% low-frequency).
      l1     robust to outlier rays.
      l2     plain MSE. Kept only as the baseline it is.
    """
    if kind in ("l2", "mse", "rmse"):
        return ((pred - y) ** 2).mean()
    if kind == "l2si":
        c = (pred * y).sum() / (pred * pred).sum().clamp_min(1e-12)
        return ((c * pred - y) ** 2).mean()
    if kind == "lncc":
        return lncc_loss(pred, y, win=lncc_win)
    if kind == "ncc":
        p = pred - pred.mean()
        t = y - y.mean()
        return 1.0 - (p * t).sum() / (p.norm() * t.norm()).clamp_min(1e-12)
    if kind == "ramp":
        V, nv, nu = pred.shape
        # same window the FDK uses (filters.DEFAULT_RAMP_WINDOW), so the band this loss weights
        # is the band the reconstruction actually keeps -- one source of truth, no drift.
        fp = ramp_filter(pred.reshape(V * nv, nu), du,
                         window=DEFAULT_RAMP_WINDOW).reshape(V, nv, nu)
        fy = ramp_filter(y.reshape(V * nv, nu), du,
                         window=DEFAULT_RAMP_WINDOW).reshape(V, nv, nu)
        return ((fp - fy) ** 2).mean()
    if kind == "l1":
        return (pred - y).abs().mean()
    raise ValueError(f"unknown sinogram loss '{kind}'")


class _BaseEstimator:
    def __init__(self, cfg, P_nom, u_coords, v_coords, device, *, dx=1.0, dy=1.0, dz=1.0,
                 loss="l2si", lncc_win=9,
                 smooth_w=1e-2, views_per_iter: int | None = None, opt="adam",
                 decay: float = 1.0, decay_reset: str = "call", view_seed: int | None = 0):
        self.cfg, self.P_nom = cfg, P_nom
        self.u, self.v = u_coords, v_coords
        self.device = device
        self.dx, self.dy, self.dz = dx, dy, dz
        self.loss_kind, self.lncc_win = loss, lncc_win
        self.smooth_w = smooth_w
        # Stochastic view subsampling. A cone-beam view is nv*nu rays -- ~200x a fan-beam view --
        # so unlike the 2D project we cannot afford every view on every inner iteration. None =
        # all views (correct, slow); an int = that many random views per iteration (an unbiased
        # estimate of the same gradient, and what makes PER~10 inner iters affordable).
        self.views_per_iter = views_per_iter
        self.V = cfg.n_views
        # (`n_samples` / `view_chunk` / `row_chunk` / `fp16_vol` / `backend` all lived here and
        # were all ray-march knobs. The operator is LEAP's now -- it has no ray samples, does
        # its own chunking, and there is nothing to select. Removed 2026-07-29. The forward
        # VALUE is LEAP's (Joseph, pinned); the d/dP this class optimizes through is that
        # kernel's exact gradient, wired in `leap_projector.LEAPProject`.)
        # OPTIMIZER. "adam" is ours; "gd" is Thies' (plain gradient descent, step size s0 with an
        # exponential decay t per iteration, TMI 2025 II-C). Subclasses build the optimizer, so
        # they read these.
        self.opt_kind, self.decay, self.decay_reset = opt, decay, decay_reset
        self._it = 0                       # iterations since the last decay reset
        # REPRODUCIBILITY. The random view subset used to be drawn from the global CUDA RNG, which
        # made two identically-configured runs diverge (measured: rot traces differing 15-20%
        # relative, e.g. 2.33/0.47/0.36/0.29 vs 2.47/0.55/0.42/0.34 at steps 0/5/10/15). That is
        # larger than several of the differences this project ranks configs by. A dedicated seeded
        # generator makes the view schedule identical across runs; what remains is the
        # atomic-add ordering in the gradient kernel's backward, which is far smaller.
        self._vgen = None
        if view_seed is not None:
            self._vgen = torch.Generator(device=device)
            self._vgen.manual_seed(int(view_seed))

    def _make_opt(self, params, lr):
        if self.opt_kind == "gd":
            return torch.optim.SGD(params, lr=lr)
        if self.opt_kind == "adam":
            return torch.optim.Adam(params, lr=lr)
        raise ValueError(f"unknown optimizer '{self.opt_kind}' (adam | gd)")

    # -- subclasses provide these ------------------------------------------------------
    def _theta(self) -> torch.Tensor:               # (V, 6), differentiable
        raise NotImplementedError

    def _opt(self) -> torch.optim.Optimizer:
        """Return this fit's optimizer; spline subclasses control moment/schedule resets."""
        return self._optim

    # -- shared ------------------------------------------------------------------------
    def _project(self, image, theta, views=None):
        P_nom = self.P_nom if views is None else self.P_nom[views]
        th = theta if views is None else theta[views]
        P = params_to_Pmot(th, P_nom)
        return forward_project_3d_batched(
            image[None, None], P[None], self.u, self.v,
            dx=self.dx, dy=self.dy, dz=self.dz)[0]

    def _smooth(self, theta):
        """Optional second-difference term; disabled by both published spline schemes."""
        if self.smooth_w <= 0:
            return theta.new_zeros(())
        d2 = theta[2:] - 2 * theta[1:-1] + theta[:-2]
        return self.smooth_w * (d2 ** 2).mean()

    def refine_global(self, image, y_meas, iters=30, *, prox_anchor=None, prox_lam=0.0) -> float:
        opt = self._opt()
        last = None
        # Thies restarts his optimization (and therefore his step-size decay) from scratch; ours
        # warm-starts down the ODE. `decay_reset="call"` reproduces his within each refine_global,
        # "global" lets the step size keep shrinking across the whole run.
        if self.decay != 1.0 and self.decay_reset == "call":
            self._it = 0
        for _ in range(iters):
            if self.decay != 1.0:
                for g in opt.param_groups:
                    g["lr"] = self.lr0 * (self.decay ** self._it)
                self._it += 1
            opt.zero_grad(set_to_none=True)
            theta = self._theta()
            if self.views_per_iter is None or self.views_per_iter >= self.V:
                views = None
                pred, tgt = self._project(image, theta), y_meas
            else:
                views = torch.randperm(self.V, device=self.device, generator=self._vgen)[
                    :self.views_per_iter]
                pred, tgt = self._project(image, theta, views), y_meas[views]
            loss = sinogram_data_loss(pred, tgt, self.loss_kind,
                                      lncc_win=self.lncc_win, du=self.cfg.du)
            loss = loss + self._smooth(theta)
            if prox_anchor is not None and prox_lam > 0:
                loss = loss + prox_lam * ((theta - prox_anchor) ** 2).mean()
            loss.backward()
            opt.step()
            # keep the scalar ON DEVICE: float() here forces a host sync EVERY iteration
            # (~1 ms x PER=400 x 50 steps); only the final value is ever read.
            last = loss.detach()
        return float(last) if last is not None else float("nan")

    @torch.no_grad()
    def current_params(self) -> torch.Tensor:
        return self._theta().detach()

    @torch.no_grad()
    def render_sinogram(self, image) -> torch.Tensor:
        return self._project(image, self._theta())


def make_estimator(name: str, cfg, P_nom, u, v, device, **kw):
    from .spline_motion import SplineSchemeEstimator
    if name not in ("akima_gd", "bspline_rmsprop"):
        raise ValueError(f"unknown estimator {name!r}; choose akima_gd or bspline_rmsprop")
    kw.setdefault("lr", {"akima_gd": 1000.0, "bspline_rmsprop": 0.001}[name])
    kw.setdefault("loss", "l2")
    return SplineSchemeEstimator(cfg, P_nom, u, v, device, scheme=name, **kw)
