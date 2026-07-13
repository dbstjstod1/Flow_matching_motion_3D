"""6-DoF rigid motion estimation by differentiating the cone-beam forward model.

    theta_hat = argmin_theta  L( A(x; P_nom @ T(theta)),  y_measured )

`x` is the current image estimate (which, inside the posterior loop, is itself still improving),
`y` is the measured cone-beam data, and the gradient flows analytically through the projector's
recovery of the rays from P. See `rigid_motion.py` for how theta enters P.

Three estimators, ONE CONTRACT, so the posterior loop can swap them by name:

    est.refine_global(image, y_meas, iters, prox_anchor=None, prox_lam=0.0) -> float   # last loss
    est.current_params() -> (V, 6)
    est.render_sinogram(image) -> (V, nv, nu)

They differ only in how theta is parameterized, i.e. in what stops the fit from putting an
independent rigid pose on every view:

    direct   raw (V,6), Adam. No structure at all -- the baseline, and the thing the other two
             have to beat. Its parameters and Adam moments PERSIST across calls, which is what
             gives warm-start continuity down the posterior loop's ODE.
    basis    theta = B @ c with B a cubic B-spline basis over the view axis. Band limit is
             explicit (n_ctrl control points) and the waveform is not assumed -- unlike a Fourier
             basis, which on a sinusoidal simulation is handed the answer and scores beautifully
             for the wrong reason.
    net      MotionNet6DoF, the AI_Geocal architecture, with the 2D project's BAND-LIMITED hash
             settings ("hashbl"). This is the deployed default there.

NOTE ON THE PROJECTOR BACKEND. Motion estimation needs d(loss)/dP, and the Triton ray-march has
no adjoint for it (it returns None for the ray constants, which autograd reads as zero). The
projector refuses Triton whenever Pmat requires grad, so everything here runs on grid_sample.
That makes `n_samples` and `view_chunk` the memory knobs that matter.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .filters import ramp_filter
from .motion_net import MotionNet6DoF
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
        fp = ramp_filter(pred.reshape(V * nv, nu), du, window="hann").reshape(V, nv, nu)
        fy = ramp_filter(y.reshape(V * nv, nu), du, window="hann").reshape(V, nv, nu)
        return ((fp - fy) ** 2).mean()
    if kind == "l1":
        return (pred - y).abs().mean()
    raise ValueError(f"unknown sinogram loss '{kind}'")


def bspline_basis(V: int, n_ctrl: int, device, degree: int = 3) -> torch.Tensor:
    """(V, n_ctrl) uniform cubic B-spline design matrix over the view axis.

    theta = B @ c limits the trajectory's bandwidth to n_ctrl without assuming its SHAPE -- the
    guardrail against the Fourier basis, which reconstructs a sinusoidal simulation perfectly
    because the simulation is a sinusoid, and would not survive a cough.
    """
    s = torch.linspace(0, 1, V, device=device)
    t = torch.linspace(0, 1, n_ctrl - degree + 1, device=device)
    dt = t[1] - t[0]
    knots = torch.cat([t[:1] - dt * torch.arange(degree, 0, -1, device=device), t,
                       t[-1:] + dt * torch.arange(1, degree + 1, device=device)])

    def basis(i, k, x):
        if k == 0:
            return ((x >= knots[i]) & (x < knots[i + 1])).float()
        a = torch.zeros_like(x)
        if knots[i + k] > knots[i]:
            a = (x - knots[i]) / (knots[i + k] - knots[i]) * basis(i, k - 1, x)
        b = torch.zeros_like(x)
        if knots[i + k + 1] > knots[i + 1]:
            b = (knots[i + k + 1] - x) / (knots[i + k + 1] - knots[i + 1]) * basis(i + 1, k - 1, x)
        return a + b

    B = torch.stack([basis(i, degree, s) for i in range(n_ctrl)], dim=-1)
    B[-1] = B[-2]                                    # close the right end (x == 1 falls outside)
    return B / B.sum(-1, keepdim=True).clamp_min(1e-8)


# ======================================================================================
# estimators
# ======================================================================================

class _BaseEstimator:
    def __init__(self, cfg, P_nom, u_coords, v_coords, device, *, dx=1.0, dy=1.0, dz=1.0,
                 loss="l2si", lncc_win=9, n_samples=384, view_chunk=8, row_chunk=64,
                 smooth_w=1e-2, views_per_iter: int | None = None):
        self.cfg, self.P_nom = cfg, P_nom
        self.u, self.v = u_coords, v_coords
        self.device = device
        self.dx, self.dy, self.dz = dx, dy, dz
        self.loss_kind, self.lncc_win = loss, lncc_win
        self.n_samples, self.view_chunk, self.row_chunk = n_samples, view_chunk, row_chunk
        self.smooth_w = smooth_w
        # Stochastic view subsampling. A cone-beam view is nv*nu rays -- ~200x a fan-beam view --
        # so unlike the 2D project we cannot afford every view on every inner iteration. None =
        # all views (correct, slow); an int = that many random views per iteration (an unbiased
        # estimate of the same gradient, and what makes PER~10 inner iters affordable).
        self.views_per_iter = views_per_iter
        self.V = cfg.n_views

    # -- subclasses provide these ------------------------------------------------------
    def _theta(self) -> torch.Tensor:               # (V, 6), differentiable
        raise NotImplementedError

    def _opt(self) -> torch.optim.Optimizer:
        """The optimizer PERSISTS across refine_global calls, and that is deliberate: down the
        posterior loop's ODE each call warm-starts from the last one's theta AND its Adam moments,
        so the trajectory is continuous instead of being re-fitted from scratch 30 times."""
        return self._optim

    # -- shared ------------------------------------------------------------------------
    def _project(self, image, theta, views=None):
        P_nom = self.P_nom if views is None else self.P_nom[views]
        th = theta if views is None else theta[views]
        P = params_to_Pmot(th, P_nom)
        return forward_project_3d_batched(
            image[None, None], P[None], self.u, self.v,
            dx=self.dx, dy=self.dy, dz=self.dz, n_samples=self.n_samples,
            view_chunk=self.view_chunk, row_chunk=self.row_chunk)[0]

    def _smooth(self, theta):
        """2nd-difference penalty on the trajectory. Zero for `basis`/`net`, whose smoothness is
        structural -- but harmless there, and it keeps one code path."""
        if self.smooth_w <= 0:
            return theta.new_zeros(())
        d2 = theta[2:] - 2 * theta[1:-1] + theta[:-2]
        return self.smooth_w * (d2 ** 2).mean()

    def refine_global(self, image, y_meas, iters=30, *, prox_anchor=None, prox_lam=0.0) -> float:
        opt = self._opt()
        last = float("nan")
        for _ in range(iters):
            opt.zero_grad(set_to_none=True)
            theta = self._theta()
            if self.views_per_iter is None or self.views_per_iter >= self.V:
                views = None
                pred, tgt = self._project(image, theta), y_meas
            else:
                views = torch.randperm(self.V, device=self.device)[:self.views_per_iter]
                pred, tgt = self._project(image, theta, views), y_meas[views]
            loss = sinogram_data_loss(pred, tgt, self.loss_kind,
                                      lncc_win=self.lncc_win, du=self.cfg.du)
            loss = loss + self._smooth(theta)
            if prox_anchor is not None and prox_lam > 0:
                loss = loss + prox_lam * ((theta - prox_anchor) ** 2).mean()
            loss.backward()
            opt.step()
            last = float(loss.detach())
        return last

    @torch.no_grad()
    def current_params(self) -> torch.Tensor:
        return self._theta().detach()

    @torch.no_grad()
    def render_sinogram(self, image) -> torch.Tensor:
        return self._project(image, self._theta())


class DirectMotionEstimator(_BaseEstimator):
    """Free per-view (V,6). No structure; the baseline the others must beat."""

    def __init__(self, *a, lr=0.3, **kw):
        super().__init__(*a, **kw)
        self.params = torch.zeros(self.V, 6, device=self.device, requires_grad=True)
        self._optim = torch.optim.Adam([self.params], lr=lr)

    def _theta(self):
        return self.params


class BasisMotionEstimator(_BaseEstimator):
    """theta = B @ c, cubic B-spline over the view axis. Explicit band limit, no assumed shape."""

    def __init__(self, *a, n_ctrl=20, lr=0.3, **kw):
        super().__init__(*a, **kw)
        self.B = bspline_basis(self.V, n_ctrl, self.device)          # (V, n_ctrl)
        self.c = torch.zeros(n_ctrl, 6, device=self.device, requires_grad=True)
        self._optim = torch.optim.Adam([self.c], lr=lr)
        self.smooth_w = 0.0            # the basis IS the smoothness prior; do not charge twice

    def _theta(self):
        return self.B @ self.c


class NetMotionEstimator(_BaseEstimator):
    """MotionNet6DoF (AI_Geocal's architecture, band-limited settings). Deployed default in 2D."""

    def __init__(self, *a, lr=1e-2, enc="hash", n_levels=4, base_resolution=2,
                 per_level_scale=2.0, fourier_m=8, trans_max_mm=15.0, rot_max_deg=8.0, **kw):
        super().__init__(*a, **kw)
        self.net = MotionNet6DoF(
            self.V, enc=enc, n_levels=n_levels, base_resolution=base_resolution,
            per_level_scale=per_level_scale, fourier_m=fourier_m,
            trans_max_mm=trans_max_mm, rot_max_deg=rot_max_deg).to(self.device)
        self._optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.smooth_w = 0.0            # bandwidth is structural; see motion_net.py

    def _theta(self):
        return self.net.all_params(device=self.device)


def make_estimator(name: str, cfg, P_nom, u, v, device, **kw) -> _BaseEstimator:
    n = name.lower()
    if n == "direct":
        return DirectMotionEstimator(cfg, P_nom, u, v, device, **kw)
    if n == "basis":
        return BasisMotionEstimator(cfg, P_nom, u, v, device, **kw)
    if n in ("net", "mlp", "hashbl"):
        return NetMotionEstimator(cfg, P_nom, u, v, device, **kw)
    raise ValueError(f"unknown estimator '{name}' (direct|basis|net)")
