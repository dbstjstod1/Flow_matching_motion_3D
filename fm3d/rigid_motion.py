"""Per-view rigid (6-DoF) patient motion, composed into the projection matrices.

THE PARAMETERIZATION
--------------------
    theta: (V, 6) = [tx, ty, tz | wx, wy, wz]
        translation  [mm]        (world axes: x, y in the orbit plane; z = SI = rotation axis)
        rotation     [rad]       ROTATION VECTOR (axis-angle), i.e. an element of so(3)

Rotation is an axis-angle vector, NOT Euler angles. That is not a style choice, it is what
makes the geometry bridge legal. The bridge trains on partially-corrected geometries

    x_t = FDK(y, P_nom @ T(t * theta)),   t in [0, 1]

and for `t * theta` to trace the SHORTEST PATH from "no correction" to "full correction",
the map theta -> T(theta) must be a geodesic in the motion group. exp(t*skew(w)) is exactly
the SO(3) geodesic; scaling Euler angles is not (AI_Geocal uses Rz@Ry@Rx in degrees, which is
fine there because it never interpolates). Getting this wrong does not crash -- it silently
curves the bridge, so the finite-difference velocity target no longer points where the
inference ODE travels.

We interpolate on SO(3) x R^3 (rotation geodesic, translation straight line) rather than on
SE(3) (the screw-motion exp, which couples them). Both are geodesics; the decoupled one keeps
`tx,ty,tz` literally "millimetres of translation" at every t, which is what the motion
estimator's bounds, smoothness penalty and reported RMSE all assume.

HOW MOTION ENTERS THE FORWARD MODEL
-----------------------------------
The volume is never warped. The per-view rigid transform is right-multiplied into the nominal
projection matrix, exactly as in the 2D code (`P_nominal @ T_obj`) and in AI_Geocal
(`E = E0 @ T_obj`; `P = K @ E`):

    P_moved[v] = P_nom[v] @ T(theta[v])          (V,3,4) @ (V,4,4) -> (V,3,4)

Because T right-multiplies, it acts in OBJECT space: `P T x` projects the point x after the
object has been moved by T. So theta is patient motion (equivalently, inverse gantry motion),
and the pivot is wherever the world origin is.

    PIVOT. The projector puts the world origin at the VOLUME CENTRE (`X0 = -0.5*W*dx`, and
    likewise y, z -- see projector_3d.forward_project_3d_batched). So rotation is about the
    volume centre for free, and AI_Geocal's explicit pivot correction `t_eff = tp + c - R c`
    reduces to `t_eff = tp` here (c = 0). If you ever move the world box off-centre, this file
    is where that assumption breaks.

Keeping the motion in the geometry (rather than warping the volume) is what makes the whole
project cheap: a motion update is a (V,4,4) matmul, not V volume resamplings, and
d(sinogram)/d(theta) flows analytically through the projector's ray recovery from P.
"""

from __future__ import annotations

import math

import torch


# --------------------------------------------------------------------------------------
# so(3) / SE(3)
# --------------------------------------------------------------------------------------

def skew(w: torch.Tensor) -> torch.Tensor:
    """(..., 3) rotation vector -> (..., 3, 3) skew-symmetric matrix."""
    wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
    zero = torch.zeros_like(wx)
    return torch.stack([
        torch.stack([zero, -wz, wy], dim=-1),
        torch.stack([wz, zero, -wx], dim=-1),
        torch.stack([-wy, wx, zero], dim=-1),
    ], dim=-2)


def so3_exp(w: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Rodrigues: (..., 3) rotation vector [rad] -> (..., 3, 3) rotation matrix.

    Differentiable everywhere INCLUDING at w = 0, which matters because the estimator is
    initialized at theta = 0 and the bridge evaluates t = 0. The naive formula divides by
    |w|, so its gradient is nan at the origin even though the limit is finite; we branch to
    the 2nd-order Taylor series below `eps` and -- critically -- clamp the angle BEFORE the
    division, so the nan never enters the graph (torch.where alone would still backprop a nan
    through the discarded branch).
    """
    theta = w.norm(dim=-1, keepdim=True)                     # (..., 1)
    small = theta < eps
    theta_safe = torch.where(small, torch.ones_like(theta), theta)

    # sin(t)/t and (1-cos t)/t^2, with their Taylor limits 1 and 1/2 at t -> 0
    a = torch.where(small, 1.0 - theta ** 2 / 6.0, torch.sin(theta_safe) / theta_safe)
    b = torch.where(small, 0.5 - theta ** 2 / 24.0,
                    (1.0 - torch.cos(theta_safe)) / (theta_safe ** 2))

    K = skew(w)
    K2 = K @ K
    eye = torch.eye(3, device=w.device, dtype=w.dtype).expand(K.shape)
    return eye + a[..., None] * K + b[..., None] * K2


def so3_log(R: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Inverse of `so3_exp`: (..., 3, 3) -> (..., 3) rotation vector [rad].

    Only used for diagnostics/gates (and for reporting a rotation error as an angle); the
    forward model never needs it.
    """
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos)[..., None]                        # (..., 1)
    small = theta < eps
    theta_safe = torch.where(small, torch.ones_like(theta), theta)
    # vee(R - R^T) = 2 sin(theta) * axis
    vee = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1)
    scale = torch.where(small,
                        torch.full_like(theta_safe, 0.5),
                        theta_safe / (2.0 * torch.sin(theta_safe)))
    return scale * vee


def rigid_motion_matrices(theta: torch.Tensor) -> torch.Tensor:
    """(V, 6) [tx,ty,tz mm | wx,wy,wz rad] -> (V, 4, 4) homogeneous object transform.

    T = [[R(w), t], [0, 1]], rotation about the world origin (= the volume centre).
    theta = 0 gives exactly the identity, so a zero-initialized estimator starts at the
    nominal geometry.
    """
    if theta.shape[-1] != 6:
        raise ValueError(f"theta must be (..., 6) [tx,ty,tz,wx,wy,wz]; got {tuple(theta.shape)}")
    t = theta[..., :3]
    R = so3_exp(theta[..., 3:])
    V = theta.shape[:-1]
    T = torch.zeros(*V, 4, 4, device=theta.device, dtype=theta.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def apply_rigid_motion(P_nominal: torch.Tensor, T_obj: torch.Tensor) -> torch.Tensor:
    """P_nominal (..., V, 3, 4) @ T_obj (..., V, 4, 4) -> P_moved (..., V, 3, 4).

    The 3D twin of the 2D `apply_rigid_motion`, and identical in form: the object transform
    right-multiplies the nominal projection matrix. Broadcasting means this works for both
    the unbatched (V,3,4) and the batched (B,V,3,4) layouts the projector accepts.
    """
    return torch.matmul(P_nominal, T_obj)


def params_to_Pmot(theta: torch.Tensor, P_nom: torch.Tensor) -> torch.Tensor:
    """(V,6) motion + (V,3,4) nominal geometry -> (V,3,4) motion-applied geometry.

    Differentiable in `theta`. This is the single seam through which motion enters BOTH the
    forward model and the reconstruction:

        y      = forward_project_3d(x_true, params_to_Pmot(theta_true, P_nom))   # simulate
        x_corr = fdk(y,             params_to_Pmot(theta_hat,  P_nom))           # correct
    """
    return apply_rigid_motion(P_nom, rigid_motion_matrices(theta))


# --------------------------------------------------------------------------------------
# ground-truth motion profiles (simulation)
# --------------------------------------------------------------------------------------

def _profile(kind: str, s: torch.Tensor, amp: float, cycles: float, phase: float) -> torch.Tensor:
    """One DoF's trajectory over the normalized scan time s in [0, 1]."""
    if kind == "sinusoid":
        return amp * torch.sin(2 * math.pi * cycles * s + phase)
    if kind == "linear":                       # slow drift: the pathological case for a
        return amp * (2.0 * s - 1.0)           # Fourier-basis estimator, hence a default test
    if kind == "jerk":                         # a single abrupt settle (cough / swallow)
        return amp * torch.tanh(8.0 * (s - 0.5 - 0.1 * phase / math.pi))
    if kind == "step":                         # discrete repositioning partway through
        return amp * torch.where(s > 0.45 + 0.1 * phase / math.pi,
                                 torch.ones_like(s), -torch.ones_like(s))
    raise ValueError(f"unknown motion profile: {kind}")


def make_motion(
    kind: str,
    n_views: int,
    *,
    trans_mm: tuple[float, float, float] = (3.0, 3.0, 2.0),
    rot_deg: tuple[float, float, float] = (1.5, 1.5, 2.0),
    cycles: float = 1.5,
    device="cpu",
    dtype=torch.float32,
    seed: int | None = None,
) -> torch.Tensor:
    """Ground-truth per-view rigid motion, (V, 6) with rotations in RADIANS.

    Amplitudes default to a head-and-neck scale, not a thorax one: a few mm of translation and
    a couple of degrees of rotation, which is the range reported for intra-scan head motion
    (and the range AI_Geocal's tanh bounds allow: 10 mm / 10 deg). `rot_deg` is given in
    degrees for legibility and converted here -- everything downstream is radians.

    z (SI) gets a smaller translation and the LARGEST rotation on purpose: nodding/shaking a
    head pivots about the SI axis far more than it slides along it.

    `kind` is **"akima" (THE STANDARD -- the literature's model, see `akima_motion`)**, or one of
    sinusoid | linear | jerk | step, or "mixed" to draw a different profile per DoF. The
    non-akima kinds are OURS, not the field's; they exist to keep us honest (a Fourier-basis
    estimator scores beautifully on a sinusoid because it has been handed the answer), but a
    headline number should be reported on "akima" so it is comparable to Thies et al.
    """
    if kind == "akima":
        return akima_motion(n_views, trans_mm=max(trans_mm), rot_deg=max(rot_deg),
                            device=device, dtype=dtype, seed=seed)

    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)

    s = torch.arange(n_views, dtype=torch.float64) / max(n_views - 1, 1)
    amps = list(trans_mm) + [math.radians(d) for d in rot_deg]

    kinds = ["sinusoid", "linear", "jerk", "step"]
    theta = torch.zeros(n_views, 6, dtype=torch.float64)
    for d in range(6):
        k = kinds[torch.randint(len(kinds), (1,), generator=g).item()] if kind == "mixed" else kind
        phase = float(torch.rand(1, generator=g).item() * 2 * math.pi) if seed is not None else 0.0
        cyc = cycles * (1.0 + 0.3 * float(torch.rand(1, generator=g).item() - 0.5)) if seed is not None else cycles
        theta[:, d] = _profile(k, s, amps[d], cyc, phase)

    return theta.to(device=device, dtype=dtype)


def akima_motion(
    n_views: int,
    *,
    n_nodes: int = 10,
    trans_mm: float = 5.0,
    rot_deg: float = 5.0,
    zero_centre: bool = True,
    device="cpu",
    dtype=torch.float32,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """**THE FIELD'S STANDARD head-motion model** (Thies et al., IEEE TMI 2025, arXiv:2401.09283).

        "the rigid motion state of each projection is unrelated to that of the neighboring
         views. In a typical CBCT scan, however, multiple projection images are acquired per
         second. Hence, we impose an additional temporal smoothness constraint on the motion
         patterns by fitting an Akima spline to each of the six motion parameters separately."

    An independent **Akima spline** per DoF through `n_nodes` evenly spaced nodes, one at each end
    of the scan. Thies use **10 nodes to SIMULATE** motion and 30 to ESTIMATE it; the evaluation
    amplitude is **5 mm / 5 deg** (they train their quality metric on a wider 10 mm / 15 deg).
    Splines are ZERO-CENTRED, which is their handling of the unobservable global pose -- see
    `motion_error`, which fits the SE(3) gauge properly rather than assuming it is the mean.

    Physically: head motion is "slow oscillations as well as uniformly increasing deviations from
    an initial position" (Wagner et al., Invest Radiol 2003), and at ~5 ms exposure per projection
    it is INTER-frame only, never intra-frame.

    THIS MOVES FASTER THAN OUR OWN `sinusoid` PROFILE, and that matters. 10 nodes over 360 views
    is one node per 40 views, so adjacent nodes can swing 10 deg in 40 views: **0.22-0.37 deg per
    view of rotation**, against 0.13 for a 1.5-cycle sinusoid at the same 5 deg amplitude. The
    faster the rotation about the gantry axis, the more unevenly the effective view angles are
    spaced -- which is exactly what `geometry_3d.view_angular_weights` exists to correct (worth
    +1.21 dB on average over five draws of THIS profile).

    Needs scipy (`Akima1DInterpolator`); scipy is already a hard dependency of the repo.
    """
    import numpy as np
    from scipy.interpolate import Akima1DInterpolator

    if seed is not None:
        rng = np.random.default_rng(seed)
    elif generator is not None:
        rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,),
                                                      generator=generator).item()))
    else:
        rng = np.random.default_rng()

    tn = np.linspace(0.0, n_views - 1, n_nodes)
    v = np.arange(n_views, dtype=np.float64)
    cols = []
    for d in range(6):
        amp = trans_mm if d < 3 else math.radians(rot_deg)
        s = Akima1DInterpolator(tn, rng.uniform(-amp, amp, n_nodes))(v)
        cols.append(s - s.mean() if zero_centre else s)
    return torch.tensor(np.stack(cols, -1), device=device, dtype=dtype)


def random_motion(
    n_views: int,
    *,
    trans_mm: float = 5.0,
    rot_deg: float = 3.0,
    cycles: tuple[float, float] = (0.5, 3.0),
    kind: str = "akima",
    n_nodes: int = 10,
    device="cpu",
    dtype=torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Random per-view motion for TRAINING the flow-matching prior, (V, 6) [mm | rad].

    `kind="akima"` (the DEFAULT) is the literature's model -- see `akima_motion`. Train the prior
    on the same motion family the field evaluates on, or the bridge it learns is not the bridge
    inference walks.

    `kind="sinusoid"` is the old per-DoF random-amplitude / random-frequency / random-phase
    sinusoid, the 3D twin of the 2D `RandomMotionConfig`. Kept for the ablation only: it is
    SMOOTHER than real head motion (0.13 vs 0.22-0.37 deg/view of rotation at 5 deg amplitude),
    so a prior trained on it has never seen how fast a head can actually turn.
    """
    if kind == "akima":
        return akima_motion(n_views, n_nodes=n_nodes, trans_mm=trans_mm, rot_deg=rot_deg,
                            device=device, dtype=dtype, generator=generator)
    if kind != "sinusoid":
        raise ValueError(f"kind must be akima|sinusoid, got {kind!r}")

    def U(lo, hi, n):
        return lo + (hi - lo) * torch.rand(n, generator=generator)

    s = torch.arange(n_views, dtype=torch.float32)[:, None] / max(n_views - 1, 1)   # (V,1)
    amp = torch.cat([U(-trans_mm, trans_mm, 3),
                     U(-math.radians(rot_deg), math.radians(rot_deg), 3)])          # (6,)
    cyc = U(cycles[0], cycles[1], 6)
    ph = U(0.0, 2 * math.pi, 6)
    theta = amp[None, :] * torch.sin(2 * math.pi * cyc[None, :] * s + ph[None, :])   # (V,6)
    return theta.to(device=device, dtype=dtype)


# --------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------
def motion_error(theta_hat: torch.Tensor, theta_true: torch.Tensor, *,
                 fit_gauge: bool = True, gauge_iters: int = 400,
                 cfg=None) -> dict[str, float]:
    """Per-view rigid motion error -- translation [mm], rotation [deg] -- with the SE(3) GAUGE
    QUOTIENTED OUT BY FITTING IT.

    Blind motion correction cannot see a global rigid pose: replacing the object by `G x` and every
    view's transform by `T_v G^-1` leaves the measurements bit-for-bit identical. So theta_hat only
    ever estimates theta_true UP TO one unknown right-multiplied G, and a metric that does not
    remove G is reporting the gauge, not the estimator.

    So G is FITTED: the 6 parameters minimizing sum_v ||T_hat_v G - T_true_v||^2, and the residual
    is reported as `trans_rmse_mm` / `rot_rmse_deg`. The raw numbers come back too; the gap between
    them is the gauge. VERIFIED by injecting a known gauge (T_v <- T_v G, |t_g| = 5.59 mm,
    |w_g| = 3.53 deg): the raw error reads 5.59 mm / 3.53 deg and the fitted residual is 0.0000.

    `trans_rmse_mm_meansub` is the cheaper thing one is tempted to do instead -- subtract the mean
    translation offset. It is kept only so the two can be compared. Under T'_v = T_v G^-1 the
    translations shift by -R_v R_g^T t_g, which depends on the VIEW through R_v, so mean-subtraction
    is exact only in the limit of small per-view rotations. MEASURED on the same injected gauge, it
    leaves 0.16 mm of the 5.59 -- i.e. for the few-degree motions this project targets it is a good
    approximation, and NOT the explanation for a residual of a millimetre or more. If the fitted
    residual is still large, the error is real (or lives in a weakly-observable direction, such as
    translation along the beam axis, which cone-beam geometry constrains only through magnification).
    """
    t_h, R_h = theta_hat[:, :3], so3_exp(theta_hat[:, 3:])
    t_t, R_t = theta_true[:, :3], so3_exp(theta_true[:, 3:])

    def err(t, R):
        ang = so3_log(R.transpose(-1, -2) @ R_t).norm(dim=-1)
        return (float((t - t_t).pow(2).sum(-1).mean().sqrt()),
                float(torch.rad2deg(ang.pow(2).mean().sqrt())))

    raw_t, raw_r = err(t_h, R_h)
    d = t_h - t_t
    out = {
        "trans_rmse_mm_raw": raw_t,
        "rot_rmse_deg_raw": raw_r,
        "trans_rmse_mm_meansub": float((d - d.mean(0, keepdim=True)).pow(2).sum(-1).mean().sqrt()),
    }
    if not fit_gauge:
        out["trans_rmse_mm"], out["rot_rmse_deg"] = raw_t, raw_r
        return out

    # T_hat_v @ G ~= T_true_v   =>   R_h R_g ~= R_t  and  R_h t_g + t_h ~= t_t
    g = torch.zeros(6, device=theta_hat.device, requires_grad=True)
    opt = torch.optim.Adam([g], lr=0.05)
    for _ in range(gauge_iters):
        opt.zero_grad(set_to_none=True)
        Rg = so3_exp(g[None, 3:])[0]
        Rc = R_h @ Rg
        tc = (R_h @ g[:3][None, :, None])[..., 0] + t_h
        # Chordal (Frobenius) distance on SO(3) keeps the two blocks commensurate without a
        # hand-tuned weight; /100 puts mm and radians on a comparable scale.
        loss = ((Rc - R_t) ** 2).sum(dim=(-1, -2)).mean() + ((tc - t_t) ** 2).sum(-1).mean() / 100.0
        loss.backward()
        opt.step()

    with torch.no_grad():
        Rg = so3_exp(g[None, 3:])[0]
        t_c = (R_h @ g[:3][None, :, None])[..., 0] + t_h
        R_c = R_h @ Rg
    ct, cr = err(t_c, R_c)
    out["trans_rmse_mm"], out["rot_rmse_deg"] = ct, cr
    out["gauge_trans_mm"] = float(g[:3].detach().norm())
    out["gauge_rot_deg"] = float(torch.rad2deg(g[3:].detach().norm()))
    if cfg is not None:
        out.update(_beam_frame_split(t_c - t_t, cfg, theta_hat.device))
    return out


def _beam_frame_split(d: torch.Tensor, cfg, device) -> dict[str, float]:
    """Split a per-view translation residual (V,3) into the PER-VIEW BEAM FRAME.

    THIS IS THE SPLIT THAT MATTERS, and quoting a single translation RMSE instead of it is
    actively misleading. A cone beam barely sees translation along its OWN AXIS: sliding the object
    1 mm toward the source at SOD = 1000 mm changes the magnification by 0.1% and changes nothing
    else. That direction is very weakly observable, it ROTATES WITH THE GANTRY, and it is NOT the
    SE(3) gauge that `motion_error` already fits out -- it is a genuine ill-conditioning of the
    cone-beam forward model, and no amount of iterating will remove it.

    MEASURED (hash-MLP + LNCC, oracle image, 2000 iters, sinusoid motion): of a 1.18 mm
    fitted-gauge translation residual, 92.3% of the energy lay along the beam axis, 7.1% lateral,
    0.6% axial. The estimator had in fact converged -- 0.32 mm laterally, 0.09 mm axially -- and
    its reconstruction sat 0.4 dB from the true-theta oracle, which a real 1.2 mm error could not
    do. Driving `trans_rmse_mm` to zero means chasing a quantity the measurements do not contain.

    Frame, for view angle b (source at C = SOD*(cos b, sin b, 0)):
        e_depth = -C/|C|,  toward the isocentre   -- the weakly observed one
        e_lat   = in-plane, perpendicular to it   -- well observed; it is detector u
        e_axial = +z (SI)                         -- well observed; it is detector v
    """
    V = d.shape[0]
    b = cfg.angle_start + cfg.angle_span * torch.arange(
        V, device=device, dtype=torch.float32) / float(V)
    if cfg.clockwise:
        b = -b
    cb, sb = torch.cos(b), torch.sin(b)
    z = torch.zeros(V, device=device)
    e_dep = torch.stack([-cb, -sb, z], -1)
    e_lat = torch.stack([-sb, cb, z], -1)
    e_ax = torch.stack([z, z, torch.ones_like(z)], -1)

    def rms(e):
        return float((d * e).sum(-1).pow(2).mean().sqrt())

    dep, lat, ax = rms(e_dep), rms(e_lat), rms(e_ax)
    return {
        "trans_depth_mm": dep,                                # expect this to stay large
        "trans_lat_mm": lat,
        "trans_axial_mm": ax,
        "trans_obs_mm": float((lat ** 2 + ax ** 2) ** 0.5),   # <- the honest headline
    }
