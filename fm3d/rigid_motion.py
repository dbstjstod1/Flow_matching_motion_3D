"""Per-view rigid (6-DoF) patient motion, composed into the projection matrices.

THE PARAMETERIZATION
--------------------
    theta: (V, 6) = [tx, ty, tz | wx, wy, wz]
        translation  [mm]        (world axes: x, y in the orbit plane; z = SI = rotation axis)
        rotation     [rad]       ROTATION VECTOR (axis-angle), i.e. an element of so(3)

AMPLITUDES ARE PEAK-TO-PEAK, EVERYWHERE IN THIS REPO
----------------------------------------------------
Every `trans_mm` / `rot_deg` argument, CLI flag and stored run-arg is the **full swing** of the
motion parameter, not a +- bound. `akima_motion` therefore draws its spline nodes from
`uniform(-amp/2, +amp/2)`. This is Thies' convention, established from their released sampler
(`refs/thies_moco_diff_likelihood/motion_compensation_data_loader.py`), which draws
`(rand(n) - 0.5) * amplitude` -- so their paper's "5 mm / 5 deg" evaluation is nodes in +-2.5.
It is cross-checked against a figure the paper states independently: at that setting the
UNCOMPENSATED median RPE is "around 3 mm" (TMI L568), and this code reproduces 3.05 mm.

**Converted on 2026-07-28** (before that, the same arguments meant the +- node bound, i.e. HALF
the number they mean now). Anything reading amplitudes back out of a stored run predates the
switch unless its args carry `amp_units="p2p"` -- use `amp_from_run_args` to read them, which
doubles the legacy values. `scripts/gate_motion_amp.py` check 9 asserts that `p2p = 2A` is
bit-identical to the pre-switch `+-A`, i.e. that this was a RELABELLING and no physical motion
changed.

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


def bridge_P_and_dP(theta: torch.Tensor, P_nom: torch.Tensor,
                    s: float) -> tuple[torch.Tensor, torch.Tensor]:
    """P(s) = params_to_Pmot(s*theta, P_nom) and its EXACT derivative dP/ds, in closed form.

    This is what makes the analytic bridge tangent possible: because the bridge scales the
    AXIS-ANGLE vector (the SO(3) geodesic -- see the module docstring for why that
    parameterization was chosen), R(s) = exp(s*skew(w)) shares its generator across s, so

        dR/ds = skew(w) @ R(s)          (exact -- same generator, so it commutes)
        dt/ds = t                       (translation is the straight line s*t)

    and dP/ds = P_nom @ dT/ds with dT/ds = [[skew(w) @ R(s), t], [0, 0]] (bottom row ZERO --
    dT/ds is a tangent, not a rigid transform). With Euler angles no such closed form exists
    per component; this identity is the analytic twin of the finite-difference argument in
    `train_fm3d.bridge_pair`. Gated against a float64 central difference in
    scripts/gate_fdk_tangent.py.
    """
    if theta.shape[-1] != 6:
        raise ValueError(f"theta must be (..., 6); got {tuple(theta.shape)}")
    t = theta[..., :3]
    w = theta[..., 3:]
    R = so3_exp(float(s) * w)                                # (..., 3, 3)
    K = skew(w)
    Vshape = theta.shape[:-1]
    T = torch.zeros(*Vshape, 4, 4, device=theta.device, dtype=theta.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = float(s) * t
    T[..., 3, 3] = 1.0
    dT = torch.zeros_like(T)
    dT[..., :3, :3] = K @ R
    dT[..., :3, 3] = t
    return torch.matmul(P_nom, T), torch.matmul(P_nom, dT)


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
    trans_mm: float | tuple[float, float, float] = (6.0, 6.0, 4.0),
    rot_deg: float | tuple[float, float, float] = (3.0, 3.0, 4.0),
    cycles: float = 1.5,
    device="cpu",
    dtype=torch.float32,
    seed: int | None = None,
) -> torch.Tensor:
    """Ground-truth per-view rigid motion, (V, 6) with rotations in RADIANS.

    **Amplitudes are PEAK-TO-PEAK** (see the module header). Defaults are a head-and-neck scale,
    not a thorax one: a few mm of translation and a couple of degrees of rotation, which is the
    range reported for intra-scan head motion. `rot_deg` is given in degrees for legibility and
    converted here -- everything downstream is radians.

    z (SI) gets a smaller translation and the LARGEST rotation on purpose: nodding/shaking a
    head pivots about the SI axis far more than it slides along it.

    `kind` is **"akima" (THE STANDARD -- the literature's model, see `akima_motion`)**, or one of
    sinusoid | linear | jerk | step, or "mixed" to draw a different profile per DoF. The
    non-akima kinds are OURS, not the field's; they exist to keep us honest (a Fourier-basis
    estimator scores beautifully on a sinusoid because it has been handed the answer), but a
    headline number should be reported on "akima" so it is comparable to Thies et al.
    """
    if kind == "akima":
        # Per-axis amplitudes apply here too (this used to collapse the tuples via max(),
        # which silently broke the "z gets a smaller translation" contract for the default
        # kind). Isotropic tuples -- what every current caller passes -- are bit-identical
        # to the old behaviour: numpy's uniform(low, high, n) consumes the same underlying
        # stream regardless of the bounds, so only the per-DoF SCALING changes.
        return akima_motion(n_views, trans_mm=trans_mm, rot_deg=rot_deg,
                            device=device, dtype=dtype, seed=seed)

    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)

    s = torch.arange(n_views, dtype=torch.float64) / max(n_views - 1, 1)
    # SCALARS ARE ISOTROPIC, as they already are on the `akima` path (`akima_motion`'s own
    # signature takes `float | tuple`). Without this the non-akima kinds raised
    # `TypeError: 'float' object is not iterable` for every caller that passes a plain number --
    # which is what `run_posterior3d.build_world` does for ALL kinds, so `--motion_kind mixed`
    # (and sinusoid/linear/jerk/step) crashed before the first projection. That included the
    # reproduction recipe printed in run_posterior3d's own --motion_kind comment.
    def _triple(a):
        return [float(a)] * 3 if isinstance(a, (int, float)) else [float(v) for v in a]

    # `_profile` scales by the PEAK, and the arguments are peak-to-peak -> halve.
    amps = [0.5 * a for a in _triple(trans_mm)] + \
           [0.5 * math.radians(d) for d in _triple(rot_deg)]

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
    trans_mm: float | tuple[float, float, float] = 10.0,
    rot_deg: float | tuple[float, float, float] = 10.0,
    zero_centre: bool = True,
    amp_mode: str = "fixed",
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

    `trans_mm` / `rot_deg` accept a scalar (isotropic, the literature's 5 mm / 5 deg) or a
    per-axis (x, y, z) tuple; the scalar path is bit-identical to the historical behaviour.

    `amp_mode` selects what those amplitudes MEAN, and this is the train/eval split Thies draw:

      "fixed"  (DEFAULT, the EVALUATION protocol) -- every DoF gets exactly the given amplitude.
               Thies IV: *"A random motion pattern is sampled for each patient with an amplitude
               of 5 mm for translation and 5 deg for rotation which is kept constant across
               different methods and optimization algorithms."* Never change this for a reported
               number: it is what makes our SSIM comparable to their 0.94.

      "thies"  (the TRAINING protocol, OUR reading) -- the given amplitudes are a MAXIMUM and
               each DoF draws its own fraction of it. Thies II-B: *"The spline-based motion model
               with 10 nodes per spline is used with a maximal amplitude of 10 mm for translation
               parameters and 15 deg for rotation parameters. To ensure that all motion states
               that could be encountered during optimization are represented in the training
               data, we include motion patterns with unequal amplitude across the different
               motion parameters as well as motion patterns that perturb the data only
               slightly."*
               Realized as `a_d = A_d * u_d`, `u_d ~ U(0,1)` INDEPENDENTLY PER DoF -- the "unequal
               amplitude" clause only. This is what OUR PRIOR was trained with
               (`logs/fm3d_cq500_leap`); its meaning is frozen for reproducibility.

      "thies_hn" (the TRAINING protocol, THEIR RELEASED SAMPLER -- use this for the Thies bench)
               -- `a_d = A_d * min(|N(0, u_d)|, 1)`, `u_d ~ U(0,1)` per DoF: a CLIPPED
               HALF-NORMAL whose std is itself uniform. Transcribed from their released 2D
               sibling (`refs/thies_moco_diff_likelihood/autofocus_data_set.py:64-87`:
               `min(abs(normal(0., max_amp*rand(1))), max_amp)`), which is the closest public
               code to the TMI paper's unreleased supplementary pseudo-code. Its mass near zero
               IS the paper's third clause, *"motion patterns that perturb the data only
               slightly"*: measured over 2e5 draws, severity (max frac over 6 DoF) < 0.5 in
               12.5% of draws vs 1.5% under "thies" -- an 8x difference, and it is exactly the
               mid-optimization states Eq. 6 walks through. (Their file also carries a
               radians-vs-degrees unit bug, `0.26 rad` passed as degrees; NOT propagated --
               see the 2D PROVENANCE.)

    THE THIRD CLAUSE IS DELIBERATELY NOT IMPLEMENTED IN "thies" (our prior's mode) -- and the
    reason is not that it is trivially covered. It is a real mechanism
    in Thies' setting: a pattern's severity is the MAX over six independent uniforms, which
    concentrates near 1, so the per-DoF draw above produces globally mild patterns essentially
    never (MEASURED over 4000 draws: median severity 0.86, **0.00%** below 20% severity, 0.10%
    below 30%). For a network that sees ONE static corrupted volume per sample, the mild regime is
    unreachable without an explicit injection -- hence their "as well as".
    OUR GEOMETRY BRIDGE SUPPLIES IT EXACTLY. `bridge_P_and_dP` reconstructs at `P(s*theta)` while
    the data was formed at `P(theta)`, and the axis-angle scaling shares its generator, so the
    residual motion at bridge point s is EXACTLY `(1-s)*theta`. With t (hence s) drawn uniform,
    every single draw sweeps the residual amplitude uniformly over `[0, |theta|]` -- and since a
    uniform-node Akima pattern scales linearly, `(1-s)*Akima(A)` is distributed exactly as
    `Akima((1-s)A)`. Not approximately: identically. The mild states are present in every draw,
    not in a sampled fraction of them.
    An injection would also be actively WRONG for us: it puts mild images at LOW s, while low s in
    the posterior loop always carries LARGE residual motion (the loop's t advances in step with
    the correction, and our estimator lags rather than leads -- it is the measured weak link, and
    c2f deliberately stays coarse until t=0.5). It would spend training on states off the
    trajectory inference actually walks. Revisit ONLY if the estimator ever converges FASTER than
    the ODE, which would put small residuals at small t for real.
    (Implemented as a `p_slight` knob on 2026-07-28, then DELETED the same day once the bridge
    equivalence above was worked out -- a knob whose only justification is fidelity to a mechanism
    we already have structurally. Gate `gate_motion_amp.py` check 4 asserts the equivalence.)
    THE THIES BENCH IS THE OPPOSITE CASE: its frozen quality net has no bridge -- one static
    corrupted volume per sample IS its whole world -- so there the clause must be explicit,
    which is what "thies_hn" is for (`bench/thies/data.py` uses it as of 2026-08-06).

    WHY THE TRAINING MODE MATTERS HERE and not only for Thies' quality-metric net: a common
    amplitude bound is not the same as equal realized amplitudes, but it nearly is -- the max of
    10 uniform node draws lands at ~0.9 A, so all six DoFs come out within ~10% of each other.
    The state our posterior loop actually lives in is the opposite: the residual after a few
    steps is ~92% concentrated in the beam-axis translation (see `motion_error` / the
    depth-unobservability finding), i.e. five DoFs nearly correct and one badly wrong. Under
    "fixed" the prior never sees that. Note also that our estimator's own tanh bounds are
    15 mm / 8 deg, so intermediate motion states genuinely can exceed the 5 mm / 5 deg the
    evaluation starts from -- which is Thies' stated rationale, and it transfers.

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
    # `trans_mm`/`rot_deg` may be a scalar (isotropic -- the training default, bit-identical
    # to the historical behaviour) or a per-axis 3-tuple. numpy's uniform(low, high, n)
    # advances the stream identically whatever the bounds, so the scalar path's RNG sequence
    # is untouched by this generalization.
    t_amp = trans_mm if isinstance(trans_mm, (tuple, list)) else (trans_mm,) * 3
    r_amp = rot_deg if isinstance(rot_deg, (tuple, list)) else (rot_deg,) * 3

    # Per-DoF amplitude fractions. amp_mode="fixed" consumes NOTHING from the stream, so it stays
    # BIT-IDENTICAL to every seeded draw this repo has ever made (gates, val_fm3d's seed 1000+i,
    # every reproduction in data/); "thies" consumes exactly uniform(6) as it always has, so the
    # deployed prior's draws are likewise untouched. "thies_hn" consumes uniform(6) + normal(6).
    if amp_mode == "fixed":
        frac = np.ones(6)
    elif amp_mode == "thies":
        frac = rng.uniform(0.0, 1.0, 6)
    elif amp_mode == "thies_hn":
        # Their released sampler: amp = min(|N(0, A*u)|, A), u ~ U(0,1) per DoF, normalized by A.
        # The half-normal's mass near zero is the paper's "perturb the data only slightly".
        frac = np.minimum(np.abs(rng.normal(0.0, rng.uniform(0.0, 1.0, 6))), 1.0)
    else:
        raise ValueError(f"amp_mode must be fixed|thies|thies_hn, got {amp_mode!r}")

    cols = []
    for d in range(6):
        amp = float(t_amp[d]) if d < 3 else math.radians(float(r_amp[d - 3]))
        # AMPLITUDES ARE PEAK-TO-PEAK (see the module header): the node bound is HALF of it.
        # This is Thies' convention -- their released sampler is `(rand(n) - 0.5) * amplitude`.
        half = 0.5 * amp * float(frac[d])
        s = Akima1DInterpolator(tn, rng.uniform(-half, half, n_nodes))(v)
        cols.append(s - s.mean() if zero_centre else s)
    return torch.tensor(np.stack(cols, -1), device=device, dtype=dtype)


AMP_UNITS = "p2p"          # stamped into saved run args; see `amp_from_run_args`


def amp_from_run_args(a: dict) -> tuple[float | None, float | None]:
    """Read (trans_mm, rot_deg) out of a stored run/checkpoint arg dict, IN PEAK-TO-PEAK.

    Runs saved before 2026-07-28 stored the +- node bound under the same key, so their numbers
    are HALF of what the same key means now. They are recognised by the absence of `amp_units`
    and doubled here. Always go through this instead of `a["trans_mm"]` when reproducing the
    motion of a finished run -- reading a legacy run raw silently halves its amplitude.
    """
    t, r = a.get("trans_mm"), a.get("rot_deg")
    if a.get("amp_units") != AMP_UNITS:                      # legacy: half-range
        t = None if t is None else 2.0 * float(t)
        r = None if r is None else 2.0 * float(r)
    return t, r


def random_motion(
    n_views: int,
    *,
    trans_mm: float = 10.0,
    rot_deg: float = 6.0,
    cycles: tuple[float, float] = (0.5, 3.0),
    kind: str = "akima",
    n_nodes: int = 10,
    amp_mode: str = "fixed",
    device="cpu",
    dtype=torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Random per-view motion for TRAINING the flow-matching prior, (V, 6) [mm | rad].

    `kind="akima"` (the DEFAULT) is the literature's model -- see `akima_motion`. Train the prior
    on the same motion family the field evaluates on, or the bridge it learns is not the bridge
    inference walks.

    `amp_mode="thies"` additionally adopts the paper's TRAINING amplitude protocol (per-DoF
    unequal fractions of a maximum, plus occasional slight perturbations) -- see `akima_motion`
    for what that means and why the "fixed" default is the right thing for EVALUATION. The
    trainer passes "thies"; every evaluation path leaves it at "fixed".

    `kind="sinusoid"` is the old per-DoF random-amplitude / random-frequency / random-phase
    sinusoid, the 3D twin of the 2D `RandomMotionConfig`. Kept for the ablation only: it is
    SMOOTHER than real head motion (0.13 vs 0.22-0.37 deg/view of rotation at 5 deg amplitude),
    so a prior trained on it has never seen how fast a head can actually turn.
    """
    if kind == "akima":
        return akima_motion(n_views, n_nodes=n_nodes, trans_mm=trans_mm, rot_deg=rot_deg,
                            amp_mode=amp_mode, device=device, dtype=dtype, generator=generator)
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

    g, t_c, R_c = _fit_gauge(t_h, R_h, t_t, R_t, gauge_iters)
    ct, cr = err(t_c, R_c)
    out["trans_rmse_mm"], out["rot_rmse_deg"] = ct, cr
    out["gauge_trans_mm"] = float(g[:3].detach().norm())
    out["gauge_rot_deg"] = float(torch.rad2deg(g[3:].detach().norm()))
    if cfg is not None:
        out.update(_beam_frame_split(t_c - t_t, cfg, theta_hat.device))
    return out


def _fit_gauge(t_h, R_h, t_t, R_t, iters: int = 400):
    """Fit the unobservable global pose G minimizing sum_v ||T_hat_v G - T_true_v||^2.

    Returns (g, t_corrected, R_corrected) with g the 6-vector [t | w] of G. Shared by
    `motion_error` and `reprojection_error` so the two never disagree about what the gauge is.
    """
    # dtype FROM THE INPUT, not the default. An fp64 caller (the RPE comparison runs in double --
    # the raw RPE is a difference of two nearly-equal projected point sets) otherwise dies in the
    # first matmul, and a silently-fp32 gauge would cap the very digit being compared.
    g = torch.zeros(6, device=t_h.device, dtype=t_h.dtype, requires_grad=True)
    opt = torch.optim.Adam([g], lr=0.05)
    for _ in range(iters):
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
    return g, t_c, R_c


def trajectory_roughness(theta: torch.Tensor, theta_true: torch.Tensor | None = None,
                         hf_cycle: int = 15) -> dict[str, float]:
    """How JITTERY is a recovered trajectory, independent of how ACCURATE it is.

    `rot_rmse_deg` cannot see this. Two estimators can land on the same RMS error with completely
    different error STRUCTURE: one smoothly offset, one tracking the truth while shaking. They are
    not equally good for the reconstruction -- a per-view jitter scatters each view's rays
    independently, which an FDK cannot average away, while a smooth offset is close to a gauge.
    This is the axis on which a spline/GD estimator (Thies') is expected to differ from our
    full-bandwidth hash-MLP, whose encoder deliberately carries FAR more bandwidth than a
    few-hundred-view trajectory contains (n_levels 16, base 16, scale 1.5 -> a finest grid of
    ~7000 cells across 360 views; the 2D project measured that that jitters).

    Two views of the same thing:
      `rough_*`   RMS of the SECOND DIFFERENCE along the view axis -- a local curvature, in mm and
                  degrees per view^2. Reported for theta_hat and, when given, for theta_true, plus
                  their ratio. A ratio near 1 means the recovered trajectory is as smooth as the
                  real one; >> 1 is jitter.
      `hf_frac_*` fraction of the trajectory's spectral energy above `hf_cycle` cycles per scan.
                  The simulator is an Akima spline through 10 nodes over 360 views, so essentially
                  ALL of its energy sits below ~5 cycles; anything an estimator puts above 15 is
                  invention, not signal.
    """
    def stats(th):
        t, w = th[:, :3], torch.rad2deg(th[:, 3:])
        d2t = t[2:] - 2 * t[1:-1] + t[:-2]
        d2w = w[2:] - 2 * w[1:-1] + w[:-2]
        f = torch.fft.rfft(th - th.mean(0, keepdim=True), dim=0).abs() ** 2   # (V/2+1, 6)
        tot = float(f.sum())
        # A trajectory that never left its zero init has no spectrum at all; reporting a
        # "high-frequency fraction" of numerical noise there would read as jitter (a frozen
        # estimator once printed hf 45.8%). Report 0 and let rot_rmse say it did nothing.
        hf = float(f[hf_cycle:].sum()) / tot if tot > 1e-20 else 0.0
        return (float(d2t.pow(2).sum(-1).mean().sqrt()),
                float(d2w.pow(2).sum(-1).mean().sqrt()), hf)

    rt, rr, hf = stats(theta)
    out = {"rough_trans_mm": rt, "rough_rot_deg": rr, "hf_frac": hf}
    if theta_true is not None:
        tt, tr, thf = stats(theta_true)
        out.update({"rough_trans_mm_true": tt, "rough_rot_deg_true": tr, "hf_frac_true": thf,
                    "rough_trans_ratio": rt / max(tt, 1e-12),
                    "rough_rot_ratio": rr / max(tr, 1e-12)})
    return out


def zero_centre_gauge(theta: torch.Tensor, iters: int = 20) -> torch.Tensor:
    """Move a trajectory onto the ZERO-MEAN gauge: return theta' with T'_v = T_v G, mean_v theta' = 0.

    THIS USES NO GROUND TRUTH, unlike the gauge fit in `motion_error`. It is the convention the
    simulator already follows -- `akima_motion(zero_centre=True)` subtracts each DoF's mean -- and,
    per that function's docstring, the one Thies et al. use to handle the unobservable global pose.
    Adopting it is therefore not a metric trick but matching the field's parameterization: a blind
    estimator cannot see G, so it may as well report the representative the reference uses.

    MEASURED (2026-07-26, akima55 baseline): our raw RPE is 2.117 mm and its radial profile is FLAT
    (1.87 / 2.00 / 2.48 across the 25/50/100 mm shells), the signature of a pure global SHIFT rather
    than a rotation error. The gauge-fitted residual is 0.627 mm. Our estimator drifts because it
    runs 2500 iterations against a reference image reconstructed with its own theta -- the pair
    drift together -- whereas Thies' 100 GD steps from zero never leave the neighbourhood of zero.

    Right-multiplication by a CONSTANT G is an exact gauge transform, so the reconstruction changes
    only by a global rigid pose and every rigidly-aligned image metric is untouched. Subtracting the
    mean of theta directly is NOT that (it is only its first-order approximation), which is why this
    solves the fixed point T_v G instead.
    """
    T = rigid_motion_matrices(theta)
    G = torch.eye(4, device=theta.device, dtype=theta.dtype)
    for _ in range(iters):
        Tc = T @ G
        m = torch.cat([Tc[:, :3, 3], so3_log(Tc[:, :3, :3])], dim=-1).mean(0)   # (6,)
        if float(m.norm()) < 1e-9:
            break
        G = G @ torch.linalg.inv(rigid_motion_matrices(m[None])[0])
    Tc = T @ G
    return torch.cat([Tc[:, :3, 3], so3_log(Tc[:, :3, :3])], dim=-1)


def sphere_points(radii=(25.0, 50.0, 100.0), n_per: int = 100, device="cpu",
                  dtype=torch.float32) -> torch.Tensor:
    """(len(radii)*n_per, 3) FIXED points on spheres around the isocentre [mm].

    Thies' RPE is defined on "a fixed set of 300 3D points with radii 25 mm, 50 mm and 100 mm
    around the isocenter" (TMI 2025, IV-B) -- 100 per shell here. The paper does not say how they
    are distributed on each shell, so we use the Fibonacci lattice: deterministic (no seed to
    report), and as close to equal-area as a fixed point set gets, which is what "a fixed set"
    has to mean for the metric to be reproducible.

    The radii matter more than the arrangement: 25 mm is deep brain, 100 mm is the skull surface,
    and a rotation error shows up ~4x larger on the outer shell. That spread is the point.
    """
    g = (1.0 + 5.0 ** 0.5) / 2.0
    i = torch.arange(n_per, device=device, dtype=torch.float64) + 0.5
    phi = torch.acos(1.0 - 2.0 * i / n_per)                 # polar, equal-area in cos
    psi = 2.0 * math.pi * i / g                             # golden-angle azimuth
    unit = torch.stack([torch.sin(phi) * torch.cos(psi),
                        torch.sin(phi) * torch.sin(psi),
                        torch.cos(phi)], dim=-1)            # (n_per, 3)
    pts = torch.cat([float(r) * unit for r in radii], dim=0)
    return pts.to(dtype)


def _project_points(P: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """(V,3,4) projection matrices, (N,3) world points [mm] -> (V,N,2) detector coords [mm].

    Our P has K = diag(SDD, SDD, 1) and principal point (0,0), and `detector_coords_3d` lays the
    panel out in the same physical millimetres, so the homogeneous divide lands directly in
    detector mm -- no pixel conversion, which is exactly the assumption `gate_coarse_est.py`
    checks from the other direction.
    """
    h = torch.einsum("vij,nj->vni", P, torch.cat([pts, torch.ones_like(pts[:, :1])], dim=-1))
    return h[..., :2] / h[..., 2:3].clamp_min(1e-8)


def reprojection_error(theta_hat: torch.Tensor, theta_true: torch.Tensor, P_nom: torch.Tensor, *,
                       radii=(25.0, 50.0, 100.0), n_per: int = 100,
                       gauge_iters: int = 400) -> dict[str, float]:
    """**RPE** -- Thies' headline motion metric (his gradient-based method reaches 0.61 mm mean).

    "computed by forward projecting a fixed set of 300 3D points with radii 25 mm, 50 mm, and
    100 mm around the isocenter onto the detector planes using the recovered and target
    geometries" (TMI 2025, IV-B). So it is a DETECTOR-DOMAIN distance in millimetres, averaged
    over points and views -- not a parameter error. That is its virtue: it weights each degree of
    freedom by how much it actually moves the data, which is why it does not reward driving the
    beam-axis translation (see `_beam_frame_split`) that the measurements barely constrain.

    TWO NUMBERS ARE RETURNED AND THEY ANSWER DIFFERENT QUESTIONS.
      `rpe_mm`        RAW -- recovered vs target geometry exactly as the paper defines it. This
                      is the ONLY one comparable to Thies' 0.61 mm.
      `rpe_mm_gauged` after fitting out the SE(3) gauge, i.e. the part of the error that is not
                      an unobservable global pose. Blind motion correction cannot see G
                      (see `motion_error`), so this is the honest estimator error -- but Thies
                      does not quotient it, so never quote this one against his number.
    Their gap is the gauge, reported as `rpe_gauge_share`.

    Also returns the per-shell means: a rotation-dominated error grows with radius, a
    translation-dominated one does not, so the three shells localise WHICH dof is failing.
    """
    dev = theta_hat.device
    pts = sphere_points(radii, n_per, device=dev, dtype=theta_hat.dtype)
    T_h = rigid_motion_matrices(theta_hat)
    T_t = rigid_motion_matrices(theta_true)
    # The gauge fit is itself an optimization, so it must run OUTSIDE no_grad -- everything
    # after it is pure evaluation.
    _, t_c, R_c = _fit_gauge(theta_hat.detach()[:, :3], so3_exp(theta_hat.detach()[:, 3:]),
                             theta_true[:, :3], so3_exp(theta_true[:, 3:]), gauge_iters)
    with torch.no_grad():
        p_t = _project_points(apply_rigid_motion(P_nom, T_t), pts)
        d_raw = (_project_points(apply_rigid_motion(P_nom, T_h), pts) - p_t).norm(dim=-1)

        T_c = torch.zeros_like(T_h)
        T_c[:, :3, :3], T_c[:, :3, 3], T_c[:, 3, 3] = R_c, t_c, 1.0
        d_g = (_project_points(apply_rigid_motion(P_nom, T_c), pts) - p_t).norm(dim=-1)

    out = {"rpe_mm": float(d_raw.mean()), "rpe_mm_median": float(d_raw.median()),
           "rpe_mm_p95": float(d_raw.flatten().quantile(0.95)),
           "rpe_mm_gauged": float(d_g.mean()), "rpe_mm_gauged_median": float(d_g.median())}
    out["rpe_gauge_share"] = float(1.0 - d_g.mean() / d_raw.mean().clamp_min(1e-12))
    # Per shell, RAW and GAUGED. The RADIAL PROFILE is a diagnosis, not decoration: a rotation
    # error displaces a point by ~r*angle, so it grows 1:2:4 across the 25/50/100 mm shells
    # (gated in scripts/gate_rpe.py), whereas a translation error is FLAT in r. A flat raw
    # profile therefore means the raw number is dominated by a global shift -- which is exactly
    # what the SE(3) gauge is -- and the gauged profile says what is left.
    for k, r in enumerate(radii):
        sl = slice(k * n_per, (k + 1) * n_per)
        out[f"rpe_mm_r{int(r)}"] = float(d_raw[:, sl].mean())
        out[f"rpe_mm_gauged_r{int(r)}"] = float(d_g[:, sl].mean())
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
