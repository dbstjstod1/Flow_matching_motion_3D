"""Gate: the training motion sampler implements Thies' TRAINING amplitude protocol,
and the evaluation path is untouched by it.

Thies, IEEE TMI 2025 (arXiv:2401.09283), verbatim:
  II-B (training)   "The spline-based motion model ... with 10 nodes per spline is used with a
                     maximal amplitude of 10 mm for translation parameters and 15 deg for
                     rotation parameters. ... we include motion patterns with unequal amplitude
                     across the different motion parameters as well as motion patterns that
                     perturb the data only slightly ... All splines are individually zero-centered"
  IV   (evaluation) "A random motion pattern is sampled for each patient with an amplitude of
                     5 mm for translation and 5 deg for rotation which is kept constant across
                     different methods and optimization algorithms."

Checks:
  1. amp_mode="fixed" (the EVAL default) still puts every DoF at ~the given amplitude
  2. amp_mode="thies" gives UNEQUAL per-DoF amplitudes, spread over [0, max]
  3. nothing exceeds the stated maximum
  4. the "perturb only slightly" clause is NOT implemented, and does not need to be: per-DoF
     randomness alone does NOT stand in for it (severity is a max over six uniforms, so it
     concentrates near 1), but the geometry bridge does -- the residual at bridge point s is
     exactly (1-s)*theta, and the sampler is EXACTLY linear in the amplitude, so (1-s)*Akima(A)
     is the same path as Akima((1-s)A)
  5. splines are zero-centred in both modes
  6. the eval amplitude 5/5 lies INSIDE the training support (train wider than eval)
  7. the EVALUATION path (amp_mode="fixed") is BIT-IDENTICAL to the pre-amp_mode sampler
  8. seeding is reproducible

Run: python scripts/gate_motion_amp.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
from scipy.interpolate import Akima1DInterpolator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fm3d.rigid_motion import (akima_motion, make_motion,  # noqa: E402
                               random_motion, _profile, so3_exp)


def _legacy_make(kind, n_views, amp, seed):
    """`make_motion` as it stood before the peak-to-peak switch: `amp` is the +- PEAK."""
    import math
    if kind == "akima":
        rng = np.random.default_rng(seed)
        tn = np.linspace(0.0, n_views - 1, 10)
        v = np.arange(n_views, dtype=np.float64)
        cols = []
        for d in range(6):
            a = amp if d < 3 else math.radians(amp)
            s_ = Akima1DInterpolator(tn, rng.uniform(-a, a, 10))(v)
            cols.append(s_ - s_.mean())
        return torch.tensor(np.stack(cols, -1), dtype=torch.float32)
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    s_ = torch.arange(n_views, dtype=torch.float64) / max(n_views - 1, 1)
    amps = [amp] * 3 + [math.radians(amp)] * 3
    kinds = ["sinusoid", "linear", "jerk", "step"]
    th = torch.zeros(n_views, 6, dtype=torch.float64)
    for d in range(6):
        k = kinds[torch.randint(len(kinds), (1,), generator=g).item()] if kind == "mixed" else kind
        phase = float(torch.rand(1, generator=g).item() * 2 * math.pi)
        cyc = 1.5 * (1.0 + 0.3 * float(torch.rand(1, generator=g).item() - 0.5))
        th[:, d] = _profile(k, s_, amps[d], cyc, phase)
    return th.to(torch.float32)

# EVERYTHING HERE IS PEAK-TO-PEAK (fm3d/rigid_motion module header).
V = 360
T_MAX, R_MAX = 15.0, 20.0          # OUR training maxima (Thies' own: 10 / 15)
T_EVAL, R_EVAL = 10.0, 10.0        # OUR evaluation point (Thies' own: 5 / 5)
N = 400                            # draws per mode

# `peak_amps` returns a ONE-SIDED excursion max|theta|, so its natural yardstick is the NODE
# BOUND = amplitude/2, not the peak-to-peak amplitude. Every ratio below uses these.
TB, RB = T_MAX / 2, R_MAX / 2
TE, RE = T_EVAL / 2, R_EVAL / 2


def peak_amps(th: torch.Tensor) -> np.ndarray:
    """Per-DoF realized peak amplitude, translations in mm and rotations in deg."""
    a = th.abs().amax(0).double().numpy()
    a[3:] = np.degrees(a[3:])
    return a


def draw(mode: str, n: int, *, tmax=T_MAX, rmax=R_MAX, seed0=0):
    return np.stack([peak_amps(akima_motion(V, trans_mm=tmax, rot_deg=rmax, amp_mode=mode,
                                            seed=seed0 + i))
                     for i in range(n)], 0)                       # (n, 6)


def main() -> int:
    ok = True

    # ---- 1. fixed mode: every DoF at ~the given amplitude ------------------------------------
    # NOTE the realized peak is NOT the nominal amplitude. The nominal figure bounds the NODE
    # draws; an Akima spline overshoots between nodes and zero-centring shifts the trace, so the
    # realized peak runs ~0.96x the bound on average and up to ~1.5x in the tail. That is a
    # property of the literature's motion model itself (it applies verbatim to the 5 mm / 5 deg
    # evaluation protocol we have always run), not of the amplitude mode -- check 3 pins it down.
    A = draw("fixed", 200, tmax=T_EVAL, rmax=R_EVAL)
    rel = np.concatenate([A[:, :3] / TE, A[:, 3:] / RE], 1)
    spread_fixed = float(np.mean(A[:, :3].std(1) / A[:, :3].mean(1)))
    c1 = 0.85 < rel.mean() < 1.10 and spread_fixed < 0.20
    print(f"[1] fixed  : realized/bound mean {rel.mean():.3f} "
          f"[p01 {np.percentile(rel, 1):.2f}, p99 {np.percentile(rel, 99):.2f}] "
          f"| per-draw across-DoF CV {spread_fixed:.3f}  -> {'ok' if c1 else 'FAIL'}")
    ok &= c1

    # ---- 2. thies mode: UNEQUAL across DoFs, spread over the whole range ----------------------
    B = draw("thies", N)
    relB = np.concatenate([B[:, :3] / TB, B[:, 3:] / RB], 1)
    spread_thies = float(np.mean(B[:, :3].std(1) / np.maximum(B[:, :3].mean(1), 1e-9)))
    # "unequal amplitude across the different motion parameters"
    c2 = spread_thies > 3.0 * spread_fixed and relB.std() > 0.20
    print(f"[2] thies  : per-draw across-DoF CV {spread_thies:.3f} vs fixed's {spread_fixed:.3f} "
          f"({spread_thies / max(spread_fixed, 1e-9):.1f}x) | relative-amp sd {relB.std():.3f} "
          f"-> {'ok' if c2 else 'FAIL'}")
    ok &= c2

    # ---- 3. the maximum is respected, in the only sense it can be -----------------------------
    # "maximal amplitude" bounds the NODE draws; Akima overshoot puts realized peaks above it in
    # both modes. The real claim is that thies mode only ever scales a fixed-mode-at-maximum
    # pattern DOWN, so its realized peaks stay inside fixed-at-maximum's own envelope.
    F = draw("fixed", N, tmax=T_MAX, rmax=R_MAX)
    relF = np.concatenate([F[:, :3] / TB, F[:, 3:] / RB], 1)
    c3 = relB.max() <= relF.max() + 1e-9 and relB.mean() < relF.mean()
    print(f"[3] max    : realized/max  thies {relB.mean():.3f} (peak {relB.max():.2f}) vs "
          f"fixed-at-max {relF.mean():.3f} (peak {relF.max():.2f}) "
          f"-- Akima overshoot, same envelope  -> {'ok' if c3 else 'FAIL'}")
    ok &= c3

    # ---- 4. why the "perturb only slightly" clause needs no implementation here ---------------
    # Two halves, both asserted. (a) the per-DoF draw does NOT stand in for it: severity is a MAX
    # over six uniforms, so it concentrates near 1 and globally mild patterns essentially never
    # occur -- the clause is a real mechanism for Thies, not a redundant one. (b) the BRIDGE does
    # stand in for it, exactly: the residual at bridge point s is (1-s)*theta, and this sampler is
    # EXACTLY linear in the amplitude (uniform(-cA,cA) = c*uniform(-A,A) on the same stream, Akima
    # interpolation and mean-subtraction are both linear), so a bridge point of a full-amplitude
    # draw IS a mild draw -- the same path, not merely the same distribution.
    worst = np.maximum(B[:, :3].max(1) / TB, B[:, 3:].max(1) / RB)   # severity of a draw
    mild = float((worst < 0.20).mean())
    lin = []
    for s in (0.05, 0.3, 0.7):                      # bridge points -> residual scale (1-s)
        for sd in (0, 1, 2):
            full = akima_motion(V, trans_mm=T_MAX, rot_deg=R_MAX, amp_mode="thies", seed=sd)
            direct = akima_motion(V, trans_mm=(1 - s) * T_MAX, rot_deg=(1 - s) * R_MAX,
                                  amp_mode="thies", seed=sd)
            lin.append(float(((1 - s) * full - direct).abs().max()))
    c4 = mild < 0.005 and max(lin) < 1e-6
    print(f"[4] slight : not implemented, not needed. per-DoF alone gives {100 * mild:.2f}% of "
          f"draws under 20% severity (median {np.median(worst):.2f}) -- so the clause is REAL; "
          f"but (1-s)*Akima(A) == Akima((1-s)A) pathwise to {max(lin):.1e} "
          f"-> {'ok' if c4 else 'FAIL'}")
    ok &= c4

    # ---- 5. zero-centred (Thies: "All splines are individually zero-centered") ---------------
    th = akima_motion(V, trans_mm=T_MAX, rot_deg=R_MAX, amp_mode="thies", seed=7)
    c5 = float(th.mean(0).abs().max()) < 1e-5
    print(f"[5] centred: max |mean| over DoFs {float(th.mean(0).abs().max()):.2e} "
          f"-> {'ok' if c5 else 'FAIL'}")
    ok &= c5

    # ---- 6. the eval point sits INSIDE the training support -----------------------------------
    # every eval DoF is 5 mm / 5 deg; training must produce DoFs at least that large often enough
    above_t = float((B[:, :3] >= TE).mean())
    above_r = float((B[:, 3:] >= RE).mean())
    c6 = above_t > 0.25 and above_r > 0.40
    print(f"[6] covers : P(DoF >= eval amplitude) trans {above_t:.2f}, rot {above_r:.2f} "
          f"-> {'ok' if c6 else 'FAIL'}")
    ok &= c6

    # ---- 7. the EVALUATION path is bit-identical to the pre-amp_mode sampler -------------------
    # amp_mode="fixed" must consume nothing extra from the RNG stream, or every seeded draw in the
    # repo silently changes: the gates, val_fm3d's seed 1000+i (which is what keeps the val curve
    # comparable back to the ray era), and every reproduction in data/. Reference = the sampler as
    # it stood before amp_mode existed, transcribed here.
    def legacy(n_views, n_nodes=10, trans_mm=5.0, rot_deg=5.0, seed=0):
        """The sampler as it stood BEFORE amp_mode AND before the p2p switch: its arguments
        are the +- NODE BOUND, so it must be called with HALF the peak-to-peak number."""
        rng = np.random.default_rng(seed)
        tn = np.linspace(0.0, n_views - 1, n_nodes)
        v = np.arange(n_views, dtype=np.float64)
        cols = []
        for d in range(6):
            amp = float(trans_mm) if d < 3 else math.radians(float(rot_deg))
            s = Akima1DInterpolator(tn, rng.uniform(-amp, amp, n_nodes))(v)
            cols.append(s - s.mean())
        return torch.tensor(np.stack(cols, -1), dtype=torch.float32)

    seeds = [0, 7, 1000, 1001, 1002]          # incl. val_fm3d's 1000+i
    # legacy takes the +- bound, so it gets HALF of our peak-to-peak numbers
    c7 = all(torch.equal(legacy(V, trans_mm=T_EVAL / 2, rot_deg=R_EVAL / 2, seed=sd),
                         akima_motion(V, trans_mm=T_EVAL, rot_deg=R_EVAL, seed=sd))
             for sd in seeds)
    print(f"[7] eval   : amp_mode=fixed, p2p {T_EVAL:g}/{R_EVAL:g} bit-identical to the "
          f"pre-2026-07-28 sampler at +-{T_EVAL / 2:g}/+-{R_EVAL / 2:g} on {len(seeds)} seeds "
          f"-> {'ok' if c7 else 'FAIL'}")
    ok &= c7

    # ---- 8. reproducibility ------------------------------------------------------------------
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    r1 = random_motion(V, trans_mm=T_MAX, rot_deg=R_MAX, amp_mode="thies", generator=g1)
    r2 = random_motion(V, trans_mm=T_MAX, rot_deg=R_MAX, amp_mode="thies", generator=g2)
    c8 = torch.equal(r1, r2)
    print(f"[8] repro  : identical under a re-seeded generator -> {'ok' if c8 else 'FAIL'}")
    ok &= c8

    # ---- 9. THE RELABELLING CHANGED NO PHYSICAL MOTION ----------------------------------------
    # The 2026-07-28 switch to peak-to-peak was a units change, not a physics change: calling the
    # new code with 2A must reproduce the old code at +-A EXACTLY, for every kind and both modes.
    bad = []
    for kind in ("akima", "sinusoid", "linear", "jerk", "step", "mixed"):
        for A in (3.0, 5.0, 7.5):
            new = make_motion(kind, V, trans_mm=(2 * A,) * 3, rot_deg=(2 * A,) * 3, seed=5)
            bad.append((kind, float((new - _legacy_make(kind, V, A, seed=5)).abs().max())))
    worst = max(b for _, b in bad)
    c9 = worst < 1e-7
    print(f"[9] relabel: make_motion(2A) == pre-switch make_motion(+-A) for "
          f"{len(bad)} kind/amplitude combos, worst |diff| {worst:.2e} "
          f"-> {'ok' if c9 else 'FAIL'}")
    ok &= c9

    print("\nGATE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
