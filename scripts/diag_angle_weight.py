"""Is the oracle FDK's rotation loss just the UNIFORM ANGULAR WEIGHT? Test the fix.

`fdk_conebeam_3d_batched` ends with `recon * angle_span / V` -- one weight for every view, i.e.
"the views are equiangular". Rigid motion about the GANTRY AXIS z breaks exactly that and nothing
else: rotating the object about z maps the source circle onto ITSELF, so the trajectory is still a
circle and the ramp is still along u -- the views simply stop being evenly spaced. Measured cost:
-2.01 dB, against 0.00 for every translation (`diag_oracle_fdk.py` [B]).

THE FIX: read the effective view angle out of the projection matrix itself. The source S is the
null vector of P (P @ [S;1] = 0), so beta_v = atan2(S_y, S_x), and the correct FDK weight for view
v is its own angular share d beta_v (central difference), not angle_span/V.

Applying it needs no change to the FDK: the weight is a per-view scalar and the reconstruction is
linear in the sinogram, so pre-scaling view v of the sinogram by (d beta_v) / (angle_span / V) and
then letting the FDK apply its uniform weight yields EXACTLY the per-view weighting.

    python scripts/diag_angle_weight.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.dataset_cq500 import CQ500Generator
from fm3d.geometry_3d import ConeBeam3DConfig, measured_region_mask
from fm3d.rigid_motion import make_motion, params_to_Pmot


def psnr(a, b, m):
    e = (a - b)[m]
    rng = float(b[m].max() - b[m].min())
    return float(20 * np.log10(rng / (e.pow(2).mean().sqrt().item() + 1e-12)))


def source_positions(P: torch.Tensor) -> torch.Tensor:
    """(V,3) source positions: the null space of each (3,4) projection matrix.

    P maps a world point to a homogeneous detector coordinate; the one world point that projects
    to the degenerate [0,0,0] is the source itself. So S is the right null vector of P, taken as
    the last right-singular vector and dehomogenized."""
    _, _, Vh = torch.linalg.svd(P.double())          # (V,3,4) -> Vh (V,4,4)
    n = Vh[:, -1, :]                                 # (V,4)
    # Divide by the homogeneous component AS IT IS. Do not clamp it: the sign of a singular
    # vector is arbitrary and differs from view to view, so `clamp(min=eps)` would map a
    # perfectly good negative w to +1e-12 and send that view's source to ~1e12 mm, with the
    # wrong sign. (It did: the translation cases read -32 dB until this was fixed.)
    if (n[:, 3].abs() < 1e-9).any():
        raise RuntimeError("a projection matrix has its source at infinity (parallel beam?)")
    return n[:, :3] / n[:, 3:4]


def angular_weights(P: torch.Tensor, angle_span: float) -> torch.Tensor:
    """(V,) per-view angular share d beta_v, from the ACTUAL source trajectory in P."""
    S = source_positions(P)
    beta = torch.atan2(S[:, 1], S[:, 0])                              # (V,)
    b = np.unwrap(beta.cpu().numpy())                                 # remove the 2pi jumps
    d = np.empty_like(b)
    d[1:-1] = (b[2:] - b[:-2]) / 2.0
    d[0] = b[1] - b[0]
    d[-1] = b[-1] - b[-2]
    return torch.tensor(np.abs(d), device=P.device, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/CQ500")
    ap.add_argument("--shape", type=int, nargs=3, default=(256, 256, 256))
    ap.add_argument("--views", type=int, default=360)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda"
    cfg = ConeBeam3DConfig.thies(n_views=args.views)
    gen = CQ500Generator(args.root, cfg=cfg, device=dev, split="train",
                         shape=tuple(args.shape), voxel_mm=1.0, verbose=False)
    meas = measured_region_mask(gen.shape, (gen.dz, gen.dy, gen.dx), cfg, device=dev)
    gt = gen.volume(0)
    g3 = gt[0, 0]
    V = cfg.n_views
    uni = cfg.angle_span / V                       # the weight the FDK applies today

    # sanity: on the NOMINAL orbit the per-view weight must already BE the uniform one
    w_nom = angular_weights(gen.P_nom, cfg.angle_span)
    print(f"nominal orbit: d beta = {w_nom.mean():.6f} +- {w_nom.std():.2e} rad "
          f"(uniform {uni:.6f})  -> ratio {float(w_nom.mean()) / uni:.6f}")

    with torch.no_grad():
        static = gen.fdk(gen.simulate(0, gen.P_nom[None]), gen.P_nom[None])[0]
    p_static = psnr(static, g3, meas)
    print(f"STATIC FDK ceiling                 {p_static:6.2f} dB\n")
    print(f"{'motion':<34s} {'FDK (uniform)':>14s} {'FDK (dbeta from P)':>20s} {'gain':>7s}")

    base = make_motion("sinusoid", V, device=dev, seed=args.seed,
                       trans_mm=(10.0,) * 3, rot_deg=(10.0,) * 3)
    cases = {}
    for k, name in enumerate(["tx (5 mm)", "rx (5 deg)", "rz (5 deg, GANTRY AXIS)"]):
        idx = {0: 0, 1: 3, 2: 5}[k]
        th = torch.zeros_like(base)
        th[:, idx] = base[:, idx]
        cases[name] = th
    th_r = base.clone(); th_r[:, :3] = 0
    cases["ALL rotation"] = th_r
    cases["full 6-DoF (5 mm / 5 deg)"] = base
    mixed = make_motion("mixed", V, device=dev, seed=args.seed,
                        trans_mm=(10.0,) * 3, rot_deg=(10.0,) * 3)
    cases["full 6-DoF, 'mixed' profile"] = mixed

    for name, th in cases.items():
        P = params_to_Pmot(th, gen.P_nom)
        with torch.no_grad():
            y = gen.simulate(0, P[None])
            x_uni = gen.fdk(y, P[None])[0]
            w = angular_weights(P, cfg.angle_span)                  # (V,)
            y_w = y * (w / uni)[None, :, None, None]                # exact per-view reweighting
            x_fix = gen.fdk(y_w, P[None])[0]
        a, b = psnr(x_uni, g3, meas), psnr(x_fix, g3, meas)
        print(f"{name:<34s} {a:8.2f} dB    {b:14.2f} dB   {b - a:+6.2f}")

    print(f"\n(static ceiling {p_static:.2f} dB)")


if __name__ == "__main__":
    main()
