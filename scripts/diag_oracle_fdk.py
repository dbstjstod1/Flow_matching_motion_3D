"""Why does the ORACLE (t=1, the TRUE Pmat) not reach static quality? Four experiments.

The premise under test: "if you know Pmat exactly you can reconstruct perfectly." That is true of
an EXACT inverse. FDK is not one -- it is an ANALYTIC inverse DERIVED FOR A CIRCULAR ORBIT, and
per-view rigid motion makes the effective source trajectory non-circular. Two steps of our FDK
assume the circle and cannot see Pmat at all:

    * the ramp is filtered along `u`, always. FDK's derivation puts the filter along the
      trajectory's tangent, which for a circle IS u.
    * the final `angle_span / V` is a UNIFORM angular weight, i.e. equiangular views. A rotation
      about the gantry axis z changes the EFFECTIVE view angle (beta_eff = beta_v - rz_v), so the
      views stop being equiangular and this weight is simply wrong.

Everything that CAN read Pmat does: the 1/w^2 distance weight takes w from `P @ X`, and the cosine
pre-weight depends only on (u, v, SDD), which rigid motion leaves invariant. So there is no bug to
find in the P-dependent half -- the question is how much the circle-dependent half costs.

  [A] GLOBAL rigid motion (the SAME T on every view). The trajectory stays a circle, just moved,
      and the views stay equiangular. If FDK is sound, this must reconstruct as well as static.
      If it does NOT, the fault is a bug, not the geometry.
  [B] Per-axis decomposition: which DoF actually costs the dB? Prediction: rotation about z
      (the gantry axis) is the expensive one, translation is free.
  [C] ITERATIVE least-squares from the FDK image with the TRUE Pmat. If Pmat is exactly right,
      an exact inverse must climb back toward the static ceiling. THIS is the decisive test of the
      user's premise -- and of whether the remaining gap is FDK's or the geometry's.
  [D] PER-VIEW residual: does the oracle fail uniformly across views, or on particular ones?
      Correlated against |d rz / d view| -- the quantity the uniform angular weight ignores.

    python scripts/diag_oracle_fdk.py
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/CQ500")
    ap.add_argument("--out", default="data/diag_oracle")
    ap.add_argument("--shape", type=int, nargs=3, default=(256, 256, 256))
    ap.add_argument("--views", type=int, default=360)
    ap.add_argument("--cg_iters", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)
    cfg = ConeBeam3DConfig.thies(n_views=args.views)
    gen = CQ500Generator(args.root, cfg=cfg, device=dev, split="train",
                         shape=tuple(args.shape), voxel_mm=1.0, verbose=False)
    meas = measured_region_mask(gen.shape, (gen.dz, gen.dy, gen.dx), cfg, device=dev)
    gt = gen.volume(0)
    g3 = gt[0, 0]
    V = cfg.n_views

    with torch.no_grad():
        y0 = gen.simulate(0, gen.P_nom[None])
        static = gen.fdk(y0, gen.P_nom[None])[0]
    p_static = psnr(static, g3, meas)
    print(f"STATIC FDK (the ceiling)                      {p_static:6.2f} dB\n", flush=True)

    def oracle(theta, tag):
        """simulate with theta, reconstruct with the SAME theta -> the oracle recon."""
        with torch.no_grad():
            y = gen.simulate(0, params_to_Pmot(theta, gen.P_nom)[None])
            x = gen.fdk(y, params_to_Pmot(theta, gen.P_nom)[None])[0]
        p = psnr(x, g3, meas)
        print(f"  {tag:<42s} {p:6.2f} dB   ({p - p_static:+.2f} vs static)", flush=True)
        return x, y, p

    # ---- [A] GLOBAL rigid motion: the orbit stays a circle -------------------------------
    print("[A] GLOBAL rigid motion (same T on every view -> the orbit is STILL a circle)")
    print("    If FDK is sound this costs ~nothing. A large loss here would mean a BUG.")
    g_const = torch.zeros(V, 6, device=dev)
    g_const[:, :3] = torch.tensor([5.0, -3.0, 2.0], device=dev)          # 5 mm-ish translation
    g_const[:, 3:] = torch.tensor([0.0, 0.0, 5.0 * np.pi / 180], device=dev)   # 5 deg about z
    oracle(g_const, "global rigid: 6 mm + 5 deg about z")
    gz = torch.zeros(V, 6, device=dev)
    gz[:, 5] = 5.0 * np.pi / 180
    oracle(gz, "global rigid: 5 deg about z ONLY")

    # ---- [B] per-axis: which DoF costs the dB? -------------------------------------------
    print("\n[B] PER-VIEW motion, one DoF at a time (sinusoid, 1.5 cycles over the scan)")
    base = make_motion("sinusoid", V, device=dev, seed=args.seed,
                       trans_mm=(10.0,) * 3, rot_deg=(10.0,) * 3)
    for k, name in enumerate(["tx (5 mm)", "ty (5 mm)", "tz (5 mm)",
                              "rx (5 deg)", "ry (5 deg)", "rz (5 deg, THE GANTRY AXIS)"]):
        th = torch.zeros_like(base)
        th[:, k] = base[:, k]
        oracle(th, name)
    print("    ---")
    th_t = base.clone(); th_t[:, 3:] = 0
    oracle(th_t, "ALL translation, no rotation")
    th_r = base.clone(); th_r[:, :3] = 0
    oracle(th_r, "ALL rotation, no translation")
    x_full, y_full, p_full = oracle(base, "full 6-DoF (5 mm / 5 deg)")

    # ---- [C] the decisive test: an EXACT inverse with the TRUE Pmat ------------------------
    print(f"\n[C] ITERATIVE least-squares with the TRUE Pmat, {args.cg_iters} gradient steps")
    print("    Does an EXACT inverse, handed the exact geometry, climb back to the ceiling?")
    P_true = params_to_Pmot(base, gen.P_nom)[None]
    x = x_full.clone()                                              # start from the oracle FDK
    # simple preconditioned gradient descent on 0.5||A x - y||^2 (A is the ray-march projector)
    for it in range(args.cg_iters):
        xr = x.detach().requires_grad_(True)
        r = gen.project(xr[None, None], P_true) - y_full
        loss = 0.5 * (r ** 2).sum()
        loss.backward()
        gr = xr.grad
        with torch.no_grad():
            # exact line search along -g:  step = <g,g> / ||A g||^2
            Ag = gen.project(gr[None, None], P_true)
            step = (gr * gr).sum() / (Ag * Ag).sum().clamp_min(1e-30)
            x = (x - step * gr).detach()
        if (it + 1) % 3 == 0 or it == 0:
            print(f"    it {it + 1:3d}  {psnr(x, g3, meas):6.2f} dB   "
                  f"(FDK oracle {p_full:.2f}, static {p_static:.2f})", flush=True)

    # ---- [D] per-view residual ------------------------------------------------------------
    print("\n[D] PER-VIEW residual of the ORACLE FDK image, vs the motion's angular rate")
    with torch.no_grad():
        y_hat = gen.project(x_full[None, None], P_true)
        res = ((y_hat - y_full) ** 2).mean(dim=(2, 3))[0].sqrt()     # (V,)
    rz = base[:, 5] * 180 / np.pi
    drz = torch.zeros_like(rz)
    drz[1:-1] = (rz[2:] - rz[:-2]) / 2.0                             # deg per view
    dbeta = 360.0 / V
    r = res.cpu().numpy(); d = drz.abs().cpu().numpy()
    cc = float(np.corrcoef(r, d)[0, 1])
    print(f"    view residual: min {r.min():.4g}  mean {r.mean():.4g}  max {r.max():.4g} "
          f"(max/mean {r.max() / r.mean():.2f}x)")
    print(f"    |d rz / d view|: max {d.max():.3f} deg  vs the UNIFORM angular step "
          f"{dbeta:.3f} deg  ->  up to {100 * d.max() / dbeta:.0f}% of it")
    print(f"    corr(residual, |d rz/d view|) = {cc:+.3f}")
    print("    (the `angle_span/V` weight assumes d beta_eff/d view is CONSTANT; rz makes it not)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    ax[0].plot(r, lw=1)
    ax[0].set_ylabel("per-view RMS residual")
    ax[0].set_title(f"oracle FDK, true Pmat | corr with |drz/dview| = {cc:+.3f}", fontsize=10)
    ax[1].plot(d, lw=1, color="crimson")
    ax[1].axhline(dbeta, ls="--", c="k", lw=0.8, label=f"uniform step {dbeta:.2f} deg")
    ax[1].set_ylabel("|d rz / d view| [deg]"); ax[1].set_xlabel("view"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "per_view.png"), dpi=110)
    print(f"    -> {args.out}/per_view.png")


if __name__ == "__main__":
    main()
