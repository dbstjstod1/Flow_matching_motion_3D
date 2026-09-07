"""Gate the LINEAR BRIDGE (`train_fm3d --bridge linear`, the bridge-shape ablation arm,
2026-08-11).

    x_t = (1-t)*x_0 + t*x_1,   x_0 = FDK(y_theta, P_nom),   x_1 = the static FDK

The arm's ENTIRE claim is "same endpoints as the data bridge, only the path differs" -- so the
gate checks exactly that claim plus the wiring mistakes a pixel-line bridge can still hide
(a flipped t convention scores IDENTICALLY on train loss early on; only the endpoints expose it).
The math between the endpoints is an affine map, so unlike gate_bridge_data there is no
truncation-limited bar anywhere: every check here is at float precision.

  L1  t=0 endpoint  == FDK(y_theta, P_nom) computed independently, to the bit (catches a wrong
                     y or a double to_net).
  L2  t=1 endpoint  == the static FDK memo (catches the FLIPPED t convention -- the one error
                     this bridge makes silently).
  L3  cross-arm     both endpoints match the DATA bridge's to float noise -- the comparability
                     claim the ablation rests on ("the arms differ ONLY in the path").
  L4  tangent       dx == x_1 - x_0 == the path secant EXACTLY (the path is a line; any
                     discrepancy is wiring, not truncation).

    python scripts/gate_bridge_linear.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fm3d.dataset_cq500 import CQ500Generator
from fm3d.geometry_3d import ConeBeam3DConfig
from fm3d.rigid_motion import params_to_Pmot, random_motion
from train_fm3d import bridge_pair_data, bridge_pair_linear

ROOT = "data/CQ500"
n_fail = 0


def check(name, ok, detail=""):
    global n_fail
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}   {detail}")
    if not ok:
        n_fail += 1


def rel(a, b):
    return float(torch.linalg.vector_norm(a - b) / (torch.linalg.vector_norm(b) + 1e-12))


def main():
    torch.manual_seed(0)
    dev = "cuda"
    cfg = ConeBeam3DConfig.thies(n_views=360)
    gen = CQ500Generator(ROOT, cfg, device=dev, split="val", shape=(256, 256, 256),
                         voxel_mm=1.0, sim_native=True, verbose=False)
    idx = 0
    # generator= is NOT redundant with torch.manual_seed -- the Akima nodes come from numpy and
    # are unseeded otherwise; see the identical note in gate_bridge_data (2026-08-05 flake).
    th = random_motion(cfg.n_views, trans_mm=15.0, rot_deg=20.0, amp_mode="thies",
                       device=dev, generator=torch.Generator().manual_seed(0))
    y = gen.simulate(idx, params_to_Pmot(th, gen.P_nom)[None])

    with torch.no_grad():
        print("\nL1/L2  endpoints (exact by definition -- failures here are wiring)")
        x0, dx = bridge_pair_linear(gen, idx, torch.tensor(0.0), y)
        cold = gen.to_net(gen.fdk(y, gen.P_nom[None])[0])
        check("t=0 == FDK(y_theta, P_nom), the cold start", rel(x0, cold) < 1e-7,
              f"rel {rel(x0, cold):.3e}")

        x1, _ = bridge_pair_linear(gen, idx, torch.tensor(1.0), y)
        stat = gen.static_anchor_net(idx)
        check("t=1 == the static FDK (t convention not flipped)", rel(x1, stat) < 1e-7,
              f"rel {rel(x1, stat):.3e}")

        print("\nL3  cross-arm: the DATA bridge's endpoints are THESE endpoints")
        d0, _ = bridge_pair_data(gen, idx, torch.tensor(0.0), th)
        d1, _ = bridge_pair_data(gen, idx, torch.tensor(1.0), th)
        check("t=0 matches --bridge data t=0", rel(x0, d0) < 1e-6, f"rel {rel(x0, d0):.3e}")
        check("t=1 matches --bridge data t=1", rel(x1, d1) < 1e-6, f"rel {rel(x1, d1):.3e}")

        print("\nL4  tangent: constant, == x_1 - x_0, == the path secant (a line has no "
              "truncation)")
        check("dx == x_1 - x_0", rel(dx, x1 - x0) < 1e-6, f"rel {rel(dx, x1 - x0):.3e}")
        xa, dxa = bridge_pair_linear(gen, idx, torch.tensor(0.25), y)
        xb, dxb = bridge_pair_linear(gen, idx, torch.tensor(0.75), y)
        sec = (xb - xa) / 0.5
        check("dx == the secant over [0.25, 0.75]", rel(dx, sec) < 1e-5,
              f"rel {rel(dx, sec):.3e}")
        check("dx is t-independent", rel(dxa, dxb) < 1e-7, f"rel {rel(dxa, dxb):.3e}")
        mid = 0.5 * (x0 + x1)
        xm, _ = bridge_pair_linear(gen, idx, torch.tensor(0.5), y)
        check("t=0.5 is the pixel midpoint", rel(xm, mid) < 1e-6, f"rel {rel(xm, mid):.3e}")

    print(f"\n{'ALL GATES PASS' if n_fail == 0 else f'{n_fail} GATE(S) FAILED'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
