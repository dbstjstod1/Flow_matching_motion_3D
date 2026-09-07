"""Gate the DATA BRIDGE (`train_fm3d --bridge data`, the default since 2026-08-05).

    x_t = FDK( A(x; P_nom @ T((1-t)*theta)), P_nom )

The whole reason it replaced the anchored geometry bridge is that its two endpoints are exact by
CONSTRUCTION rather than by a detrend, so those are the first two things to check -- if either
drifts, the switch has bought nothing. The third check is the tangent's WIRING through the whole
chain: the KERNEL itself (`leap_forward_tangent`, the exact s-derivative of LEAP's pinned Joseph
forward) is certified against a float64 autograd jvp in gate_leap_forward_tangent (rel 2e-6);
what that gate cannot see is everything around it -- bridge_P_and_dP's dP/ds, the d/dt = -d/ds
sign, to_net_tangent's gain-only map, and the FDK applied to the sinogram derivative (linear to
2.5e-3: `ohnesorge_pad`'s clamp is the only nonlinearity and the head never reaches the panel
edge, measured 2026-08-05).

  D1  t=0 endpoint  == FDK(y_theta, P_nom), the inference cold start, to the bit.
  D2  t=1 endpoint  == the static FDK (gen.static_anchor_net), to the bit.
  D3  tangent       the deployed analytic path against two fp32 counterparties: the retained fd
                    mode (same chain, crude derivative -- catches a silently switched default)
                    and the path secant (derivative-of-the-chain vs chain-of-the-derivative --
                    catches a sign/scale/wiring error). Both bars are set by the COUNTERPARTIES'
                    truncation, not the kernel's precision; the h-sweep 0.05..0.00625 showed
                    cos converging 0.9959 -> 0.99925, so the residual at any fixed h is the
                    chord's, not the tangent's.
  D4  path monotone: the distance to the static FDK decreases in t (the path is a bridge, not a
                    detour), and the t=0 point is the same image the geometry bridge starts from.

    python scripts/gate_bridge_data.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fm3d.dataset_cq500 import CQ500Generator
from fm3d.geometry_3d import ConeBeam3DConfig
from fm3d.rigid_motion import params_to_Pmot, random_motion
from train_fm3d import bridge_pair_data

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
    # generator= is NOT redundant with torch.manual_seed: the Akima nodes come from numpy
    # (`akima_motion` builds an np.random.default_rng, UNSEEDED unless a generator is passed
    # to derive the seed from), so without it this gate drew a DIFFERENT motion every run and
    # D3's truncation-limited bars turned it into a coin flip -- observed 1-then-2 failures
    # on back-to-back runs, 2026-08-05. Same mechanism the trainer's --seed uses.
    th = random_motion(cfg.n_views, trans_mm=15.0, rot_deg=20.0, amp_mode="thies",
                       device=dev, generator=torch.Generator().manual_seed(0))

    with torch.no_grad():
        print("\nD1/D2  endpoints (exact by construction -- that is the point of this bridge)")
        x0, _ = bridge_pair_data(gen, idx, torch.tensor(0.0), th)
        cold = gen.to_net(gen.fdk(gen.simulate(idx, params_to_Pmot(th, gen.P_nom)[None]),
                                  gen.P_nom[None])[0])
        check("t=0 == FDK(y_theta, P_nom), the cold start", rel(x0, cold) < 1e-6,
              f"rel {rel(x0, cold):.3e}")

        x1, _ = bridge_pair_data(gen, idx, torch.tensor(1.0), th)
        check("t=1 == the static FDK", rel(x1, gen.static_anchor_net(idx)) < 1e-6,
              f"rel {rel(x1, gen.static_anchor_net(idx)):.3e}")

        print("\nD3  the ANALYTIC tangent (the deployed path) through the whole chain")
        t0 = 0.5
        _, dx = bridge_pair_data(gen, idx, torch.tensor(t0), th)          # mode="analytic"
        _, dx_fd = bridge_pair_data(gen, idx, torch.tensor(t0), th, mode="fd", delta=0.02)
        r = rel(dx, dx_fd)
        cosf = float(torch.sum(dx * dx_fd) / (torch.linalg.vector_norm(dx)
                                              * torch.linalg.vector_norm(dx_fd) + 1e-12))
        # The KERNEL is certified against a float64 jvp in gate_leap_forward_tangent (rel 2e-6);
        # here the fd path is the crude counterparty, so the bar is ITS error, not the kernel's.
        # If these two ever agreed to 1e-3 something has silently switched the default back.
        check("analytic vs fd(0.02) within fd's own error", r < 8e-2, f"rel {r:.3e}")
        check("analytic vs fd(0.02) direction", cosf > 0.99, f"cos {cosf:.6f}")

        h = 0.05
        xp, _ = bridge_pair_data(gen, idx, torch.tensor(t0 + h), th)
        xm, _ = bridge_pair_data(gen, idx, torch.tensor(t0 - h), th)
        sec = (xp - xm) / (2 * h)
        cos = float(torch.sum(dx * sec) / (torch.linalg.vector_norm(dx)
                                           * torch.linalg.vector_norm(sec) + 1e-12))
        # The chord over h = 0.05 carries its own truncation, so this checks the tangent points
        # ALONG the path -- it is a wiring check (right sign, right scale), not a precision one.
        check("tangent aligns with the path secant", cos > 0.99, f"cos {cos:.5f}")
        check("tangent magnitude within the secant's own truncation", rel(dx, sec) < 2e-1,
              f"rel {rel(dx, sec):.3e}  (the h=0.05 chord is itself ~10% off)")

        print("\nD4  the path is a bridge (distance to the static FDK decreases in t)")
        stat = gen.static_anchor_net(idx)
        ds = []
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            xt, _ = bridge_pair_data(gen, idx, torch.tensor(t), th)
            ds.append(float(torch.sqrt(torch.mean((xt - stat) ** 2)) / 2.0 * 100.0))
        mono = all(b < a for a, b in zip(ds, ds[1:]))
        check("monotone toward the static FDK", mono,
              "  ".join(f"t={t:.2f} {d:.2f}%" for t, d in zip((0, .25, .5, .75, 1.0), ds)))

    print(f"\n{'ALL GATES PASS' if n_fail == 0 else f'{n_fail} GATE(S) FAILED'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
