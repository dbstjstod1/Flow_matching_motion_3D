"""Why is the DATA bridge's tangent 60x noisier than the GEOM bridge's finite difference?

`gate_bridge_data.py` D3 fails: delta 0.02 vs 0.002 disagree by 6.7%, where the geometry
bridge's own `--tangent fd` sat ~1e-3 from the exact derivative. The two differences are NOT
the same object:

  geom: d/ds of a BACKprojection with a fixed sinogram -- smooth in P, 256^3 grid
  data: d/ds of a FORWARD projection through the 612^3 NATIVE volume -- every ray crosses sharp
        bone edges, and the ledger already records that LEAP's projector has GRID RIPPLE whose
        local slope differs from the trend in sign (leap-geometry-gradient-2026-07-29).

So the suspicion is a classic FD U-curve: truncation error at large delta, ripple/quantization
noise at small delta, with a plateau (or none) in between. This sweeps delta and reports each
step's disagreement with its neighbours, plus the pairwise cosines. If there is a plateau, the
FD is usable and the gate bar is simply wrong; if the curve never flattens, the data bridge needs
a real forward-projection jvp before it can be a training default.

    python scripts/diag_bridge_data_tangent.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fm3d.dataset_cq500 import CQ500Generator
from fm3d.geometry_3d import ConeBeam3DConfig
from fm3d.rigid_motion import params_to_Pmot, random_motion

DELTAS = (0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)
T0 = 0.5


def main():
    torch.manual_seed(0)
    dev = "cuda"
    cfg = ConeBeam3DConfig.thies(n_views=360)
    gen = CQ500Generator("data/CQ500", cfg, device=dev, split="val", shape=(256, 256, 256),
                         voxel_mm=1.0, sim_native=True, verbose=False)
    idx = 0
    th = random_motion(cfg.n_views, trans_mm=15.0, rot_deg=20.0, amp_mode="thies", device=dev)
    s0 = 1.0 - T0

    def sim(sv):
        return gen.simulate(idx, params_to_Pmot(sv * th, gen.P_nom)[None])

    with torch.no_grad():
        # ---- (1) the SINOGRAM difference alone: is the noise born before the FDK? ----------
        print(f"s0 = {s0}, sweeping delta.  ||dy/ds|| and its neighbour-to-neighbour change:\n")
        print(f"{'delta':>8} {'||dy/ds||':>14} {'rel vs prev':>12} {'cos vs prev':>12} "
              f"{'rel vs 0.05':>12}")
        ref = None
        prev = None
        dys = {}
        for d in DELTAS:
            sp, sm = min(s0 + d, 1.0), max(s0 - d, 0.0)
            dy = (sim(sp) - sim(sm)) / (sp - sm)
            n = float(torch.linalg.vector_norm(dy))
            r = c = float("nan")
            if prev is not None:
                r = float(torch.linalg.vector_norm(dy - prev) / n)
                c = float(torch.sum(dy * prev) / (n * torch.linalg.vector_norm(prev)))
            if d == 0.05:
                ref = dy.clone()
            rr = float("nan") if ref is None else \
                float(torch.linalg.vector_norm(dy - ref) / n)
            print(f"{d:8.4f} {n:14.4f} {r:12.4f} {c:12.6f} {rr:12.4f}")
            dys[d] = dy
            prev = dy
        # ref may have been set mid-loop; redo the vs-0.05 column now that it exists
        print(f"\n{'delta':>8} {'rel vs delta=0.05':>20} {'cos vs delta=0.05':>20}")
        ref = dys[0.05]
        nref = torch.linalg.vector_norm(ref)
        for d in DELTAS:
            dy = dys[d]
            r = float(torch.linalg.vector_norm(dy - ref) / torch.linalg.vector_norm(dy))
            c = float(torch.sum(dy * ref) / (torch.linalg.vector_norm(dy) * nref))
            print(f"{d:8.4f} {r:20.4f} {c:20.6f}")

        # ---- (1b) THE DOMAIN THAT MATTERS: the IMAGE tangent, against a Richardson ref -----
        # The training target is dx_t/dt, not dy/ds, and the FDK's ramp AMPLIFIES the fp32
        # cancellation noise the difference carries (measured: 1.9% in the sinogram becomes 4.0%
        # in the image at delta = 0.005). So the operating point must be chosen HERE.
        #
        # The reference is Richardson extrapolation from two LARGE steps:
        #     D = (4*D(h/2) - D(h)) / 3          error O(h^4) instead of O(h^2)
        # Large steps mean no cancellation at all, so this reference is both accurate and
        # quiet -- unlike simply taking delta -> 0, which trades truncation for noise.
        def img_diff(d):
            sp, sm = min(s0 + d, 1.0), max(s0 - d, 0.0)
            dy = (sim(sp) - sim(sm)) / (sp - sm)
            return -gen.to_net_tangent(gen.fdk(dy, gen.P_nom[None])[0])

        d_h, d_h2 = 0.04, 0.02
        ref_img = (4.0 * img_diff(d_h2) - img_diff(d_h)) / 3.0
        nref = torch.linalg.vector_norm(ref_img)
        print(f"\nIMAGE-domain tangent vs Richardson({d_h}, {d_h2}) reference "
              f"[the training target]:\n")
        print(f"{'delta':>8} {'rel vs ref':>12} {'cos vs ref':>12}")
        for d in DELTAS:
            di = img_diff(d)
            r = float(torch.linalg.vector_norm(di - ref_img) / nref)
            c = float(torch.sum(di * ref_img) / (torch.linalg.vector_norm(di) * nref))
            print(f"{d:8.4f} {r:12.4f} {c:12.6f}")
        # and the reference's own self-consistency: a second Richardson pair one octave down
        ref2 = (4.0 * img_diff(0.01) - img_diff(0.02)) / 3.0
        print(f"\nRichardson(0.04,0.02) vs Richardson(0.02,0.01): rel "
              f"{float(torch.linalg.vector_norm(ref2 - ref_img) / nref):.4f}  "
              f"cos {float(torch.sum(ref2 * ref_img) / (torch.linalg.vector_norm(ref2) * nref)):.6f}"
              f"   <- if this is small, Richardson at 5 sims/draw IS the exact-enough tangent")

        # ---- (2) is it RIPPLE (non-smooth in s) or CURVATURE (smooth but bending)? ---------
        # A ripple shows up as a sinogram that is NOT monotone in s over a tiny interval.
        print("\nsinogram along a TINY s interval (ripple probe): ||y(s) - y(s0)|| for s-s0 =")
        y0 = sim(s0)
        for e in (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05):
            dp = float(torch.linalg.vector_norm(sim(s0 + e) - y0))
            dm = float(torch.linalg.vector_norm(sim(s0 - e) - y0))
            # a SMOOTH path has dp/dm -> 1 and both -> linear in e
            print(f"  e={e:7.4f}   +{dp:11.4f}   -{dm:11.4f}   ratio {dp / (dm + 1e-12):7.4f}   "
                  f"(+)/e {dp / e:11.2f}   (-)/e {dm / e:11.2f}")


if __name__ == "__main__":
    main()
