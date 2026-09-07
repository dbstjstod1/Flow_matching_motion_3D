"""Export our test30 cohort measurements for the JRM-ADM solver (A2c stage 1, 2026-08-17).

One case per (split=test, run=i, seed=1000+i), THE cohort triple every benchmark here uses
(chain_databridge_eval, cmp_thies_vs_ours) -- y/theta come from `run_posterior3d.build_world`,
so JRM-ADM sees byte-identical measurements to every other arm, and `theta_true` rides along
for the pairing assertion the cmp scripts enforce.

GEOMETRY BRIDGE TO THEIR SOLVER (measured, scripts/diag_jrm_operator_xcheck.py, 2026-08-17):
our LEAP forward and their torch-radon ConeBeam agree to rel 0.16% / scale 1.0003 with NO
u/v flips and NO direction reversal; the ONLY delta is the angle origin -- their angle a
images what our beta = a + 270 deg images, i.e. pass their solver

    angles_theirs = our_betas + pi/2      (mod 2pi)

and feed y in OUR view order, unchanged. The exchange file carries those angles explicitly.

Noise: our cohort is NOISELESS (the standing protocol); their WLS weights are yi = I*exp(-b)
with I = 5e5 (their config), computed here from the clean y -- the noiseless limit of their
own `add_noise_sino`.

    python scripts/export_cohort_for_jrm.py --ckpt logs/fm3d_databridge/ckpt_iter500000.pth \
        [--n 30] [--out refs/jrm-adm/data/ours_cohort]

(--ckpt only supplies build_world's geometry/dataset plumbing; no prior is evaluated.)
"""
import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_posterior3d import build_world


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="logs/fm3d_databridge/ckpt_iter500000.pth")
    ap.add_argument("--out", default="refs/jrm-adm/data/ours_cohort")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for i in range(args.start, args.n):
        world = build_world(ckpt=args.ckpt, dev="cuda", split="test", run=i,
                            motion_kind="akima", seed=1000 + i,
                            trans_mm=10.0, rot_deg=10.0)
        gen = world["gen"]
        y = world["y"]
        if y.ndim == 4:                                    # (1, V, nv, nu) -> (V, nv, nu)
            y = y[0]
        assert y.ndim == 3 and y.shape[0] > 1, f"unexpected y shape {tuple(y.shape)}"
        V = y.shape[0]
        betas = (torch.arange(V, dtype=torch.float64) * (2 * math.pi / V))
        angles_theirs = torch.remainder(betas + math.pi / 2, 2 * math.pi).float()
        I0 = 5.0e5
        out = {
            "b": y.cpu()[None, None],                      # their (1,1,V,nv,nu) layout
            "yi": (I0 * torch.exp(-y)).cpu()[None, None],
            "angles": angles_theirs.cpu(),
            "theta_true": world["theta_true"].cpu(),
            "gt": world["gt3"].cpu(),                      # (1,1,256,256,256) mu, our frame
            "static_fdk": world["static_fdk"].cpu(),
            "spacing_mm": 1.0,
            "det": {"nu": int(y.shape[2]), "nv": int(y.shape[1]),
                    "du": float(gen.u_coords[1] - gen.u_coords[0]),
                    "dv": float(gen.v_coords[1] - gen.v_coords[0]),
                    "src_dist": float(gen.cfg.SOD),
                    "det_dist": float(gen.cfg.SDD - gen.cfg.SOD)},
            "provenance": "export_cohort_for_jrm 2026-08-17; angles = ours + 90deg "
                          "(diag_jrm_operator_xcheck: rel 0.0016, scale 1.00034)",
        }
        p = os.path.join(args.out, f"p{i:02d}.pt")
        torch.save(out, p)
        print(f"[{i + 1}/{args.n}] {p}  y {tuple(y.shape)} du {out['det']['du']:.3f}",
              flush=True)
    print("done")


if __name__ == "__main__":
    main()
