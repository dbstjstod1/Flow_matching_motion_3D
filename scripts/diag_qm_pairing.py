"""IS THE QUALITY NET TRAINED ON MOTION-CORRUPTED VOLUMES, OR ON CONSISTENT (ORACLE) ONES?

    python scripts/diag_qm_pairing.py [--patients 3] [--qm logs/bench_thies_qm/qmnet_iter007000.pth]

`bench/thies/data.py` L167-168 simulates with P_mot and backprojects with the SAME P_mot. A
consistent pair reconstructs the object, so the training input would carry no motion mismatch at
all -- only the non-uniform angular sampling a scrambled orbit leaves behind. The paper does the
opposite (IV, "The filtered projection data is reconstructed from these perturbed matrices"):
CLEAN data, PERTURBED matrices.

FOUR VOLUMES, ONE MOTION DRAW, ONE PATIENT -- all reconstructed on the same 128^3 @ 2 mm grid and
all scored against the same reference:

    static     recon(y(P_nom),  P_nom)    the reference. Thies' I_ref.
    consistent recon(y(P_mot),  P_mot)    WHAT WE TRAIN ON TODAY
    x0         recon(y(P_mot),  P_nom)    the optimizer's STARTING state (theta = 0)
    paper      recon(y(P_nom),  P_mot)    the paper's literal construction

VIF* = 1 - VIF_scalar, so LOWER IS BETTER and 0 is perfect (`vif.py:203`); the optimizer
MINIMIZES it. Read the table with that in mind -- a HIGHER VIF* means a WORSE volume.

If `consistent` scores far better than `x0`, the training distribution sits at the converged end
of the optimization and never covers where the optimization starts. The `--qm` checkpoint, if
given, is then asked to predict all four so the extrapolation gap is measured and not inferred.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.thies.data import QMSampleSource                       # noqa: E402
from bench.thies.recon import to_unit                          # noqa: E402
from bench.thies.vif import vif_star_map_3d                    # noqa: E402
from fm3d.geometry_3d import ConeBeam3DConfig                  # noqa: E402
from fm3d.rigid_motion import akima_motion, params_to_Pmot     # noqa: E402
from fm3d.reg_metric import aligned_metrics                       # noqa: E402

DEFAULT_ROOT = "/home/mirlab/Desktop/CQ500"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--patients", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=11)
    ap.add_argument("--qm", default="logs/bench_thies_qm/qmnet_iter007000.pth")
    ap.add_argument("--amp", default="ours", choices=["ours", "thies"])
    a = ap.parse_args()

    from bench.thies.data import TRAIN_AMP, THIES_TRAIN_AMP
    cfg = ConeBeam3DConfig.thies()
    root = a.root or os.environ.get("CQ500_ROOT") or DEFAULT_ROOT
    src = QMSampleSource(root, cfg, split=a.split, device="cuda",
                      amp=dict(TRAIN_AMP if a.amp == "ours" else THIES_TRAIN_AMP))

    net = None
    if a.qm and os.path.exists(a.qm):
        from bench.thies.qmnet import QualityMetricUNet3D, THIES_F_MAPS
        ck = torch.load(a.qm, map_location="cuda", weights_only=False)
        ca = ck.get("args", {})
        net = QualityMetricUNet3D(tuple(ca.get("f_maps", THIES_F_MAPS)),
                                  norm=ca.get("norm", "none")).cuda()
        net.load_state_dict(ck["model"])
        net.eval()
        print(f"quality net: {a.qm} (iter {ck['iter']}, f_maps {net.f_maps}, "
              f"norm {ca.get('norm', 'none')})")

    P_nom = src.gen.P_nom
    sp = (src.grid.spacing[0],) * 3
    rows = []
    for k in range(a.patients):
        seed = a.seed0 + k
        theta = akima_motion(cfg.n_views, n_nodes=src.n_nodes, device="cuda",
                             seed=seed, zero_centre=True, **src.amp)
        P_mot = params_to_Pmot(theta, P_nom)
        amp_t = float(theta[:, :3].max() - theta[:, :3].min())
        amp_r = float(torch.rad2deg(theta[:, 3:].max() - theta[:, 3:].min()))
        with torch.no_grad():
            y_mot = src.gen.simulate(k, P_mot[None])
            consistent = src.recon(y_mot, P_mot, src.grid)          # <- data.py L167-168
            x0 = src.recon(y_mot, P_nom, src.grid)                  # <- theta = 0
            del y_mot
            y_sta = src.gen.simulate(k, P_nom[None])
            static = src.recon(y_sta, P_nom, src.grid)              # <- the reference
            paper = src.recon(y_sta, P_mot, src.grid)               # <- the paper's construction
            del y_sta

            ref = to_unit(static)[None, None]
            print(f"\npatient {k} seed {seed}  motion p2p {amp_t:.1f} mm / {amp_r:.1f} deg")
            print(f"  {'volume':<12s} {'PSNR':>7s} {'SSIM':>7s} {'VIF* (0=perfect)':>18s}"
                  f" {'net pred':>10s}")
            for name, v in (("consistent", consistent), ("x0", x0), ("paper", paper)):
                # aligned_metrics fits the rigid pose by gradient descent, so it must run with
                # grad ENABLED even though everything it is handed is detached.
                with torch.enable_grad():
                    m = aligned_metrics(v, static, sp)
                d = to_unit(v)[None, None]
                vs = float(vif_star_map_3d(d, ref).mean())
                pred = float(net(d).mean()) if net is not None else float("nan")
                print(f"  {name:<12s} {m['psnr_aligned']:7.2f} {m['ssim_aligned']:7.4f} "
                      f"{vs:18.4f} {pred:10.4f}")
                rows.append((name, vs, pred))
        del consistent, x0, paper, static
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    for name in ("consistent", "x0", "paper"):
        v = [r[1] for r in rows if r[0] == name]
        p = [r[2] for r in rows if r[0] == name]
        print(f"  {name:<12s} true VIF* {sum(v)/len(v):.4f}   net says {sum(p)/len(p):.4f}   "
              f"error {sum(p)/len(p) - sum(v)/len(v):+.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
