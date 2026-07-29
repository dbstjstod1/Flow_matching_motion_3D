"""The three-patient cg deliverable figure: OUTPUT vs x_t vs ORACLE FDK vs GT, one row/patient.

Rendered from the SAVED volumes (result.pt now stores x_final and x_t), so no re-run is needed --
only the rigid gauge fit, because a blind reconstruction sits at its own arbitrary SE(3) pose and
slicing it raw beside the GT would compare different anatomical planes.

THE ORACLE COLUMN IS THE CEILING (user, 2026-07-24). The loop's output is FDK(theta_hat), so the
fair ceiling is FDK(theta_TRUE) -- the same operator fed the perfect motion -- NOT the GT volume:
finite views and cone-beam sampling cap the FDK itself below GT, and grading a blind run against
GT alone overstates the remaining gap. The oracle is rebuilt here from each run's saved
theta_true (same seed, same simulated scan), and the interesting reading is OUTPUT vs ORACLE
(how much theta error still costs) and x_t vs ORACLE (whether the prior recovers what the
operator alone cannot).

EVERY PANEL CARRIES ITS OWN PSNR/SSIM. These rows exist to answer one question -- which volume is
the better deliverable? -- and a figure-level title cannot say which panel it is grading. Output
and x_t numbers come straight out of result.pt (`final`, `final_xt`); the oracle's are computed
here, by the same aligned metric.

    python scripts/fig_cg_three.py
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.dataset_cq500 import CQ500Generator
from fm3d.geometry_3d import ConeBeam3DConfig, measured_region_mask
from fm3d.reg_metric import aligned_metrics
from fm3d.rigid_motion import params_to_Pmot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="logs/fm3d_cq500/ckpt_iter500000.pth")
    ap.add_argument("--runs", nargs="+", default=["cg_v0", "cg_v1", "cg_v2"])
    ap.add_argument("--patients", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--dir", default="data")
    ap.add_argument("--split", default="val")
    ap.add_argument("--view", default="axial", choices=["axial", "coronal"])
    ap.add_argument("--crop", type=int, default=104, help="half-width of the zoom box, voxels")
    ap.add_argument("--out", default="data/cg_three_deliverable.png")
    args = ap.parse_args()

    dev = "cuda"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    ca = ck["args"]
    cfg = ConeBeam3DConfig.thies(n_views=ca["views"])
    gen = CQ500Generator(ca["root"], cfg=cfg, device=dev, split=args.split,
                         shape=tuple(ca["shape"]), voxel_mm=1.0)
    spacing = (gen.dz, gen.dy, gen.dx)
    meas = measured_region_mask(gen.shape, spacing, cfg, device=dev)

    rows = []
    for tag, pid in zip(args.runs, args.patients):
        p = os.path.join(args.dir, tag, "result.pt")
        if not os.path.isfile(p):
            print(f"skip {tag}: no {p}")
            continue
        r = torch.load(p, map_location="cpu", weights_only=False)
        if "x_final" not in r or "x_t" not in r:
            print(f"skip {tag}: no saved volumes (run predates the change)")
            continue
        gt = gen.volume(pid)
        gt3 = gt[0, 0]
        # metrics come from the run; only the ALIGNMENT is recomputed, for display
        _, out_al = aligned_metrics(r["x_final"].to(dev).float(), gt3, spacing, mask=meas,
                                    iters=300, return_aligned=True)
        _, xt_al = aligned_metrics(r["x_t"].to(dev).float(), gt3, spacing, mask=meas,
                                   iters=300, return_aligned=True)
        # the CEILING: the run's own scan (saved theta_true = same motion draw), reconstructed
        # by the same FDK with the TRUE motion. Aligned like everything else -- even the oracle
        # carries a small gauge offset from the anchoring of the orbit.
        P_true = params_to_Pmot(r["theta_true"].to(dev), gen.P_nom)
        with torch.no_grad():
            y = gen.simulate(pid, P_true[None])
            oracle = gen.fdk(y, P_true[None])[0]
        m_or, or_al = aligned_metrics(oracle, gt3, spacing, mask=meas, iters=300,
                                      return_aligned=True)
        rows.append((tag, r["final"], r["final_xt"], m_or, out_al, xt_al, or_al, gt3))
        print(f"{tag}: oracle FDK {m_or['psnr_aligned']:.2f} dB / {m_or['ssim_aligned']:.3f}",
              flush=True)
    if not rows:
        raise SystemExit("nothing to draw")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D, H, W = rows[0][5].shape
    c = args.crop
    zc, yc, xc = D // 2, H // 2, W // 2
    ys, xs = slice(yc - c, yc + c), slice(xc - c, xc + c)

    def cut(v):
        return (v[zc, ys, xs] if args.view == "axial" else v[:, yc, xs]).cpu()

    n = len(rows)
    fig, ax = plt.subplots(n, 4, figsize=(14.8, 3.9 * n), squeeze=False)
    for i, (tag, fm, fx, m_or, out_al, xt_al, or_al, gt3) in enumerate(rows):
        panels = [
            (f"FDK(theta_hat) = current OUTPUT\n"
             f"{fm['psnr_aligned']:.2f} dB / SSIM {fm['ssim_aligned']:.3f}", out_al),
            (f"carried x_t\n{fx['psnr_aligned']:.2f} dB / SSIM {fx['ssim_aligned']:.3f}", xt_al),
            (f"oracle FDK(theta_TRUE) = CEILING\n"
             f"{m_or['psnr_aligned']:.2f} dB / SSIM {m_or['ssim_aligned']:.3f}", or_al),
            ("ground truth", gt3)]
        for j, (name, v) in enumerate(panels):
            ax[i, j].imshow(cut(v), cmap="gray", vmin=0.0, vmax=0.05,
                            aspect="auto" if args.view == "coronal" else "equal")
            ax[i, j].set_title(name, fontsize=9)
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        ax[i, 0].set_ylabel(tag, fontsize=10)
    fig.suptitle(f"cg data step, 3 CQ500 val patients ({args.view} zoom) -- "
                 f"dc_op=cg kappa=0 N=50 PER=50 l2si, seeds 3/7/11", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
