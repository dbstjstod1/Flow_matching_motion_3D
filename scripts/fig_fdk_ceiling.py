"""THE FIGURE: what "ceiling" means, and what the FDK ceiling actually looks like.

THE POINT (user, 2026-07-27). The static FDK is NOT the ceiling for a motion-corrected FDK. It is
the reconstruction of a DIFFERENT, motion-free scan, and no method operating on the motion-
corrupted data can reach it with an FDK. The real ceiling for an FDK deliverable is
FDK(y_motion, P(theta_true)) -- perfect motion knowledge, same data -- and THAT VOLUME STILL HAS
STREAKS, because FDK is an analytic inverse derived for a circular, equiangular orbit and per-view
rigid motion moves the source off that orbit relative to the object.

Measured on val 0 (aligned, vs GT), all with theta_true where applicable:

                    FDK              CG (60 it)
    static data   33.34/0.8256      38.07/0.9697
    motion data   32.01/0.7294      41.04/0.9844      <- motion COSTS FDK 0.096 SSIM and GAINS CG 0.015

so the damage is the operator's, not the data's -- and a known motion is actually INFORMATIVE
(it perturbs the source trajectory off the circle, filling part of the frequency cone a circular
orbit provably misses).

WHY THE DISPLAY WINDOW MATTERS. Streaks are a low-contrast, soft-tissue-domain artefact. The
project's standard montage uses a window wide enough to show bone, which buries them. This figure
uses a BRAIN window and adds an absolute-difference row on a shared scale, which is where the
streaks are unmistakable.

    python scripts/fig_fdk_ceiling.py --run 0 --seed 3
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fm3d.dataset_cq500 import CQ500Generator                            # noqa: E402
from fm3d.geometry_3d import ConeBeam3DConfig, measured_region_mask      # noqa: E402
from fm3d.reg_metric import aligned_metrics                              # noqa: E402
from fm3d.rigid_motion import make_motion, params_to_Pmot                # noqa: E402
from run_posterior3d import cg_dc_step                                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="logs/fm3d_cq500/ckpt_iter500000.pth")
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--cg_iters", type=int, default=60)
    ap.add_argument("--win", default="-20,100", help="display window in HU (brain)")
    ap.add_argument("--out", default="data/figures/fdk_ceiling.png")
    args = ap.parse_args()
    lo, hi = [float(v) for v in args.win.split(",")]

    dev = "cuda"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    ca = ck["args"]
    cfg = ConeBeam3DConfig.thies(n_views=ca["views"])
    gen = CQ500Generator(ca["root"], cfg=cfg, device=dev, split="val",
                         shape=tuple(ca["shape"]), voxel_mm=1.0)
    sp = (gen.dz, gen.dy, gen.dx)
    meas = measured_region_mask(gen.shape, sp, cfg, device=dev)
    gt = gen.volume(args.run)
    gt3 = gt[0, 0]
    th = make_motion("akima", cfg.n_views, device=dev, seed=args.seed, trans_mm=10.0, rot_deg=10.0)

    with torch.no_grad():
        y_mot = gen.simulate(args.run, params_to_Pmot(th, gen.P_nom)[None])[0]
        y_sta = gen.simulate(args.run, gen.P_nom[None])[0]
        cold = gen.fdk(y_mot[None], gen.P_nom[None])[0]
        orac = gen.fdk(y_mot[None], params_to_Pmot(th, gen.P_nom)[None])[0]
        sfdk = gen.fdk(y_sta[None], gen.P_nom[None])[0]
        cg = cg_dc_step(orac.clone(), th, y_mot, gen, iters=args.cg_iters, lam=0.0)

    # mu -> HU, water inferred from the data
    h = torch.histc(gt3[meas & (gt3 > 0.005)], bins=200, min=0.005, max=0.03)
    mu_w = 0.005 + (float(h.argmax()) + 0.5) * (0.03 - 0.005) / 200

    def hu(v):
        return 1000.0 * (v - mu_w) / mu_w

    panels = [
        ("uncorrected FDK\n(motion data, nominal P)", cold),
        ("FDK(theta_TRUE)\n= THE FDK CEILING", orac),
        ("CG(theta_TRUE), 60 it\n= the operator ceiling", cg),
        ("static FDK\n(motion-free scan -- NOT reachable)", sfdk),
        ("ground truth", gt3),
    ]
    # every panel aligned into the GT frame, so the same anatomy is under the same pixel
    scored = []
    for name, v in panels:
        if v is gt3:
            scored.append((name, v, None))
            continue
        m, al = aligned_metrics(v, gt3, sp, mask=meas, iters=300, return_aligned=True)
        scored.append((name, al, m))

    D, H, W = gt3.shape
    za, zc = D // 2, H // 2
    fig, ax = plt.subplots(3, len(scored), figsize=(3.05 * len(scored), 9.4))
    dmax = 120.0                                            # HU, shared difference scale
    for j, (name, v, m) in enumerate(scored):
        vh = hu(v)
        gh = hu(gt3)
        for i, (sl, lab) in enumerate(((vh[za], "axial"), (vh[:, zc], "coronal"))):
            ax[i, j].imshow(sl.cpu(), cmap="gray", vmin=lo, vmax=hi)
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if j == 0:
                ax[i, j].set_ylabel(f"{lab}\nbrain window [{lo:.0f},{hi:.0f}] HU", fontsize=8)
        d = (vh[za] - gh[za]).abs()
        im = ax[2, j].imshow(d.cpu(), cmap="inferno", vmin=0, vmax=dmax)
        ax[2, j].set_xticks([]); ax[2, j].set_yticks([])
        if j == 0:
            ax[2, j].set_ylabel(f"|difference| vs GT\n0 - {dmax:.0f} HU", fontsize=8)
        t = name if m is None else (f"{name}\n{m['psnr_aligned']:.2f} dB / "
                                    f"SSIM {m['ssim_aligned']:.4f}")
        ax[0, j].set_title(t, fontsize=8.5)
    fig.colorbar(im, ax=ax[2, :].tolist(), fraction=0.015, pad=0.01, label="|error| [HU]")
    fig.suptitle(
        f"The FDK ceiling has streaks -- val {args.run}, akima 5 mm / 5 deg.  Every panel uses "
        f"the TRUE motion where motion is corrected;\nthe only difference between panels 2 and 3 "
        f"is the RECONSTRUCTION OPERATOR, on identical data and identical geometry.", fontsize=10)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"-> {args.out}")
    for name, _, m in scored:
        if m:
            print(f"{name.splitlines()[0]:34} {m['psnr_aligned']:6.2f} dB / {m['ssim_aligned']:.4f}")


if __name__ == "__main__":
    main()
