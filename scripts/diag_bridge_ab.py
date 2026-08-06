"""A/B the TWO geometry bridges, along the WHOLE path -- not just at t=1.

The question (2026-08-05, user): the deployed bridge is

    (A+anchor)  x_t = FDK(y_theta, P(t*theta))  +  t * Delta,   Delta = x_static - FDK(y_theta, P(theta))

and the anchor exists only because A's bare endpoint is not a clean image (FDK's analytic
inverse assumes a CIRCULAR orbit; P(theta) is not one). The alternative

    (B)         x_t = FDK( A(x; P((1-t)*theta)), P_nom )

lands on the static FDK at t=1 BY CONSTRUCTION -- no anchor, no detrend, and it is CHEAPER per
draw (no anchor FDK, no memoized static). It is methodologically cleaner. The only argument for
A was that its path is the family inference walks (fixed measured y, improving geometry), which
B can never reach because attenuating motion in MEASURED data needs the unknown x.

But the anchor BREAKS that argument in the middle: A+anchor is off the true reconstruction
manifold by t*||Delta||, so at t=0.5 it is no more a real reconstruction than B is. Hence this
script, which measures the three paths against each other at t = 0, 1/4, 1/2, 3/4, 1:

    A_bare(t) = FDK(y_theta, P(t*theta))                 the manifold inference actually walks
    A_anch(t) = A_bare(t) + t*Delta                      what we TRAIN on today
    B(t)      = FDK(A(x; P((1-t)*theta)), P_nom)         the clean alternative

THE DECIDING NUMBERS
  d(B, A_anch)  -- how different a prior trained on B would be from today's target
  d(B, A_bare)  vs  d(A_anch, A_bare)  -- WHICH of the two training paths is closer to the
                truth inference sees. If B wins this, it is both cleaner AND more faithful,
                and the "off-manifold" objection to B dies.

All distances are RMS in NET space, reported as % of the net range (2.0) so they are directly
comparable to the "16% of range" endpoint number that started this.

    python scripts/diag_bridge_ab.py --patients 3
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.dataset_cq500 import CQ500Generator
from fm3d.geometry_3d import ConeBeam3DConfig
from fm3d.rigid_motion import params_to_Pmot, random_motion

TS = (0.0, 0.25, 0.5, 0.75, 1.0)


def rms_pct(a, b):
    """RMS(a - b) as a percentage of the NET range (2.0)."""
    return float(torch.sqrt(torch.mean((a - b) ** 2)) / 2.0 * 100.0)


def psnr(a, b):
    mse = float(torch.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(4.0 / mse)      # net peak-to-peak = 2


def montage(path, rows, ts, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = ["A_bare  FDK(y,P(t0))", "A_anch  (+t*Delta)  TRAINED", "B  FDK(A(x;P((1-t)0)),P_nom)",
            "B - A_anch  (x5)"]
    fig, ax = plt.subplots(len(ts), 4, figsize=(13.0, 3.1 * len(ts)))
    for i, t in enumerate(ts):
        ab, aa, b = rows[i]
        z = ab.shape[0] // 2
        for j, (im, vr) in enumerate(((ab, (-1, 1)), (aa, (-1, 1)), (b, (-1, 1)),
                                      (5.0 * (b - aa), (-1, 1)))):
            a = ax[i, j]
            a.imshow(im[z], cmap="gray", vmin=vr[0], vmax=vr[1])
            a.set_xticks([]); a.set_yticks([])
            if i == 0:
                a.set_title(cols[j], fontsize=9)
        ax[i, 0].set_ylabel(f"t = {t:g}", fontsize=10)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/CQ500")
    ap.add_argument("--split", default="val")
    ap.add_argument("--patients", type=int, default=3)
    ap.add_argument("--views", type=int, default=360)
    ap.add_argument("--shape", type=int, nargs=3, default=(256, 256, 256))
    # the TRAINING amplitude, because this is a question about the TRAINING bridge
    ap.add_argument("--motion_amp", default="thies", choices=["fixed", "thies"])
    ap.add_argument("--trans_mm", type=float, default=15.0, help="peak-to-peak [mm]")
    ap.add_argument("--rot_deg", type=float, default=20.0, help="peak-to-peak [deg]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/bridge_ab")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    mgen = torch.Generator().manual_seed(args.seed + 1)

    cfg = ConeBeam3DConfig.thies(n_views=args.views)
    gen = CQ500Generator(args.root, cfg, device=dev, split=args.split,
                         shape=tuple(args.shape), voxel_mm=1.0, sim_native=True)
    print(f"CQ500 '{args.split}': {gen.n_slabs} patients | motion {args.motion_amp} "
          f"{args.trans_mm:g} mm / {args.rot_deg:g} deg peak-to-peak\n")

    acc = {k: [] for k in ("B_Aanch", "B_Abare", "Aanch_Abare", "B_static", "Aanch_static")}
    for n in range(args.patients):
        idx = n
        th = random_motion(cfg.n_views, trans_mm=args.trans_mm, rot_deg=args.rot_deg,
                           amp_mode=args.motion_amp, device=dev, generator=mgen)

        with torch.no_grad():
            # ---- the shared measurement: full motion, simulated on the native grid --------
            y_full = gen.simulate(idx, params_to_Pmot(th, gen.P_nom)[None])
            filt = gen.fdk_filtered(y_full)
            x_static = gen.static_anchor_net(idx)                       # = B(1), by construction
            x1_geo = gen.to_net(gen.fdk(y_full, params_to_Pmot(th, gen.P_nom)[None],
                                        filtered=filt)[0])              # = A_bare(1)
            dlt = x_static - x1_geo
            print(f"patient {idx}:  ||Delta|| = {rms_pct(x_static, x1_geo):5.2f}% of range   "
                  f"(the anchor's size; A_bare's endpoint error)")

            rows = []
            for t in TS:
                a_bare = gen.to_net(gen.fdk(y_full, params_to_Pmot(t * th, gen.P_nom)[None],
                                            filtered=filt)[0])
                a_anch = a_bare + t * dlt
                y_t = gen.simulate(idx, params_to_Pmot((1.0 - t) * th, gen.P_nom)[None])
                b = gen.to_net(gen.fdk(y_t, gen.P_nom[None])[0])
                del y_t

                r = dict(B_Aanch=rms_pct(b, a_anch), B_Abare=rms_pct(b, a_bare),
                         Aanch_Abare=rms_pct(a_anch, a_bare),
                         B_static=rms_pct(b, x_static), Aanch_static=rms_pct(a_anch, x_static))
                for k, v in r.items():
                    acc[k].append(v)
                print(f"  t={t:4.2f}  |B-A_anch| {r['B_Aanch']:5.2f}%  "
                      f"|B-A_bare| {r['B_Abare']:5.2f}%  |A_anch-A_bare| {r['Aanch_Abare']:5.2f}%"
                      f"   (PSNR B vs A_anch {psnr(b, a_anch):5.2f} dB)")
                rows.append((a_bare.cpu().numpy(), a_anch.cpu().numpy(), b.cpu().numpy()))
                del a_bare, a_anch, b
            del y_full, filt
            torch.cuda.empty_cache()

        montage(os.path.join(args.out, f"bridge_ab_p{idx:03d}.png"), rows, TS,
                f"CQ500 patient {idx} | A_bare vs A_anch (deployed) vs B | "
                f"motion {args.trans_mm:g} mm / {args.rot_deg:g} deg p2p")
        np.savez_compressed(os.path.join(args.out, f"bridge_ab_p{idx:03d}.npz"),
                            ts=np.array(TS),
                            a_bare=np.stack([r[0] for r in rows]),
                            a_anch=np.stack([r[1] for r in rows]),
                            b=np.stack([r[2] for r in rows]))
        del rows
        print()

    nt = len(TS)
    print("=" * 84)
    print("MEAN over patients, per t  [RMS as % of net range]")
    print(f"{'t':>6} {'|B-A_anch|':>12} {'|B-A_bare|':>12} {'|A_anch-A_bare|':>16} "
          f"{'|B-static|':>12} {'|A_anch-static|':>15}")
    for i, t in enumerate(TS):
        m = {k: float(np.mean(v[i::nt])) for k, v in acc.items()}
        print(f"{t:6.2f} {m['B_Aanch']:12.2f} {m['B_Abare']:12.2f} {m['Aanch_Abare']:16.2f} "
              f"{m['B_static']:12.2f} {m['Aanch_static']:15.2f}")
    print("=" * 84)
    print("VERDICT KEY:  |B-A_anch| small  -> the two training targets agree; B is free to adopt.")
    print("              |B-A_bare| < |A_anch-A_bare|  -> B is ALSO closer to the manifold")
    print("              inference walks, so the last argument for the anchor is gone.")


if __name__ == "__main__":
    main()
