"""Per-patient visual comparison montage: JRM-ADM port vs our loop (A2d, 2026-08-18).

For every test30 patient whose JRM-ADM result exists, renders one PNG:

    rows    = axial z=128 / coronal y=128 (GT-volume indices; recons shown in their own
              gauge pose -- offsets are ~1 mm/2 deg, visually negligible, and each panel's
              title carries the RIGID-ALIGNED PSNR/SSIM vs GT, the honest number)
    columns = motion-corrupted FDK (cold start) | FDK(theta_hat_JRM) | FDK(theta_hat_ours)
              | JRM-ADM x_est (retrained prior, our protocol port) | our x_t
              | static FDK (scanner ceiling) | GT

The two FDK(theta_hat) panels are the FDK-CLASS comparison (user request 2026-08-18): the
same vanilla motion-compensated FDK fed each method's estimated motion, so the panels isolate
MOTION-ESTIMATE quality from prior/regularization style. JRM's theta comes through
jrm_theta_convert (gated end-to-end by gate_jrm_theta_convert: rel ~5e-3); ours is the
deliverable K-step average (result["theta"]).

Sources: refs/jrm-adm/data/ours_cohort/pNN.pt (gt/static), refs/jrm-adm/data/recon_ours_v2/
pNN_result.pt (their arm; mu rescaled 0.02/0.0193, 224^3 padded to 256^3 at center),
data/fm3d_test30/pNN/result.pt (our x_t), cold FDK recomputed via build_world (same triple).

    python scripts/fig_jrm_cmp.py [--out data/jrm_cmp] [--only 0 1 2]
"""
import argparse
import glob
import os
import re
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fm3d.reg_metric import aligned_metrics
from fm3d.rigid_motion import params_to_Pmot
from jrm_theta_convert import jrm_thetas_to_ours
from run_posterior3d import build_world

JRM = "refs/jrm-adm/data/recon_ours_v2"
EXCH = "refs/jrm-adm/data/ours_cohort"
OURS = "data/fm3d_test30"
MU_RATIO = 0.02 / 0.0193          # their mu constant -> ours (exact scale, both zero at air)


def pad_center(v, shape=(256, 256, 256)):
    out = torch.zeros(shape, dtype=v.dtype, device=v.device)
    o = [(a - b) // 2 for a, b in zip(shape, v.shape)]
    out[o[0]:o[0] + v.shape[0], o[1]:o[1] + v.shape[1], o[2]:o[2] + v.shape[2]] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/jrm_cmp")
    ap.add_argument("--only", type=int, nargs="*", default=None)
    ap.add_argument("--ckpt", default="logs/fm3d_databridge/ckpt_iter500000.pth")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    done = sorted(int(re.search(r"p(\d+)_result", p).group(1))
                  for p in glob.glob(os.path.join(JRM, "p*_result.pt")))
    if args.only is not None:
        done = [i for i in done if i in args.only]
    print(f"patients with JRM results: {done}")

    for i in done:
        tag = f"p{i:02d}"
        exch = torch.load(os.path.join(EXCH, f"{tag}.pt"), map_location="cuda",
                          weights_only=False)
        jrm = torch.load(os.path.join(JRM, f"{tag}_result.pt"), map_location="cuda",
                         weights_only=False)
        ours = torch.load(os.path.join(OURS, tag, "result.pt"), map_location="cuda",
                          weights_only=False)
        # pairing assertion, same discipline as cmp_thies_vs_ours
        dev = float((jrm["theta_true"].cuda() - ours["theta_true"].cuda()).abs().max())
        assert dev < 1e-5, f"{tag}: JRM and ours saw different motion (|dtheta| {dev:.2e})"

        world = build_world(ckpt=args.ckpt, dev="cuda", split="test", run=i,
                            motion_kind="akima", seed=1000 + i, trans_mm=10.0, rot_deg=10.0)
        gen = world["gen"]
        with torch.no_grad():
            y = world["y"]
            cold = gen.fdk(y, gen.P_nom[None])[0]
            if cold.ndim > 3:
                cold = cold.reshape(256, 256, 256)
            # FDK-class panels: the SAME vanilla motion-compensated FDK under each method's
            # estimated theta (reporting policy: vanilla weights; Voronoi is loop-internal).
            th_jrm = jrm_thetas_to_ours(jrm["thetas_est"].float().cuda())
            fdk_jrm = gen.fdk(y, params_to_Pmot(th_jrm, gen.P_nom)[None])[0]
            th_ours = ours["theta"].float().cuda()
            fdk_ours = gen.fdk(y, params_to_Pmot(th_ours, gen.P_nom)[None])[0]
            fdk_jrm = fdk_jrm.reshape(256, 256, 256) if fdk_jrm.ndim > 3 else fdk_jrm
            fdk_ours = fdk_ours.reshape(256, 256, 256) if fdk_ours.ndim > 3 else fdk_ours

        gt = exch["gt"].float().cuda()
        static = exch["static_fdk"].float().cuda()
        xt = ours["x_t"].float().cuda()
        xe = pad_center(jrm["x_est"][0, 0].float().cuda() * MU_RATIO)

        panels = [("motion FDK (cold)", cold),
                  ("FDK(theta JRM)", fdk_jrm), ("FDK(theta ours)", fdk_ours),
                  ("JRM-ADM (ours-retrained)", xe),
                  ("our x_t", xt), ("static FDK", static), ("GT", gt)]
        titles = []
        for name, v in panels:
            if name in ("GT",):
                titles.append(name)
                continue
            m = aligned_metrics(v, gt, (1., 1., 1.), mask=None, iters=200)
            titles.append(f"{name}\n{m['psnr_aligned']:.2f} dB / {m['ssim_aligned']:.3f}")

        fig, ax = plt.subplots(2, len(panels), figsize=(3.2 * len(panels), 7))
        for c, ((name, v), ttl) in enumerate(zip(panels, titles)):
            vn = v.cpu().numpy()
            ax[0, c].imshow(np.clip(vn[128], 0, 0.06), cmap="gray")
            ax[1, c].imshow(np.clip(vn[:, 128], 0, 0.06), cmap="gray", origin="lower")
            ax[0, c].set_title(ttl, fontsize=9)
            for r in (0, 1):
                ax[r, c].axis("off")
        fig.suptitle(f"{tag}  |  test30 (run={i}, seed={1000 + i})  |  akima 10/10 p2p  |  "
                     f"aligned metrics vs GT  |  JRM runtime {jrm['runtime_sec'] / 60:.0f} min",
                     fontsize=11)
        plt.tight_layout()
        p = os.path.join(args.out, f"{tag}_cmp.png")
        plt.savefig(p, dpi=110)
        plt.close(fig)
        print(f"{tag} -> {p}", flush=True)


if __name__ == "__main__":
    main()
