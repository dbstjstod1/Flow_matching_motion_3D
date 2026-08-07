"""Score runs under the HU convention, which is the one comparable to Thies' 0.83 -> 0.94.

WHY THIS EXISTS. Every SSIM this project has printed is computed on raw attenuation (mu, air = 0)
with `data_range` = the volume's own peak. Thies reports 0.83 uncorrected -> 0.94 compensated, and
under OUR convention the same uncorrected volume scores 0.60 -- an offset that is present BEFORE
any method runs, so "0.763 vs 0.94" was never a like-for-like comparison. Neither the evaluation
region (0.602 masked vs 0.593 tight vs 0.711 unmasked), nor the SSIM variant (2D per-slice 0.595,
data_range = max-min 0.635, window 11 0.722), nor the motion amplitude (the paper says "maximal
amplitude ... for translation parameterS", i.e. per-DoF, which is what we simulate) explains it.

WHAT DOES. Converting to Hounsfield units first: HU = 1000*(mu - mu_water)/mu_water, clipped to a
fixed window, with `data_range` = the WINDOW WIDTH. That reproduces Thies' uncorrected 0.83 almost
exactly (we get 0.808 on val 0). The mechanism is not the data_range but the OFFSET: SSIM's
luminance term (2*mu_x*mu_y + C1)/(mu_x^2 + mu_y^2 + C1) is NOT shift-invariant, and moving air
from 0 to -1000 makes the local means large, so that term saturates near 1 nearly everywhere.

mu_water is INFERRED from the data (the mode of the soft-tissue histogram) rather than assumed,
because the reconstructions are in absolute mu [1/mm] (the FDK self-normalizes).

    python scripts/metrics_hu.py data/runs/akima55/c2f_v*/result.pt
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.dataset_cq500 import CQ500Generator
from fm3d.geometry_3d import ConeBeam3DConfig, measured_region_mask
from fm3d.reg_metric import rigid_align, ssim
from fm3d.rigid_motion import make_motion, params_to_Pmot

LO, HI = -1000.0, 3000.0            # the window; data_range is its width


def main():
    paths = [p for p in sys.argv[1:] if os.path.isfile(p)]
    if not paths:
        raise SystemExit(__doc__)
    dev = "cuda"
    ck = torch.load("logs/fm3d_cq500_leap/ckpt_iter500000.pth", map_location=dev, weights_only=False)
    ca = ck["args"]
    cfg = ConeBeam3DConfig.thies(n_views=ca["views"])
    gen = CQ500Generator(ca["root"], cfg=cfg, device=dev, split="val",
                         shape=tuple(ca["shape"]), voxel_mm=1.0)
    sp = (gen.dz, gen.dy, gen.dx)
    meas = measured_region_mask(gen.shape, sp, cfg, device=dev)

    print(f"HU window [{LO:.0f},{HI:.0f}], SSIM vs the static FDK, rigidly aligned first")
    print(f"{'run':22} {'uncorrected':>12} {'OUT FDK(th)':>12} {'x_t':>12}")
    tot = [0.0, 0.0, 0.0]
    for p in sorted(paths):
        r = torch.load(p, map_location=dev, weights_only=False)
        # result.pt does not store the args; the deferred-metric snapshot dir does.
        meta = os.path.join(os.path.dirname(p), "snaps", "meta.pt")
        if not os.path.isfile(meta):
            raise SystemExit(f"{meta} missing -- cannot know which patient/seed {p} used")
        a = torch.load(meta, map_location="cpu", weights_only=False)["args"]
        run, seed = a["run"], a["seed"]
        gt = gen.volume(run)
        th = make_motion("akima", cfg.n_views, device=dev, seed=seed, trans_mm=10.0, rot_deg=10.0)
        with torch.no_grad():
            y = gen.project(gt, params_to_Pmot(th, gen.P_nom)[None])[0]
            sfdk = gen.fdk(gen.project(gt, gen.P_nom[None]), gen.P_nom[None])[0]
            cold = gen.fdk(y[None], gen.P_nom[None])[0]
        # soft-tissue mu = the mode of the in-head histogram
        h = torch.histc(gt[0, 0][meas & (gt[0, 0] > 0.005)], bins=200, min=0.005, max=0.03)
        mu_w = 0.005 + (float(h.argmax()) + 0.5) * (0.03 - 0.005) / 200

        def hu(v):
            return (1000.0 * (v - mu_w) / mu_w).clamp(LO, HI)

        ref = hu(sfdk)
        row = []
        for v in (cold, r["x_final"].to(dev).float(), r["x_t"].to(dev).float()):
            al, _ = rigid_align(v, sfdk, sp, mask=meas, iters=300)
            row.append(ssim(hu(al), ref, data_range=HI - LO, mask=meas))
        tot = [a_ + b_ for a_, b_ in zip(tot, row)]
        print(f"{os.path.basename(os.path.dirname(p))[:22]:22} "
              f"{row[0]:12.4f} {row[1]:12.4f} {row[2]:12.4f}", flush=True)
    n = len(paths)
    print(f"{'MEAN':22} {tot[0]/n:12.4f} {tot[1]/n:12.4f} {tot[2]/n:12.4f}")
    print("\nThies (TMI 2025, 30 patients): 0.83 uncorrected -> 0.94 compensated.")


if __name__ == "__main__":
    main()
