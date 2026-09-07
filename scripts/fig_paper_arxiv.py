"""arXiv manuscript result figures (the SPIE figs 1(a/b) and 3 are reused from figs/spie/).

    python scripts/fig_paper_arxiv.py                       # both figures, default cases
    python scripts/fig_paper_arxiv.py --ours p00,p16 --bench p14,p16

Writes to figs/arxiv/:
    fig_ours_qual.{png,pdf}  -- OUR method per case: uncorrected | FDK(theta_hat) | x_t
                                | FDK(theta_true) | ground truth   (2 rows/case: ax+co)
    fig4_bench.{png,pdf}     -- cross-method: uncorrected | learned autofocus | JRM-ADM (native)
                                | proposed | ground truth (pixel-linear bridge only in Fig. 6) (2 rows/case: ax+co)

Every volume is rigidly aligned to the GT before slicing (the SE(3) gauge moves anatomy by
several voxels; see fm3d/reg_metric.py). Metric labels use each run's STORED cohort numbers
where they exist (`final` / `final_xt` -- the tables quote them; a fresh gauge fit lands on a
slightly different local optimum); panels no run stores (uncorrected, FDK(theta_true)) are
scored here with the loop's own convention (mask=measured region, iters=200).

The cohort dirs are the standing test30 runs; pairing across dirs is guaranteed by the
(split=test, run=i, seed=1000+i) convention that every cmp script asserts.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.reg_metric import aligned_metrics          # noqa: E402
from fm3d.rigid_motion import params_to_Pmot         # noqa: E402

MU_WATER = 0.02
CKPT = "logs/fm3d_databridge/ckpt_iter500000.pth"

BENCH_ARMS = [  # (panel title, run dir or special key); method NAMES, not authors (user, 09-03)
    ("Uncorrected FDK", "x0"),
    ("Learned autofocus", "data/bench_thies_test30"),
    ("JRM-ADM", "jrm"),                  # native cohort, refs/jrm-adm/data/recon_ours_v2
    ("Proposed", "data/fm3d_test30_databridge"),
    ("Ground truth", "gt"),
]
OURS_ARMS = [
    ("Uncorrected FDK", "x0"),
    ("Proposed, FDK($\\hat{\\theta}$)", "output"),
    ("Proposed, $x_t$ (final iterate)", "x_t"),
    ("FDK at true motion", "ceil"),
    ("Ground truth", "gt"),
]


def win(mu, level=40.0, width=420.0):
    hu = (mu - MU_WATER) / MU_WATER * 1000.0
    lo, hi = level - width / 2, level + width / 2
    return np.clip((hu - lo) / (hi - lo), 0, 1)


def load_world(tag, dev):
    from run_posterior3d import build_world
    idx = int(tag[1:])
    w = build_world(ckpt=CKPT, dev=dev, split="test", run=idx, seed=1000 + idx,
                    motion_kind="akima", trans_mm=10.0, rot_deg=10.0)
    with torch.no_grad():
        w["x0"] = w["gen"].fdk(w["y"], w["gen"].P_nom[None])[0]
        w["ceil"] = w["gen"].fdk(
            w["y"], params_to_Pmot(w["theta_true"], w["gen"].P_nom)[None])[0]
    return w


def score_and_align(vol, w, stored=None):
    """Align to GT; return (aligned ndarray, 'PP.PP dB / 0.SSS' label)."""
    m, al = aligned_metrics(vol.float(), w["gt3"], w["spacing"], mask=w["meas"],
                            iters=200, return_aligned=True)
    if stored is not None:
        m = stored
    return al.detach().float().cpu().numpy(), \
        f"{m['psnr_aligned']:.2f} dB / {m['ssim_aligned']:.3f}"


def build_case(tag, w, arms, dev):
    """One patient's panels for one figure: [(title, volume ndarray, label|None)]."""
    r_fm = torch.load(f"data/fm3d_test30_databridge/{tag}/result.pt",
                      map_location=dev, weights_only=False)
    out = []
    for title, src in arms:
        if src == "gt":
            out.append((title, w["gt3"].float().cpu().numpy(), None))
            continue
        stored = None
        if src == "x0":
            vol = w["x0"]
        elif src == "ceil":
            vol = w["ceil"]
        elif src == "output":
            vol, stored = r_fm["x_final"].to(dev), r_fm["final"]
        elif src == "x_t":
            vol, stored = r_fm["x_t"].to(dev), r_fm["final_xt"]
        elif src == "jrm":                      # native JRM-ADM: their grid/mu -> our frame
            j = torch.load(f"refs/jrm-adm/data/recon_ours_v2/{tag}_result.pt",
                           map_location=dev, weights_only=False)
            assert (j["theta_true"].to(dev) - w["theta_true"]).abs().max() < 1e-5, tag
            v = j["x_est"][0, 0].float().to(dev) * (0.02 / 0.0193)
            vol = torch.zeros((256, 256, 256), dtype=v.dtype, device=dev)
            o = [(256 - n) // 2 for n in v.shape]
            vol[o[0]:o[0] + v.shape[0], o[1]:o[1] + v.shape[1], o[2]:o[2] + v.shape[2]] = v
        else:                                   # a cohort dir (bench figure)
            r = torch.load(os.path.join(src, tag, "result.pt"),
                           map_location=dev, weights_only=False)
            vol = (r["out_vol"] if "out_vol" in r else r["x_t"]).to(dev)
            stored = r.get("final_xt")          # None for the Thies dir -> fresh score
        out.append((title, *score_and_align(vol, w, stored)))
    return out


def draw(cases, path, panel_w=1.55):
    """cases: list of per-case panel lists (same length); 2 rows (ax, co) per case.

    Slices are shown UNCROPPED at the native 256x256 aspect (user's call, 2026-08-20)."""
    n = len(cases[0])
    hco = 1.0
    ratios = []
    for c in range(len(cases)):
        if c:
            ratios.append(0.16)                 # spacer row carrying case-2 labels
        ratios += [1.0, hco]
    fig = plt.figure(figsize=(panel_w * n, panel_w * (sum(ratios) + 0.30)))
    gs = fig.add_gridspec(len(ratios), n, height_ratios=ratios,
                          wspace=0.02, hspace=0.03,
                          left=0.03, right=0.995, top=0.93, bottom=0.005)
    for c, panels in enumerate(cases):
        r0 = c * 3                              # rows: [ax, co, spacer] per case
        for j, (title, v, lab) in enumerate(panels):
            zc, yc = v.shape[0] // 2, v.shape[1] // 2
            axa = fig.add_subplot(gs[r0, j])
            axc = fig.add_subplot(gs[r0 + 1, j])
            axa.imshow(win(v[zc]), cmap="gray", vmin=0, vmax=1)
            axc.imshow(win(v[:, yc])[::-1], cmap="gray", vmin=0, vmax=1)
            if c == 0:
                axa.set_title(title if lab is None else f"{title}\n{lab}",
                              fontsize=6.4, pad=2.5)
            elif lab is not None:
                axa.set_title(lab, fontsize=6.2, pad=1.5)
            for ax in (axa, axc):
                ax.set_xticks([]), ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
            if j == 0:
                axa.set_ylabel(f"case {c + 1}\naxial", fontsize=6.6)
                axc.set_ylabel("coronal", fontsize=6.6)
    for ext in ("png", "pdf"):
        fig.savefig(path + "." + ext, dpi=400, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print("wrote", path + ".{png,pdf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="p00,p16", help="cases for the ours-only figure")
    ap.add_argument("--bench", default="p14,p16", help="cases for the cross-method figure")
    ap.add_argument("--out", default="figs/arxiv")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tags = sorted(set(a.ours.split(",")) | set(a.bench.split(",")))
    worlds = {t: load_world(t, dev) for t in tags}

    for name, arms, taglist in (("fig_ours_qual", OURS_ARMS, a.ours.split(",")),
                                ("fig4_bench", BENCH_ARMS, a.bench.split(","))):
        cases = [build_case(t, worlds[t], arms, dev) for t in taglist]
        print(f"[{name}] cases {taglist}")
        for t, panels in zip(taglist, cases):
            for title, _, lab in panels:
                if lab:
                    print(f"  {t} {title:28s} {lab}")
        draw(cases, os.path.join(a.out, name))


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
