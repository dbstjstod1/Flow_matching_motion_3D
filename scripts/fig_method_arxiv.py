"""arXiv manuscript Fig. 2 (method): the geometry bridge (a) and the inference loop as an
algorithm flowchart (b). Pure plotting, no GPU: panel (a) reads the stored bridge strip
`data/fig_assets/bridgeB_p00.npz` (five FDK reconstructions of patient p00 at
t = 0, 0.25, 0.5, 0.75, 1, produced 2026-08-05 by the SPIE figure pipeline).

    python scripts/fig_method_arxiv.py            # -> docs/arxiv/fig1_method.{pdf,png}
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

TW = 6.5                       # letter, 1-in margins
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
F1, F2, F3 = "#eef4fc", "#fdf1ec", "#eaf7f2"
INK, INK2, MUTED, LINE = "#0b0b0b", "#52514e", "#8a8a85", "#bdbdb8"
MU_WATER = 0.02
FS_TITLE, FS_LAB, FS_BODY = 8.2, 6.5, 6.0   # panel titles / panel labels / all other text

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.5,
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
})


def to_hu(v):
    return (v - MU_WATER) / MU_WATER * 1000.0


def win(img, level=40.0, width=400.0):
    lo, hi = level - width / 2, level + width / 2
    return np.clip((img - lo) / (hi - lo), 0, 1)


# ---------------------------------------------------------------- drawing helpers
def rbox(ax, x, y, w, h, fc, ec, lw=0.9, ls="-", r=0.015, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z, clip_on=False))


def arrow(ax, p0, p1, color=INK2, lw=0.9, rad=0.0, ls="-", z=4, shrink=1.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7,
                                 color=color, lw=lw, linestyle=ls, zorder=z, clip_on=False,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=shrink, shrinkB=shrink))


def step_box(ax, x, y, w, h, fc, ec, num, title, lines, tfs=FS_LAB, lfs=FS_BODY):
    """A loop block: coloured header band with the step number and name, body lines below."""
    rbox(ax, x, y, w, h, "white", ec, lw=0.9)
    hh = 0.13
    ax.add_patch(FancyBboxPatch((x, y + h - hh), w, hh,
                                boxstyle="round,pad=0,rounding_size=0.015",
                                fc=fc, ec="none", zorder=3, clip_on=False))
    # square off the header's bottom corners so it reads as a band, not a pill
    ax.add_patch(plt.Rectangle((x, y + h - hh), w, hh / 2, fc=fc, ec="none", zorder=3,
                               clip_on=False))
    ax.text(x + 0.012, y + h - hh / 2, num, ha="left", va="center", fontsize=tfs,
            color=ec, weight="bold", zorder=5)
    ax.text(x + w / 2 + 0.012, y + h - hh / 2, title, ha="center", va="center",
            fontsize=tfs, color=INK, weight="bold", zorder=5)
    ax.text(x + w / 2, y + h - hh - 0.04, "\n".join(lines), ha="center", va="top",
            fontsize=lfs, color=INK, zorder=5, linespacing=1.6)


# ---------------------------------------------------------------- figure
def main(npz, outdir):
    br = np.load(npz)
    ts = [0.0, 0.25, 0.5, 0.75, 1.0]
    strip = [to_hu(br[f"t{t:.2f}"]) for t in ts]

    fig = plt.figure(figsize=(TW, 4.3))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.06, 1.0], hspace=0.20)

    # ======================================================== (a) the geometry bridge
    ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.0, 1.03, "(a)  Training: a flow-matching prior on the geometry bridge",
            fontsize=FS_TITLE, weight="bold", color=INK, va="bottom")

    gap = 0.014
    wpan = (1.0 - 4 * gap) / 5.0
    # mid-coronal slices [z, x], z flipped so the vertex is up: the same slice, the same
    # flip and the same uncropped 256x256 frame as the coronal row of Fig. 3
    # (fig_paper_arxiv.py: win(v[:, yc])[::-1]).
    yc = strip[0].shape[1] // 2
    sag = [vol[:, yc, :][::-1] for vol in strip]
    nrow, ncol = sag[0].shape
    bb = ax.get_position(); ax_w_in, ax_h_in = bb.width * TW, bb.height * fig.get_figheight()
    hpan = (wpan * ax_w_in) * (nrow / ncol) / ax_h_in      # exact pixel aspect
    ytop = 0.90
    ypan = ytop - hpan
    sub = {0.0: "uncorrected FDK", 0.5: "half the motion", 1.0: "motion-free FDK"}
    for k, (t, im) in enumerate(zip(ts, sag)):
        xl = k * (wpan + gap)
        a = ax.inset_axes([xl, ypan, wpan, hpan])
        a.imshow(win(im, 40, 420), cmap="gray", vmin=0, vmax=1,
                 interpolation="nearest", aspect="auto")
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color(LINE); sp.set_linewidth(0.5)
        lab = f"$t={t:g}$"
        if t in sub:
            lab += f"   ({sub[t]})"
        ax.text(xl + wpan / 2, ytop + 0.02, lab, ha="center", va="bottom",
                fontsize=FS_LAB, color=INK)

    # the flow axis under the strip
    ya = ypan - 0.075
    arrow(ax, (0.0, ya), (1.0, ya), color=C2, lw=1.4, shrink=0)
    for k in range(5):
        xm = k * (wpan + gap) + wpan / 2
        ax.plot([xm], [ya], "o", ms=3.0, color=C2, zorder=5, mec="white", mew=0.6,
                clip_on=False)
    ax.text(0.0, ya - 0.085,
            r"$x_t=\mathrm{FDK}\!\left(A\!\left(x;\,P_\mathrm{nom}T((1-t)\theta)\right),\,"
            r"P_\mathrm{nom}\right)$"
            r"$\qquad$ target $\;v_\phi(x_t,t)\approx\dfrac{dx_t}{dt}$, "
            r"the projector's exact geometry derivative carried through the linear FDK",
            ha="left", va="center", fontsize=FS_BODY, color=INK)

    # ======================================================== (b) the inference loop
    ax = fig.add_subplot(gs[1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.0, 1.06, "(b)  Inference: blind predictor-corrector loop along the bridge",
            fontsize=FS_TITLE, weight="bold", color=INK, va="bottom")

    # top row: initialisation (left) and output (right) pills
    yp, hp = 0.855, 0.125
    rbox(ax, 0.0, yp, 0.47, hp, "#f6f6f4", LINE, lw=0.7, r=0.02)
    ax.text(0.014, yp + hp / 2, "Initialize", ha="left", va="center", fontsize=FS_BODY,
            color=INK, weight="bold")
    ax.text(0.125, yp + hp / 2,
            r"$x\leftarrow\mathrm{FDK}(y,P_\mathrm{nom}),\;\;\hat\theta\leftarrow 0,"
            r"\;\;t\leftarrow 0$", ha="left", va="center", fontsize=FS_BODY, color=INK)
    rbox(ax, 0.53, yp, 0.47, hp, "#f6f6f4", LINE, lw=0.7, r=0.02)
    ax.text(0.544, yp + hp / 2, "Output", ha="left", va="center", fontsize=FS_BODY,
            color=INK, weight="bold")
    ax.text(0.625, yp + hp / 2,
            r"final iterate $x$ $\;(t=1)$   and   $\mathrm{FDK}(y,P_\mathrm{nom}T(\hat\theta))$",
            ha="left", va="center", fontsize=FS_BODY, color=INK)

    # loop container, full width
    xc, wc, yc, hc = 0.0, 1.0, 0.0, 0.745
    rbox(ax, xc, yc, wc, hc, "none", LINE, lw=0.7, ls=(0, (3, 2)), r=0.02, z=1)
    ax.text(0.5, yc + hc - 0.025,
            r"repeat for $k=1,\dots,N$   ($N=50$,  $\Delta t=1/N$)",
            ha="center", va="top", fontsize=FS_BODY, color=INK2, style="italic")

    bw, bh, by = 0.29, 0.455, 0.175
    bx = [0.012, 0.355, 0.698]
    step_box(ax, bx[0], by, bw, bh, F1, C1, "1", "Predict",
             [r"$x_\mathrm{pred}=x+\Delta t\;v_\phi(x,t)$",
              "frozen 3D U-Net, $32^3$ patches,",
              "two tilings blended"])
    step_box(ax, bx[1], by, bw, bh, F2, C2, "2", "Estimate",
             [r"$\hat\theta=\arg\min_\theta\|A_{P_\mathrm{nom}T(\theta)}\,x_\mathrm{pred}-y\|^2$",
              r"hash-encoded MLP: view $\to$ 6-DoF pose",
              "Adam, 200 iters on 24 views, warm start"])
    step_box(ax, bx[2], by, bw, bh, F3, C3, "3", "Correct",
             [r"$z=\arg\min_z\|A_{P_\mathrm{nom}T(\hat\theta)}\,z-y\|^2$  (5 CG)",
              r"$z\leftarrow z+\kappa\,(D_\mathrm{TV}(z)-z)$,  $\kappa=0.3$",
              r"$x\leftarrow z,\quad t\leftarrow t+\Delta t$"])

    ym = by + bh / 2
    # init -> predict (down into the loop)
    arrow(ax, (bx[0] + bw / 2, yp), (bx[0] + bw / 2, by + bh), color=INK2, lw=0.9)
    # predict -> estimate -> correct, with the quantity handed over
    arrow(ax, (bx[0] + bw, ym), (bx[1], ym), color=INK2, lw=0.9)
    ax.text((bx[0] + bw + bx[1]) / 2, ym + 0.045, r"$x_\mathrm{pred}$", ha="center",
            va="bottom", fontsize=FS_BODY, color=INK2)
    arrow(ax, (bx[1] + bw, ym), (bx[2], ym), color=INK2, lw=0.9)
    ax.text((bx[1] + bw + bx[2]) / 2, ym + 0.045, r"$\hat\theta$", ha="center",
            va="bottom", fontsize=FS_BODY, color=INK2)
    # correct -> output (up out of the loop) after the last step
    arrow(ax, (bx[2] + bw / 2, by + bh), (bx[2] + bw / 2, yp), color=INK2, lw=0.9)
    ax.text(bx[2] + bw / 2 + 0.012, (by + bh + yp) / 2, r"$k=N$", ha="left",
            va="center", fontsize=FS_BODY, color=INK2)
    # feedback: correct -> predict along the container floor
    yf = yc + 0.065
    x3, x1 = bx[2] + bw / 2, bx[0] + bw / 2
    ax.plot([x3, x3, x1], [by, yf, yf], color=INK2, lw=0.9, zorder=4, clip_on=False)
    arrow(ax, (x1, yf), (x1, by), color=INK2, lw=0.9, shrink=0)
    ax.text((x3 + x1) / 2, yf + 0.012, r"$k<N$:  the corrected image enters the next flow step",
            ha="center", va="bottom", fontsize=FS_BODY, color=INK2)

    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fig1_method.{ext}"))
    plt.close(fig)
    print("fig1_method written to", outdir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/fig_assets/bridgeB_p00.npz")
    ap.add_argument("--outdir", default="docs/arxiv")
    a = ap.parse_args()
    main(a.npz, a.outdir)
