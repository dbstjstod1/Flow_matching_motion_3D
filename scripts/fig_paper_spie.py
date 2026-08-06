"""SPIE Medical Imaging manuscript figures (4-page format, 6.75 in text width).

Three figures, written to figs/spie/:
    fig1_method.png   -- the geometry bridge (training) + the blind loop (inference)
    fig2_images.png   -- qualitative axial/coronal comparison, representative patient
    fig3_quant.png    -- per-patient SSIM and the motion-error convergence over the 30-patient cohort

Colours are the three validated categorical slots (blue / orange / aqua); they are the only
hues used, and every series is also directly labelled so identity is never colour-alone.

    python scripts/fig_paper_spie.py --scratch <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------- style
TW = 6.75                       # SPIE single-column text width, inches
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8a85"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
})

MU_WATER = 0.02

# Head bounding box on the 256^3 grid (measured from the GT mask, plus a margin). Cropping
# to it stops ~20% of every panel being air, and lets the panels shrink for the page budget.
CROP_AX = (22, 246, 18, 238)    # axial  (y0,y1,x0,x1) -> 224 x 220
CROP_CO = (38, 216, 18, 238)    # coronal (z0,z1,x0,x1) -> 178 x 220


def to_hu(v):
    return (v - MU_WATER) / MU_WATER * 1000.0


def win(img, level=40.0, width=400.0):
    """Grey-window an HU image to [0,1]."""
    lo, hi = level - width / 2, level + width / 2
    return np.clip((img - lo) / (hi - lo), 0, 1)


def assert_crop_covers_head(gt_mu, zc, yc, tag):
    """Fail loudly if CROP_AX / CROP_CO would cut into the patient.

    The crops are fixed constants (so the figure's aspect ratio, and therefore the page
    layout, does not move when the displayed patient changes). They were measured on p04 and
    verified on p00 -- but a different patient could sit anywhere in the 256^3 box, and a
    silently clipped skull is exactly the kind of error that survives to print.
    """
    m = gt_mu > MU_WATER * 0.35            # anything denser than about -650 HU
    for plane, mask, (r0, r1, c0, c1) in (("axial", m[zc], CROP_AX),
                                          ("coronal", m[:, yc], CROP_CO)):
        rr, cc = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
        got = (rr[0], rr[-1], cc[0], cc[-1])
        if got[0] < r0 or got[1] >= r1 or got[2] < c0 or got[3] >= c1:
            raise SystemExit(
                f"{tag}: the {plane} crop {(r0, r1, c0, c1)} clips the head {got} -- "
                f"widen CROP_AX/CROP_CO (and re-check the page count, the aspect changes)")


def box(ax, x, y, w, h, fc, ec, title, lines, tfs=7.2, lfs=6.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc=fc, ec=ec, lw=0.9, zorder=2, clip_on=False))
    ax.text(x + w / 2, y + h - 0.052, title, ha="center", va="top", fontsize=tfs,
            color=INK, weight="bold", zorder=3)
    ax.text(x + w / 2, y + h - 0.135, "\n".join(lines), ha="center", va="top",
            fontsize=lfs, color=INK, zorder=3, linespacing=1.45)


def arrow(ax, p0, p1, color=MUTED, lw=1.0, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=8,
                                 color=color, lw=lw, linestyle=ls, zorder=4, clip_on=False,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1.5, shrinkB=1.5))


# ================================================================ FIGURE 1
# The two bridge designs. (b), the inference loop, is IDENTICAL for both -- only the training
# path in (a) differs, which is the whole point of the A/B distinction.
BRIDGE = {
    # NB every string carrying LaTeX is raw: "$T(t\theta)$" in a normal literal makes \t a TAB.
    "A": dict(
        npz="bridge_p00.npz",
        title="(a)  Training: a flow-matching prior on the GEOMETRY bridge",
        eq=r"$x_t=\mathrm{FDK}\!\left(y,\,P_\mathrm{nom}T(t\theta)\right)+t\Delta$",
        note="the target is ANALYTIC (the BACKprojector's exact geometry derivative)",
        foot=r"Every point is an FDK reconstruction carrying exactly $(1-t)$ of the true "
             r"motion, and the anchor $t\Delta$ pins $x_1$ to the static FDK.",
    ),
    "B": dict(
        npz="bridgeB_p00.npz",
        title="(a)  Training: a flow-matching prior on the GEOMETRY bridge",
        eq=r"$x_t=\mathrm{FDK}\!\left(A(x;P_\mathrm{nom}T((1-t)\theta)),\,P_\mathrm{nom}\right)$",
        note="the target is ANALYTIC (the FORWARD projector's exact geometry derivative)",
        foot=r"Every point is an FDK reconstruction carrying exactly $(1-t)$ of the true "
             r"motion, so the path ends on the motion-free static FDK BY CONSTRUCTION.",
    ),
}


def fig1(scratch, outdir, bridge="A"):
    cfg = BRIDGE[bridge]
    br = np.load(os.path.join(scratch, cfg["npz"]))
    ts = [0.0, 0.25, 0.5, 0.75, 1.0]
    strip = [to_hu(br[f"t{t:.2f}"]) for t in ts]
    sl = strip[0].shape[0] // 2

    fig = plt.figure(figsize=(TW, 3.60))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.62, 1.0], hspace=0.20)

    # ---------------- (a) the geometry bridge, shown as REAL reconstructions
    ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.0, 1.055, cfg["title"], fontsize=8.2, weight="bold", color=INK, va="bottom")

    # the five thumbnails, spanning the full text width
    gap = 0.013
    wpan = (1.0 - 4 * gap) / 5.0
    x0 = 0.0
    for k, (t, vol) in enumerate(zip(ts, strip)):
        xl = x0 + k * (wpan + gap)
        a = ax.inset_axes([xl, 0.315, wpan, 0.625])
        y0c, y1c, x0c, x1c = CROP_AX
        a.imshow(win(vol[sl, y0c:y1c, x0c:x1c], 40, 420), cmap="gray", vmin=0, vmax=1,
                 interpolation="nearest")
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_color(MUTED); s.set_linewidth(0.6)
        # va="bottom" anchored ABOVE the inset's top edge (0.425+0.480=0.905): the insets are
        # child axes and paint OVER the parent's text, so an overlapping label loses its bottom half
        ax.text(xl + wpan / 2, 0.955, f"$t={t:g}$", ha="center", va="bottom",
                fontsize=7.2, color=INK)
    xend = 1.0

    # the flow arrow underneath the strip
    arrow(ax, (x0, 0.258), (xend, 0.258), color=C2, lw=1.5)
    for k in range(4):
        xm = x0 + k * (wpan + gap) + wpan + gap / 2
        ax.plot([xm], [0.258], "o", ms=2.6, color=C2, zorder=5, mec="white", mew=0.5,
                clip_on=False)

    ax.text(0.0, 0.198, cfg["eq"] + "        "
                        r"$v_\phi(x_t,t)\approx dx_t/dt$ — " + cfg["note"],
            ha="left", va="top", fontsize=6.5, color=INK)
    ax.text(0.0, 0.078, cfg["foot"], ha="left", va="top", fontsize=6.5,
            color=INK, style="italic")

    # ---------------- (b) the blind loop
    ax = fig.add_subplot(gs[1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.0, 1.05, "(b)  Inference: blind rigid-motion correction — one Euler step of "
                       "$N=50$   ($\\theta$ and $x$ bootstrap each other)",
            fontsize=8.2, weight="bold", color=INK, va="bottom")

    ax.text(0.0, 0.995, r"cold start:  $x\leftarrow\mathrm{FDK}(y,P_\mathrm{nom})$,  "
                        r"$\hat\theta\leftarrow 0$", fontsize=6.6, color=INK, va="top")

    y0, h = 0.135, 0.66
    box(ax, 0.012, y0, 0.296, h, "#eef4fc", C1, "1.  PREDICT — the prior moves",
        [r"$x_\mathrm{pred}=x+\Delta t\; v_\phi(x,t)$",
         "frozen UNet, evaluated patch-wise",
         "and blended over 2 tilings"], lfs=6.2)
    box(ax, 0.350, y0, 0.300, h, "#fdf1ec", C2, "2.  ESTIMATE the motion",
        [r"$\hat\theta=\arg\min_\theta\|A_{\theta}\,x_\mathrm{pred}-y\|^2$",
         "6-DoF per view, hash-encoding MLP,",
         "200 iters/step, 24 views/iter, warm start"], lfs=6.2)
    box(ax, 0.692, y0, 0.296, h, "#eaf7f2", C3, "3.  CORRECT — PnP data step",
        [r"$z\leftarrow\arg\min_z\|A_{\hat\theta}z-y\|^2$   (5 CG its)",
         r"$z\leftarrow z+\kappa\,(\mathrm{TV}(z)-z)$",
         r"$x\leftarrow z,\quad t\leftarrow t+\Delta t$"], lfs=6.2)

    arrow(ax, (0.308, y0 + h / 2), (0.350, y0 + h / 2), color=MUTED, lw=1.1)
    arrow(ax, (0.646, y0 + h / 2), (0.692, y0 + h / 2), color=MUTED, lw=1.1)
    ax.add_patch(FancyArrowPatch((0.845, y0), (0.155, y0), arrowstyle="-|>", mutation_scale=8,
                                 color=MUTED, lw=1.0, zorder=1, linestyle=(0, (4, 2)),
                                 connectionstyle="arc3,rad=0.13", shrinkA=2, shrinkB=2))
    ax.text(0.50, -0.055, "Gauss–Seidel: the motion is always fitted on the image the prior "
                         "just improved", ha="center", va="bottom", fontsize=6.5,
            color=INK, style="italic")

    os.makedirs(outdir, exist_ok=True)
    suffix = "" if bridge == "A" else f"_{bridge}"
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fig1_method{suffix}.{ext}"))
    plt.close(fig)
    print(f"fig1 written (bridge {bridge})")


# ================================================================ FIGURE 2
def fig2(scratch, outdir, tag="p00"):
    """The five panels of run_posterior3d's own `montage`, at a brain window.

    TWO things here are not negotiable, both settled in run_posterior3d.montage's docstring:

    * The volumes are RIGIDLY ALIGNED to the GT. Blind motion recon has an exact SE(3) gauge,
      so each reconstruction sits at its own arbitrary pose and slicing them raw at z = D//2
      puts DIFFERENT anatomical planes side by side. (`result.pt` stores the raw volumes --
      `x_final_al`/`x_al` are the aligned ones -- so the alignment is redone here.)
    * The reference panel is FDK(theta_TRUE), NOT the static FDK (user, 2026-07-27): the static
      FDK reconstructs a DIFFERENT, motion-free scan that no method working on this data can
      reach with an FDK, so showing it as the reference panel overstates the gap. The static FDK
      stays in Table 1, where it is labelled for what it is.
    """
    z = np.load(os.path.join(scratch, f"vols_{tag}_aligned.npz"))
    gt, cold = to_hu(z["gt"]), to_hu(z["cold"])
    xfin, xt, ceil = to_hu(z["xfin"]), to_hu(z["xt"]), to_hu(z["ceil"])

    rows = pickle.load(open(os.path.join(scratch, "test30.pkl"), "rb"))
    m = {r["tag"]: r for r in rows}[tag]

    panels = [
        (cold, "Uncorrected FDK", f"{m['cold']['psnr_aligned']:.2f} dB / {m['cold']['ssim_aligned']:.3f}"),
        (xfin, r"$\mathrm{FDK}(\hat\theta)$  (ours)", f"{m['out_gt']['psnr_aligned']:.2f} dB / {m['out_gt']['ssim_aligned']:.3f}"),
        (xt,   r"$x_t$  (ours, deliverable)", f"{m['xt_gt']['psnr_aligned']:.2f} dB / {m['xt_gt']['ssim_aligned']:.3f}"),
        (ceil, "FDK at GT motion", f"{m['oracle_psnr']:.2f} dB / {m['oracle_ssim']:.3f}"),
        (gt,   "Ground-truth image", ""),
    ]
    zc, yc = gt.shape[0] // 2, gt.shape[1] // 2
    ay0, ay1, ax0, ax1 = CROP_AX
    cz0, cz1, cx0, cx1 = CROP_CO
    h_ax, h_co, w_pan = ay1 - ay0, cz1 - cz0, ax1 - ax0
    assert_crop_covers_head(z["gt"], zc, yc, tag)

    W = 5.4
    fig, axes = plt.subplots(2, 5, figsize=(W, W / 5 * (h_ax + h_co) / w_pan + 0.30),
                             gridspec_kw={"height_ratios": [h_ax, h_co]})
    for j, (vol, title, score) in enumerate(panels):
        for i, im in enumerate([vol[zc, ay0:ay1, ax0:ax1],
                                vol[cz0:cz1, yc, cx0:cx1][::-1]]):
            a = axes[i, j]
            a.imshow(win(im, 40, 420), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_color(MUTED); s.set_linewidth(0.5)
            if i == 0:
                a.set_title(title, fontsize=6.4, color=INK, pad=7.5)
                if score:
                    a.text(0.5, 1.015, score, transform=a.transAxes, ha="center", va="bottom",
                           fontsize=5.7, color=INK2)
        axes[0, 0].set_ylabel("axial", fontsize=6.4, color=INK2)
        axes[1, 0].set_ylabel("coronal", fontsize=6.4, color=INK2)
    fig.subplots_adjust(wspace=0.03, hspace=0.035, top=0.885, bottom=0.005,
                        left=0.028, right=0.998)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fig2_images.{ext}"))
    plt.close(fig)
    print("fig2 written")


# ================================================================ FIGURE 3
def fig3(scratch, outdir):
    rows = pickle.load(open(os.path.join(scratch, "test30.pkl"), "rb"))
    rows = sorted(rows, key=lambda r: r["tag"])
    cur = json.load(open(os.path.join(scratch, "rpe_curves.json")))
    cur = {c["tag"]: c for c in cur}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(TW, 2.20),
                                   gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.26})

    # ---- (a) per-patient SSIM, paired
    cold = np.array([r["cold"]["ssim_aligned"] for r in rows])
    ours = np.array([r["xt_gt"]["ssim_aligned"] for r in rows])
    ref = np.array([r["sfdk_ssim"] for r in rows])
    order = np.argsort(cold)
    xx = np.arange(len(rows))

    for k in xx:
        axA.plot([k, k], [cold[order][k], ours[order][k]], color="#d8d8d4", lw=0.7, zorder=1)
    axA.plot(xx, ref[order], "_", ms=6, mew=1.3, color=C3, zorder=2,
             label=f"motion-free FDK (reference)   {ref.mean():.3f}")
    axA.plot(xx, ours[order], "o", ms=3.0, color=C2, zorder=3, mec="white", mew=0.4,
             label=f"ours, $x_t$   {ours.mean():.3f}")
    axA.plot(xx, cold[order], "o", ms=3.0, color=C1, zorder=3, mec="white", mew=0.4,
             label=f"uncorrected FDK   {cold.mean():.3f}")

    axA.set_xlabel("patient (30 held-out test cases, sorted by uncorrected SSIM)", labelpad=2)
    axA.set_ylabel("SSIM  (rigid-aligned, vs. ground truth)")
    axA.set_ylim(0.40, 1.03)
    axA.set_xlim(-1, len(rows))
    axA.set_xticks([])
    axA.grid(axis="y", color="#ececea", lw=0.5, zorder=0)
    axA.set_axisbelow(True)
    for s in ("top", "right"):
        axA.spines[s].set_visible(False)

    handles, labels = axA.get_legend_handles_labels()
    idx = [1, 0, 2]      # ours, reference, uncorrected
    leg = axA.legend([handles[i] for i in idx], [labels[i] for i in idx],
                     loc="center left", bbox_to_anchor=(0.005, 0.36), frameon=False,
                     handlelength=1.0, handletextpad=0.5, labelspacing=0.30, fontsize=6.6)
    for t, i in zip(leg.get_texts(), idx):
        t.set_color(INK2)
    axA.set_title("(a)  every patient improves", fontsize=8, loc="left", color=INK, pad=4)

    # ---- (b) motion-error convergence
    steps = cur[rows[0]["tag"]]["steps"]
    t = np.array(steps) / 50.0
    C = np.array([cur[r["tag"]]["rpe_curve"] for r in rows])
    r0 = np.array([cur[r["tag"]]["rpe0"] for r in rows])

    for i in range(C.shape[0]):
        axB.plot(t, C[i], color=C2, lw=0.5, alpha=0.22, zorder=2)
    axB.plot(t, C.mean(0), color=C2, lw=1.8, zorder=4, label="ours (mean of 30)")
    axB.axhline(r0.mean(), color=C1, lw=1.2, ls=(0, (4, 2)), zorder=3,
                label="uncorrected")

    axB.set_yscale("log")
    axB.set_xlabel("flow time  $t$  (ODE step / 50)", labelpad=2)
    axB.set_ylabel("RPE  (mm)")
    axB.set_xlim(-0.02, 1.0)
    axB.set_ylim(0.12, 9.0)
    axB.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.2, 0.5, 1, 2, 5]))
    axB.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axB.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:g}"))
    axB.grid(axis="y", color="#ececea", lw=0.5, zorder=0)
    axB.set_axisbelow(True)
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)
    axB.annotate(f"{r0.mean():.2f} mm", (0.99, r0.mean()), xytext=(0, -11),
                 textcoords="offset points", ha="right", fontsize=6.8, color=C1, weight="bold")
    axB.annotate(f"{C.mean(0)[-1]:.2f} mm", (t[-1], C.mean(0)[-1]), xytext=(-2, -11),
                 textcoords="offset points", ha="right", fontsize=6.8, color=C2, weight="bold")
    leg = axB.legend(loc="lower left", bbox_to_anchor=(0.0, -0.02), frameon=False,
                     handlelength=1.4, handletextpad=0.5, labelspacing=0.30, fontsize=6.6)
    for tx in leg.get_texts():
        tx.set_color(INK2)
    axB.set_title("(b)  motion converges with the flow", fontsize=8, loc="left", color=INK, pad=4)

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fig3_quant.{ext}"))
    plt.close(fig)
    print("fig3 written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--outdir", default="figs/spie")
    ap.add_argument("--only", default=None)
    ap.add_argument("--fig2_tag", default="p00", help="which test30 patient Fig. 2 shows")
    ap.add_argument("--bridge", default="A", choices=["A", "B"],
                    help="A = the deployed geometry bridge (+anchor); B = motion attenuated in "
                         "the data (no anchor). Only Fig. 1(a) differs.")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    if a.only in (None, "1"):
        fig1(a.scratch, a.outdir, bridge=a.bridge)
    if a.only in (None, "2"):
        fig2(a.scratch, a.outdir, tag=a.fig2_tag)
    if a.only in (None, "3"):
        fig3(a.scratch, a.outdir)
