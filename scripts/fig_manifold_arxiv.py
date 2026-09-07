"""Concept (manifold) figure for the arXiv manuscript -- no GPU, pure schematic.

    python scripts/fig_manifold_arxiv.py            # -> figs/arxiv/fig0_manifold.{png,pdf}

Two panels over the same image-space cartoon:
  (a) a clean-image generative prior: its transport path runs from Gaussian noise to the
      clean manifold, so the states a blind correction loop actually visits (the family of
      motion-corrupted FDKs) lie outside its training distribution;
  (b) the geometry bridge: the SAME family of states IS the training path, the analytic
      tangent is defined at every point, and inference walks it with the estimator and the
      data step keeping the iterate on the bridge while t runs to 1.

Style matches the manuscript's data figures (fig_paper_spie.py): serif, the three
categorical slots, direct labels, no chartjunk.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8a85"
RED = "#c8452c"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 7.5,
    "figure.dpi": 400, "savefig.dpi": 400,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def bezier(p0, p1, p2, s):
    s = np.asarray(s)[:, None]
    return (1 - s) ** 2 * p0 + 2 * (1 - s) * s * p1 + s ** 2 * p2


# The bridge curve: from the uncorrected reconstruction (upper left, off-manifold) into the
# clean manifold (lower right). Shared by both panels so the two stories are comparable.
P0 = np.array([0.09, 0.84])
P1 = np.array([0.44, 0.64])
P2 = np.array([0.76, 0.17])
S = np.linspace(0, 1, 200)
CURVE = bezier(P0, P1, P2, S)


def manifold(ax):
    """The clean-reconstruction manifold: a smooth wobbled ellipse, lower right."""
    phi = np.linspace(0, 2 * np.pi, 400)
    r = 1 + 0.10 * np.sin(2 * phi + 0.7) + 0.06 * np.cos(3 * phi)
    cx, cy, rx, ry = 0.83, 0.115, 0.235, 0.205
    xs = cx + rx * r * np.cos(phi)
    ys = cy + ry * r * np.sin(phi)
    ax.fill(xs, ys, color=C3, alpha=0.13, lw=0, zorder=0)
    ax.plot(xs, ys, color=C3, lw=0.9, alpha=0.75, zorder=1)
    ax.text(cx + 0.02, cy - 0.075, "clean (static) CT\nreconstructions",
            ha="center", va="center", fontsize=6.6, color="#0e7a54", zorder=2)
    ax.text(cx + rx * 0.92, cy + ry * 1.02, r"$\mathcal{M}$", fontsize=9.5,
            color="#0e7a54", zorder=2)


def base(ax, title):
    ax.set_xlim(-0.02, 1.10)
    ax.set_ylim(-0.10, 1.13)
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(0.00, 1.115, title, fontsize=8.2, fontweight="bold", va="top", color=INK)
    manifold(ax)
    ax.text(0.00, -0.085, "image space", fontsize=6.6, color=MUTED, style="italic")


def curve_dir(s):
    d = 2 * (1 - s) * (P1 - P0) + 2 * s * (P2 - P1)
    return d / np.linalg.norm(d)


def panel_a(ax):
    base(ax, "(a)  diffusion posterior sampling")
    # noise cloud, upper right
    rng = np.random.default_rng(3)
    pts = rng.normal([0.88, 0.88], 0.038, (110, 2))
    ax.scatter(pts[:, 0], pts[:, 1], s=2.2, color=MUTED, alpha=0.55, lw=0, zorder=2)
    ax.text(0.88, 0.975, "noise $z\\sim\\mathcal{N}(0,I)$\n(the loop starts here)",
            ha="center", va="bottom", fontsize=6.8, color=INK2)
    # its fixed generative path: noise -> manifold (ORANGE, matching the training-path
    # color of panel b). Drawn as an EXPLICIT bezier polyline so the schedule circles sit
    # exactly on it (user's catch, 2026-09-01: the arc3 control estimate had put them off
    # the curve). The path IS the noise schedule; its fixed steps are the open circles.
    # the path ends near M's CENTER, not its rim -- the sampler models the clean
    # distribution itself (user's call, 2026-09-01)
    Ao, Bo = np.array([0.865, 0.80]), np.array([0.83, 0.13])
    Co = np.array([0.932, 0.53])
    OP = bezier(Ao, Co, Bo, S)
    ax.plot(OP[:, 0], OP[:, 1], color=C2, lw=1.6, zorder=3)
    ax.add_patch(FancyArrowPatch(tuple(OP[-8]), tuple(OP[-1]), color=C2, lw=1.6,
                                 arrowstyle="-|>", mutation_scale=11, zorder=3))
    ax.text(0.945, 0.62, "noise schedule\n(sampling path)", fontsize=6.6, color=C2,
            ha="left", va="center")
    n0 = bezier(Ao, Co, Bo, [0.10])[0]
    n1 = bezier(Ao, Co, Bo, [0.40])[0]
    n2 = bezier(Ao, Co, Bo, [0.68])[0]
    n3 = bezier(Ao, Co, Bo, [0.90])[0]
    for q in (n0, n1, n2, n3):
        ax.plot(*q, "o", ms=4.0, color=C2, mfc="white", mew=1.1, zorder=5)
    # the states a blind correction loop traverses: NO marked points -- the corrected
    # estimates land at arbitrary positions along this family (user's call, 2026-09-01)
    ax.plot(CURVE[:, 0], CURVE[:, 1], ls=(0, (4, 3)), color=MUTED, lw=1.5, zorder=2)
    ax.plot(*P0, "o", ms=4.2, color=INK2, zorder=3)
    ax.text(P0[0] + 0.035, P0[1] + 0.045, "uncorrected reconstruction",
            fontsize=6.6, color=INK2, ha="left")
    # the DPS-style walk, ping-pong idiom with the user's 09-01 trajectory semantics:
    # the FIRST endpoint prediction lands at an ARBITRARY clean CT (deep in M, far from
    # the data); the motion + data correction then yanks the estimate back to the
    # data-consistent artifact family, its first landing NEAR THE START of the grey line
    # (the uncorrected reconstruction); across iterations the blue landings creep toward
    # the grey line's entry into M and the green landings march along the family toward M.
    # landings (user's semantics, 2026-09-03, final): FAIRNESS -- panel b idealizes our
    # correct step as landing ON the bridge, so the same idealization applies here: the
    # green corrections land ON the grey family (near its goal end, advancing toward it).
    # The blue predictions stay INSIDE M and on the RIGHT of the grey line (the orange
    # side), so the blue arrows never cross the grey family.
    b1 = np.array([0.865, 0.100])               # prediction 1: an arbitrary clean CT
    b2 = np.array([0.795, 0.165])               # prediction 2: closer to the goal
    c1 = bezier(P0, P1, P2, [0.70])[0]          # correction 1: ON the family, near goal
    c2 = bezier(P0, P1, P2, [0.85])[0]          # correction 2: further along it

    def cyc(a, b, color, rad, lw=1.3):
        ax.add_patch(FancyArrowPatch(tuple(a), tuple(b), color=color, lw=lw,
                                     arrowstyle="-|>", mutation_scale=8, zorder=4,
                                     connectionstyle=f"arc3,rad={rad}"))

    cyc(n0, b1, C1, 0.12)
    cyc(b1, c1, C3, 0.15)
    cyc(c1, n1, RED, -0.08)
    cyc(n1, b2, C1, -0.05, lw=1.1)
    cyc(b2, c2, C3, 0.10, lw=1.1)
    cyc(c2, n2, RED, -0.08, lw=1.0)
    for b in (b1, b2):
        ax.plot(*b, "o", ms=2.8, color=C1, zorder=5)
    for c in (c1, c2):
        ax.plot(*c, "o", ms=2.8, color=C3, zorder=5)
    ax.text(0.868, 0.290, r"$\cdots$", fontsize=10, color=INK2, ha="center",
            va="center", rotation=-85, zorder=4)
    # legend for the cycle, lower left (mirrors panel b's legend style)
    def leg(y, color, txt):
        ax.add_patch(FancyArrowPatch((0.015, y), (0.068, y), color=color, lw=1.2,
                                     arrowstyle="-|>", mutation_scale=6.5,
                                     shrinkA=0, shrinkB=0, zorder=3))
        ax.text(0.078, y, txt, fontsize=6.3, color=INK2, va="center")

    leg(0.315, C1, "1. endpoint prediction $\\hat{x}_0$")
    leg(0.258, C3, "2. estimate $\\hat\\theta$ + data step")
    leg(0.201, RED, "3. re-noise, onto the next schedule step")


def panel_b(ax):
    base(ax, "(b)  proposed geometry bridge")
    # the SAME curve, now the training path itself
    ax.plot(CURVE[:, 0], CURVE[:, 1], color=C2, lw=2.2, zorder=2)
    # amplitude annotation: guide arc above the curve
    a0 = bezier(P0, P1, P2, [0.10])[0] + [0.06, 0.05]
    a1 = bezier(P0, P1, P2, [0.90])[0] + [0.10, 0.06]
    ax.annotate("", xy=tuple(a1), xytext=tuple(a0),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9,
                                connectionstyle="arc3,rad=-0.15"))
    ax.text(0.755, 0.575, "motion amplitude $(1-t)\\,\\theta$\nshrinks linearly to zero",
            fontsize=6.6, color=INK2, ha="center")
    # bridge samples with t labels (below-left of each dot)
    for s, lab in ((0.0, "$t=0$"), (0.33, "$t=0.33$"), (0.66, "$t=0.66$"), (1.0, "$t=1$")):
        p = bezier(P0, P1, P2, [s])[0]
        ax.plot(*p, "o", ms=4.0, color=C2, mec="white", mew=0.7, zorder=4)
        ax.text(p[0] - 0.030, p[1] - 0.052, lab, fontsize=6.2, color=INK2,
                ha="center", zorder=4)
    # (velocity expression + black tangent arrows removed 2026-09-03, user's call:
    # cleaner without them; the analytic-tangent story stays in the caption)
    # endpoints
    ax.text(P0[0] + 0.035, P0[1] + 0.045,
            "$x_0=$ FDK$(y,\\,P_{\\mathrm{nom}})$   (uncorrected)", fontsize=6.6,
            color=INK2, ha="left")
    ax.text(P2[0] + 0.035, P2[1] + 0.035, "$x_1=$ static FDK $\\in\\mathcal{M}$",
            fontsize=6.6, color=INK2, ha="left", va="bottom")
    # inference walk on the lower side: predict along the tangent overshoots slightly,
    # estimate+correct pulls back onto the bridge
    ss = np.linspace(0.03, 0.93, 9)

    def step_arrow(a, b, color):
        ax.add_patch(FancyArrowPatch(tuple(a), tuple(b), color=color, lw=1.0,
                                     arrowstyle="-|>", mutation_scale=6.0,
                                     shrinkA=0, shrinkB=0, zorder=3))

    prev = bezier(P0, P1, P2, [ss[0]])[0]
    for i, s in enumerate(ss[:-1]):
        nxt = bezier(P0, P1, P2, [ss[i + 1]])[0]
        p = bezier(P0, P1, P2, [s])[0]
        n = curve_dir(s)
        n = np.array([n[1], -n[0]])                       # lower-side normal
        off = 0.5 * (p + nxt) + n * 0.032 * (1.0 - 0.55 * s)
        step_arrow(prev, off, C1)
        step_arrow(off, nxt, C3)
        prev = nxt
    # legend for the walk, lower left (free area)
    step_arrow(np.array([0.015, 0.325]), np.array([0.068, 0.325]), C1)
    ax.text(0.078, 0.325, "1. predict (frozen prior)", fontsize=6.3, color=INK2,
            va="center")
    step_arrow(np.array([0.015, 0.265]), np.array([0.068, 0.265]), C3)
    ax.text(0.078, 0.265, "2. estimate $\\hat\\theta$ + data step (CG)", fontsize=6.3,
            color=INK2, va="center")
    ax.text(0.015, 0.195, "image and motion converge\ntogether as $t\\to 1$",
            fontsize=6.3, color=INK2, style="italic", va="top")


def main():
    outdir = "figs/arxiv"
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.95))
    panel_a(axes[0])
    panel_b(axes[1])
    fig.subplots_adjust(wspace=0.05, left=0.005, right=0.995, top=0.99, bottom=0.01)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fig0_manifold.{ext}"))
    print("wrote", os.path.join(outdir, "fig0_manifold.{png,pdf}"))


if __name__ == "__main__":
    main()
