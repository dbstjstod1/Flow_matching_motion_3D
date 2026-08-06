"""Schematic workflow figures for talks: (1) prior TRAINING, (2) blind INFERENCE loop.

Pure matplotlib (no external SVG tooling), so it regenerates anywhere the project runs:

    python scripts/fig_workflow.py            # -> figs/workflow_{blind,manifold,training,inference,overview}.*
    python scripts/fig_workflow.py --font_scale 1.3 --suffix _big     # the projector-friendly set

The layout is fixed and every font size is multiplied by --font_scale, so the two settings above
are the gated ones: 1.0 and 1.3 were both checked panel by panel for text that runs into a
neighbour. Push the scale higher and you have to re-check (and probably shorten a caption).

Every label is taken from the code it describes, so keep them in sync when the pipeline moves:
  training  -- scripts/train_fm3d.py (bridge, anchor, cache, optimizer) + fm3d/prior_patch.py
               (patch tiling, in_ch=5 global context)
  inference -- scripts/run_posterior3d.py (predictor-corrector loop, dc_op=cg, theta readout)
               + fm3d/motion_estimation.py (MotionNet6DoF, l2) + fm3d/tv.py (TV corrector)
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---- palette: one hue per ROLE, reused across both figures -------------------------------
DATA = ("#dbeafe", "#1d4ed8")        # measured data / geometry operator
GEO = ("#fef3c7", "#b45309")         # the bridge / FDK path
PRIOR = ("#ede9fe", "#6d28d9")       # the learned FM prior
EST = ("#dcfce7", "#15803d")         # motion estimation
CORR = ("#ffe4e6", "#be123c")        # data consistency + TV corrector
OUT = ("#e2e8f0", "#0f172a")         # deliverables / bookkeeping

FIG_W, FIG_H = 160.0, 90.0           # data units per panel (16 x 9 in at 10 units/in)
FS = 1.0                             # global font scale (--font_scale, for talks)


def box(ax, x, y, w, h, title, body="", color=OUT, ts=10.0, bs=8.0, ls="-", ha="center"):
    """Rounded box with a bold title line and a centered body block. (x, y) = lower-left."""
    fc, ec = color
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                fc=fc, ec=ec, lw=1.6, ls=ls, zorder=2))
    if title:
        ax.text(x + w / 2, y + h - 3.1, title, ha="center", va="center", fontsize=ts * FS,
                fontweight="bold", color=ec, zorder=3)
    if body:
        yb = y + (h - 5.6) / 2 if title else y + h / 2
        xb = x + w / 2 if ha == "center" else x + 3.5
        ax.text(xb, yb, body, ha=ha, va="center", fontsize=bs * FS,
                color="#111827", linespacing=1.45, zorder=3)


def arrow(ax, p0, p1, color="#334155", lw=1.8, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=15 * FS, lw=lw,
                                 color=color, linestyle=ls, zorder=4,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1.5, shrinkB=1.5))


def tag(ax, x, y, text, color="#334155", fs=8.0, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs * FS, color=color,
            fontstyle="italic",
            zorder=5)


def band(ax, y, text, x=2.5, ha="left"):
    """Stage caption (the numbered rows)."""
    ax.text(x, y, text, ha=ha, va="center", fontsize=10.5 * FS, fontweight="bold", color="#0f172a")


def foot(ax, y, text, fs=7.9):
    ax.text(FIG_W / 2, y, text, ha="center", va="center", fontsize=fs * FS,
            color="#475569",
            fontstyle="italic")


def frame(ax):
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_aspect("equal")
    ax.axis("off")


# ==========================================================================================
# 1. TRAINING
# ==========================================================================================
def draw_training(ax):
    frame(ax)
    ax.text(FIG_W / 2, 86.5, "Training: a flow-matching prior on the GEOMETRY BRIDGE",
            ha="center", va="center", fontsize=15.5 * FS, fontweight="bold", color="#0f172a")
    foot(ax, 82.0, "the prior is trained on exactly the image manifold the inference ODE walks: "
                   "reconstructions under a PARTIALLY CORRECTED geometry", fs=9.0)

    # ---- 1. simulate a motion-corrupted scan ---------------------------------------------
    band(ax, 77.0, "1.  Simulate a motion-corrupted scan   (CQ500, train split)")
    w, h, y1 = 33.0, 13.5, 61.5
    xs = [3.0, 41.0, 79.0, 117.0]
    box(ax, xs[0], y1, w, h, "Head volume  $x_{GT}$",
        "CQ500, $256^3$ @ 1 mm\nHU $\\rightarrow\\ \\mu$ (1/mm)", DATA)
    box(ax, xs[1], y1, w, h, "Random rigid motion  $\\theta$",
        "Akima spline, 10 nodes, per view\n6-DoF, 10 mm / 10$\\degree$ peak-to-peak", DATA)
    box(ax, xs[2], y1, w, h, "Cone-beam forward projection",
        "$y = A(x_{GT};\\, P_{nom}T(\\theta))$\nSID 785 / SDD 1200, 360 views", DATA)
    box(ax, xs[3], y1, w, h, "Motion-corrupted scan  $y$",
        "$500\\times700$ panel\n(= one bridge draw)", DATA)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + w, y1 + h / 2), (b, y1 + h / 2))

    # ---- 2. the geometry bridge ----------------------------------------------------------
    band(ax, 56.0, "2.  Geometry bridge   (whole volume, no_grad — one fused FDK+tangent pass)")
    arrow(ax, (133.0, y1), (133.0, 50.0), color="#b45309")
    tag(ax, 138.5, 55.7, "$y,\\ \\theta$", color="#b45309")

    box(ax, 3.0, 34.5, 26.0, 15.5, "Draw the time  $t$",
        "$t \\sim \\mathcal{U}[0,1]$\n(uniform, no weighting)", GEO)
    box(ax, 32.0, 34.5, 74.0, 15.5, "Anchored bridge state  +  its EXACT tangent",
        "$x_t=\\mathrm{FDK}(y,\\,P_{nom}T(t\\theta))\\;+\\;t\\,\\Delta$\n"
        "$dx_t/dt=\\partial_s\\mathrm{FDK}(y,\\,P_{nom}T(s\\theta))|_{s=t}+\\Delta$   "
        "(one fused Triton kernel)", GEO, bs=9.0)
    box(ax, 109.0, 34.5, 48.0, 15.5, "Anchor  $\\Delta$  (endpoint detrend)",
        "$\\Delta=x_{static}-\\mathrm{FDK}(y,\\,P_{nom}T(\\theta))$\n"
        "$x_{static}$ = FDK of the MOTION-FREE scan", GEO)
    arrow(ax, (29.0, 42.2), (32.0, 42.2))
    arrow(ax, (109.0, 42.2), (106.0, 42.2))

    ax.text(32.0, 31.0, "$t=0$:  $x_0=\\mathrm{FDK}(y,P_{nom})$ — the uncorrected recon "
            "= the inference cold start",
            ha="left", va="center", fontsize=8.3 * FS, color="#7c2d12")
    ax.text(32.0, 27.6, "$t=1$:  $x_1=x_{static}$ — the motion-free scan "
            "(the bare endpoint FDK($\\theta_{true}$) is 1–3 dB short)",
            ha="left", va="center", fontsize=8.3 * FS, color="#7c2d12")

    # ---- 3. patch prior ------------------------------------------------------------------
    band(ax, 23.5, "3.  Train the patch prior", x=157.0, ha="right")
    y3, h3 = 7.0, 13.5
    box(ax, 3.0, y3, 30.0, h3, "Patch sampling",
        "$32^3$ crops, origins inside\nthe MEASURED-REGION mask", PRIOR)
    box(ax, 36.0, y3, 34.0, h3, "Input channels: $in\\_ch=5$",
        "ch0 $x_t$ patch | ch1 whole $x_t$ $\\downarrow$ to $32^3$\n"
        "ch2–4 absolute $z,y,x$  (global context)", PRIOR)
    box(ax, 73.0, y3, 27.0, h3, "UNet3D   $v_\\psi(x_t,t)$",
        "base 32, $t$-embedding\ntorch.compile", PRIOR)
    box(ax, 103.0, y3, 25.0, h3, "Flow-matching loss",
        "$\\Vert v_\\psi(x_t,t)-dx_t/dt\\Vert ^2$\n(target: ch 0 only)", PRIOR)
    box(ax, 131.0, y3, 26.0, h3, "Optimizer",
        "AdamW $10^{-4}\\!\\rightarrow\\!10^{-6}$ cosine\nEMA 0.999, fp16 AMP, 500k it", PRIOR)
    for a, b in [(33.0, 36.0), (70.0, 73.0), (100.0, 103.0), (128.0, 131.0)]:
        arrow(ax, (a, y3 + h3 / 2), (b, y3 + h3 / 2), color="#6d28d9")
    arrow(ax, (18.0, 34.5), (18.0, y3 + h3), color="#6d28d9")
    tag(ax, 21.5, 26.0, "$x_t$", color="#6d28d9")
    arrow(ax, (96.0, 34.5), (112.0, y3 + h3), color="#6d28d9", rad=-0.15)
    tag(ax, 115.0, 30.0, "$dx_t/dt$", color="#6d28d9")

    foot(ax, 5.0, "MEMORY SPLIT — the OPERATOR (FDK + its tangent) runs on the FULL volume under "
                  "no_grad; the NET only ever sees $32^3$ patches.")
    foot(ax, 2.9, "A rolling cache of 8 whole-volume bridge draws is refreshed every 12 steps; "
                  "each batch of 64 patches mixes several draws, so $t$ varies within a batch.")
    foot(ax, 0.8, "Validation = prior-only ODE from the cold start on held-out patients: "
                  "25.42 dB / SSIM 0.78 (rigid-aligned).")


# ==========================================================================================
# 2. INFERENCE
# ==========================================================================================
def draw_inference(ax):
    frame(ax)
    ax.text(FIG_W / 2, 86.5, "Inference: blind rigid-motion correction — a PnP-TV "
            "predictor–corrector loop",
            ha="center", va="center", fontsize=15.5 * FS, fontweight="bold", color="#0f172a")
    foot(ax, 82.0, "$\\theta$ is UNKNOWN: the image and the motion bootstrap each other, "
                   "one Euler step of the FM ODE at a time ($N=50$ steps)", fs=9.0)

    # ---- inputs --------------------------------------------------------------------------
    yt, ht = 66.5, 12.0
    box(ax, 3.0, yt, 32.0, ht, "Measured scan  $y$",
        "motion-corrupted, blind\n(Akima 10 mm / 10$\\degree$ p2p test motion)", DATA)
    box(ax, 40.0, yt, 34.0, ht, "Cold start",
        "$x \\leftarrow \\mathrm{FDK}(y,P_{nom})$,   $\\hat\\theta \\leftarrow 0$", DATA)
    box(ax, 79.0, yt, 42.0, ht, "Trained FM prior  $v_\\psi$   (FROZEN)",
        "evaluated patch-wise and blended:\nnon-overlapping random tiling, $K=2$", PRIOR)
    arrow(ax, (35.0, yt + ht / 2), (40.0, yt + ht / 2))

    # ---- the loop panel ------------------------------------------------------------------
    ax.add_patch(FancyBboxPatch((3.0, 20.0), 154.0, 41.5,
                                boxstyle="round,pad=0,rounding_size=2.0",
                                fc="#ffffff", ec="#334155", lw=1.7, ls="--", zorder=1))
    ax.text(6.0, 58.0, "for  $k=0\\ldots N-1$,   $t=k/N$,   $N=50$",
            ha="left", va="center", fontsize=10.2 * FS, fontweight="bold", color="#334155")

    yl, hl = 28.5, 25.0
    box(ax, 6.0, yl, 44.0, hl, "1.  PREDICT — the prior moves first",
        "$x_{prior}=x+dt\\cdot v_\\psi(x,t)$\n\n"
        "evaluated on $32^3$ tiles carrying the\n"
        "global-context channels, then blended\nback to the full volume",
        PRIOR, ts=9.6, bs=8.2)
    box(ax, 55.0, yl, 46.0, hl, "2.  ESTIMATE — on the IMPROVED image",
        "$\\hat\\theta=\\arg\\min_\\theta\\; L_{2}(A(x_{prior};P_{nom}T(\\theta)),\\; y)$\n\n"
        "MotionNet6DoF over the view index\n(Instant-NGP, full band, lr $3\\!\\cdot\\!10^{-3}$)\n"
        "400 it/step, 24 views/it, warm-started,\ncoarse-to-fine estimation grid",
        EST, ts=9.6, bs=8.2)
    box(ax, 106.0, yl, 45.0, hl, "3.  CORRECT — PnP forward–backward",
        "data step (CG, 5 it, LEAP VD adjoint):\n"
        "$z \\approx \\arg\\min_z \\Vert A_{\\hat\\theta}\\,z-y\\Vert ^2$\n\n"
        "TV corrector:  $z \\leftarrow z+\\kappa\\,(\\mathrm{TV}(z)-z)$\n"
        "$\\kappa=0.3$, 5 inner it, step 0.015",
        CORR, ts=9.6, bs=8.2)
    arrow(ax, (50.0, yl + hl / 2), (55.0, yl + hl / 2))
    tag(ax, 52.5, yl + hl / 2 + 2.8, "$x_{prior}$")
    arrow(ax, (101.0, yl + hl / 2), (106.0, yl + hl / 2))
    tag(ax, 103.5, yl + hl / 2 + 2.8, "$\\hat\\theta$")

    # feedback edge, bowing BELOW the stage boxes
    arrow(ax, (128.5, yl), (28.0, yl), color="#0f172a", rad=-0.13, lw=1.9)
    tag(ax, 78.0, 25.7, "$x\\leftarrow z$,   $t\\leftarrow t+dt$   — Gauss–Seidel: the motion is "
        "always fitted on the better image", color="#0f172a", fs=8.4)

    # feeds into the loop
    arrow(ax, (57.0, yt), (45.0, yl + hl), color="#1d4ed8")
    tag(ax, 57.5, 61.0, "$x$", color="#1d4ed8")
    arrow(ax, (96.0, yt), (30.0, yl + hl), color="#6d28d9", rad=0.10)
    tag(ax, 62.0, 57.0, "frozen  $v_\\psi$", color="#6d28d9")
    arrow(ax, (19.0, yt), (19.0, 63.0), color="#1d4ed8", ls="--", lw=1.4, style="-")
    arrow(ax, (19.0, 63.0), (140.0, 63.0), color="#1d4ed8", ls="--", lw=1.4, style="-")
    arrow(ax, (140.0, 63.0), (140.0, yl + hl), color="#1d4ed8", ls="--", lw=1.4)
    arrow(ax, (78.0, 63.0), (78.0, yl + hl), color="#1d4ed8", ls="--", lw=1.4)
    tag(ax, 110.0, 64.6, "the measured $y$ is the reference for BOTH the motion fit and the "
        "data step", color="#1d4ed8")

    # ---- outputs -------------------------------------------------------------------------
    band(ax, 15.8, "Readout")
    box(ax, 3.0, 3.0, 46.0, 10.5, "Motion estimate",
        "$\\hat\\theta$ = the last step's estimate\n($K$-step averaging available offline)",
        EST, ts=9.4, bs=8.0)
    box(ax, 54.0, 3.0, 48.0, 10.5, "Two deliverables",
        "output $=\\mathrm{FDK}(y,\\,P_{nom}T(\\bar\\theta))$\n"
        "carried PnP state $x_t$ (often the better one)", CORR, ts=9.4, bs=8.0)
    box(ax, 107.0, 3.0, 50.0, 10.5, "Scored under the SE(3) gauge",
        "rigid-align, then PSNR/SSIM vs GT and vs static FDK;\n"
        "motion by RPE — never a raw translation RMSE", OUT, ts=9.4, bs=8.0)
    arrow(ax, (26.0, 20.0), (26.0, 13.5), color="#15803d")
    arrow(ax, (78.0, 20.0), (78.0, 13.5), color="#be123c")
    arrow(ax, (49.0, 8.25), (54.0, 8.25))
    arrow(ax, (102.0, 8.25), (107.0, 8.25))


# ==========================================================================================
# 3. THE PROBLEM STATEMENT: y = A_P x with BOTH P and x unknown
# ==========================================================================================
def draw_blind(ax):
    frame(ax)
    ax.text(FIG_W / 2, 86.5, "The problem is BLIND:  one measurement,  two unknowns",
            ha="center", va="center", fontsize=15.5 * FS, fontweight="bold", color="#0f172a")
    foot(ax, 82.3, "$P$ cannot be corrected without a clean $x$ — and $x$ cannot be "
                   "reconstructed without the right $P$", fs=9.2)

    # ---- the forward model -----------------------------------------------------------------
    box(ax, 3.0, 60.0, 154.0, 18.0, "", "", ("#f8fafc", "#94a3b8"))
    ax.text(24.0, 72.6, "$y \\;=\\; A_{P(\\theta)}\\; x$", ha="center", va="center",
            fontsize=21 * FS, color="#0f172a")
    ax.text(24.0, 64.6, "$P(\\theta)=P_{nom}\\,T(\\theta)$ — the nominal geometry,\n"
            "times the rigid motion we did not see", ha="center", va="center", fontsize=8.4 * FS,
            color="#334155", linespacing=1.45)
    box(ax, 47.0, 62.0, 34.0, 14.0, "$y$  —  KNOWN",
        "the measured projections\n$360\\times500\\times700 = 126$ M values", DATA, ts=10.4)
    box(ax, 84.0, 62.0, 34.0, 14.0, "$\\theta$  —  UNKNOWN",
        "6 DoF per view (a rigid head pose)\n$6\\times360 = 2\\,160$ parameters", EST, ts=10.4)
    box(ax, 121.0, 62.0, 33.0, 14.0, "$x$  —  UNKNOWN",
        "the volume we are after\n$256^3 = 16.8$ M voxels", CORR, ts=10.4)

    # ---- the circular dependency -----------------------------------------------------------
    box(ax, 3.0, 31.0, 58.0, 25.0, "Correcting  $P$  needs a clean  $x$",
        "$\\hat\\theta=\\arg\\min_\\theta \\Vert A_{P(\\theta)}\\,x-y\\Vert ^2$\n\n"
        "the fit is only as good as the image you hand it —\n"
        "a streaked $x$ happily explains the streaks.\n\n"
        "MEASURED (our estimator sweep): from the COLD FDK\n"
        "every configuration plateaus at $\\approx 2.0\\degree$; from a clean\n"
        "reference the same budget reaches $0.09$–$0.27\\degree$.",
        EST, ts=10.2, bs=8.2)
    box(ax, 99.0, 31.0, 58.0, 25.0, "Reconstructing  $x$  needs the true  $P$",
        "$\\hat x=\\arg\\min_x \\Vert A_{P(\\theta)}\\,x-y\\Vert ^2$\n\n"
        "a wrong $\\theta$ does not just blur — it BAKES its own\n"
        "artefacts into whatever image the solver returns.\n\n"
        "MEASURED (same loop, oracle $\\theta$ from step 0):\n"
        "$x_t$ reaches 40.45 dB with the true geometry,\n"
        "38.16 dB when $\\theta$ has to be estimated blind.",
        CORR, ts=10.2, bs=8.2)

    arrow(ax, (61.0, 50.0), (99.0, 50.0), color="#15803d", lw=2.0, rad=-0.30)
    arrow(ax, (99.0, 37.0), (61.0, 37.0), color="#be123c", lw=2.0, rad=-0.30)
    tag(ax, 80.0, 58.2, "hand it a better $x$", color="#15803d", fs=8.6)
    tag(ax, 80.0, 29.0, "hand it a better $\\hat\\theta$", color="#be123c", fs=8.6)
    ax.text(80.0, 45.5, "CIRCULAR", ha="center", va="center", fontsize=11.5 * FS,
            fontweight="bold", color="#0f172a")
    ax.text(80.0, 41.5, "DEPENDENCY", ha="center", va="center", fontsize=11.5 * FS,
            fontweight="bold", color="#0f172a")

    # ---- why it is hard, and what breaks it ------------------------------------------------
    box(ax, 3.0, 3.0, 75.0, 24.0, "Counting says it is solvable — the difficulty is elsewhere",
        "126 M measurements vs 16.8 M + 2 160 unknowns: NOT underdetermined.\n\n"
        "• the coupling is BILINEAR in $(x,\\theta)$ — the joint problem is non-convex,\n"
        "   and every local minimum is an image that explains the data\n"
        "• an EXACT SE(3) GAUGE: $(x,\\theta)$ and $(g\\!\\cdot\\!x,\\;\\theta\\circ g^{-1})$ "
        "give the SAME $y$,\n   so the solution is an orbit, never a single pose\n"
        "• per-view translation ALONG THE BEAM is nearly unobservable\n"
        "   (92% of a converged estimator's residual lives there)",
        OUT, ts=10.0, bs=8.2, ha="left")
    box(ax, 82.0, 3.0, 75.0, 24.0, "What breaks the cycle",
        "$\\min_{x,\\,\\theta}\\;\\frac{1}{2}\\Vert A_{P(\\theta)}x-y\\Vert ^2+"
        "\\lambda\\,\\mathrm{TV}(x)$"
        "   +   a learned prior on $x$\n\n"
        "1.  the PRIOR supplies what the data cannot: it improves $x$ without knowing $\\theta$\n"
        "2.  ALTERNATE (Gauss–Seidel): fit $\\theta$ on the improved $x$, then solve $x$ under it\n"
        "3.  the GEOMETRY BRIDGE keeps every intermediate $x$ a real reconstruction,\n"
        "     so the $\\theta$ fit never runs on an image the operator cannot produce\n"
        "4.  score modulo the gauge: rigid-align before PSNR/SSIM, report RPE for $\\theta$",
        PRIOR, ts=10.0, bs=8.2, ha="left")


# ==========================================================================================
# 4. THE CONCEPT PICTURE: a path to the clean manifold, held on the physics
# ==========================================================================================
def _bez(P, s):
    """Cubic Bezier point + unit tangent at parameter s (P: 4 control points)."""
    import numpy as np
    P = np.asarray(P, float)
    s = np.asarray(s, float)[..., None]
    b = ((1 - s) ** 3 * P[0] + 3 * (1 - s) ** 2 * s * P[1]
         + 3 * (1 - s) * s ** 2 * P[2] + s ** 3 * P[3])
    d = (3 * (1 - s) ** 2 * (P[1] - P[0]) + 6 * (1 - s) * s * (P[2] - P[1])
         + 3 * s ** 2 * (P[3] - P[2]))
    return b, d / np.linalg.norm(d, axis=-1, keepdims=True)


def draw_manifold(ax):
    import numpy as np
    frame(ax)
    ax.text(FIG_W / 2, 86.5, "Why the loop works: a learned path to the clean manifold, "
            "held on the physics at every step",
            ha="center", va="center", fontsize=15.5 * FS, fontweight="bold", color="#0f172a")
    foot(ax, 82.0, "the prior supplies the DIRECTION (it was trained on this very path); "
                   "the measured data supplies the CONSTRAINT that keeps the direction honest",
         fs=9.0)

    # ---- the two manifolds ----------------------------------------------------------------
    CLEAN = ("#ccfbf1", "#0f766e")
    ctrl = [(24.0, 24.0), (62.0, 17.0), (90.0, 54.0), (124.0, 63.5)]     # geometry bridge
    ss = np.linspace(0, 1, 200)
    bpts, _ = _bez(ctrl, ss)

    # clean manifold: a broad ribbon the bridge lands on
    cm = np.array([(98.0, 74.0), (124.0, 60.5), (152.0, 58.0)])
    cs = np.linspace(0, 1, 120)[:, None]
    cpts = (1 - cs) ** 2 * cm[0] + 2 * (1 - cs) * cs * cm[1] + cs ** 2 * cm[2]
    ax.plot(cpts[:, 0], cpts[:, 1], lw=22, color=CLEAN[0], solid_capstyle="round", zorder=1)
    ax.plot(cpts[:, 0], cpts[:, 1], lw=2.2, color=CLEAN[1], zorder=2)
    ax.text(152.0, 71.0, "$\\mathcal{M}_{clean}$", ha="right", va="center", fontsize=13 * FS,
            fontweight="bold", color=CLEAN[1], zorder=3)
    ax.text(152.0, 66.5, "motion-free images —\nwhat the scan would have been",
            ha="right", va="center", fontsize=8.4 * FS, color=CLEAN[1], zorder=3, linespacing=1.4)

    # the start: a blob of motion-corrupted reconstructions
    from matplotlib.patches import Ellipse
    ax.add_patch(Ellipse((24.0, 24.0), 34.0, 21.0, angle=-14, fc="#dbeafe", ec="#1d4ed8",
                         lw=1.6, alpha=0.55, zorder=1))
    ax.text(24.0, 30.0, "START", ha="center", va="center", fontsize=10.5 * FS, fontweight="bold",
            color="#1d4ed8", zorder=3)
    ax.text(24.0, 25.2, "$x_0=\\mathrm{FDK}(y,P_{nom})$", ha="center", va="center",
            fontsize=9.5 * FS, color="#111827", zorder=3)
    ax.text(24.0, 20.4, "streaked — and consistent\nwith the WRONG geometry",
            ha="center", va="center", fontsize=8.2 * FS, color="#1e3a8a", zorder=3, linespacing=1.4)

    # the bridge itself
    ax.plot(bpts[:, 0], bpts[:, 1], lw=16, color="#fef3c7", solid_capstyle="round", zorder=1)
    ax.plot(bpts[:, 0], bpts[:, 1], lw=2.2, color="#b45309", zorder=2)
    ax.text(60.0, 25.5, "GEOMETRY BRIDGE   $\\{\\mathrm{FDK}(y,\\,P_{nom}T(s\\theta))\\}$",
            ha="left", va="center", fontsize=10.2 * FS, fontweight="bold", color="#b45309", zorder=3)
    ax.text(60.0, 21.3, "every point on it is a REAL reconstruction of the REAL data —\n"
            "and it is exactly the path the prior was trained on ($v_\\psi \\approx dx_t/dt$)",
            ha="left", va="center", fontsize=8.4 * FS, color="#7c2d12", zorder=3, linespacing=1.4)

    # ---- the data-consistency sets: the physics, re-aimed as theta-hat converges ------------
    for s0, ang, ln, lab, al in ((0.10, 62.0, 13.0, "$\\mathcal{C}(\\hat\\theta_0)$", 0.55),
                                 (0.50, 44.0, 17.0, "$\\mathcal{C}(\\hat\\theta_k)$", 0.75),
                                 (0.94, 26.0, 17.0, "$\\mathcal{C}(\\theta_{true})$", 1.0)):
        p, _ = _bez(ctrl, np.array([s0]))
        p = p[0]
        u = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
        d = u * ln
        ax.plot([p[0] - d[0], p[0] + d[0]], [p[1] - d[1], p[1] + d[1]], lw=1.7, ls=(0, (6, 3)),
                color="#1d4ed8", alpha=al, zorder=2)
        lp = p + d + np.array([-u[1], u[0]]) * 3.8            # off the line, away from the path
        ax.text(lp[0], lp[1], lab, ha="center", va="center",
                fontsize=9.2 * FS, color="#1d4ed8", alpha=al, zorder=3)
    ax.text(96.0, 32.0, "$\\mathcal{C}(\\hat\\theta)=\\{x:\\;A_{\\hat\\theta}\\,x\\approx y\\}$  "
            "— the constraint that PHYSICALLY EXISTS:\nthe measured projections. It is not a "
            "regularizer we chose, it is the scan.",
            ha="left", va="center", fontsize=8.6 * FS, color="#1d4ed8", zorder=3, linespacing=1.5)
    ax.text(96.0, 39.0, "the ESTIMATE stage is what re-aims it:  "
            "$\\mathcal{C}(\\hat\\theta_0)\\rightarrow\\mathcal{C}(\\hat\\theta_k)"
            "\\rightarrow\\mathcal{C}(\\theta_{true})$,\nuntil the feasible set finally "
            "CONTAINS the clean image.",
            ha="left", va="center", fontsize=8.6 * FS, color="#15803d", zorder=3, linespacing=1.5)

    # ---- the loop's zig-zag: prior step out, data step back on -----------------------------
    s_nodes = np.linspace(0.06, 0.94, 8)
    pts, tan = _bez(ctrl, s_nodes)
    nrm = np.stack([-tan[:, 1], tan[:, 0]], axis=1)              # left normal (toward clean)
    for i in range(len(s_nodes) - 1):
        p, q = pts[i], pts[i + 1]
        off = p + tan[i] * (0.72 * np.linalg.norm(q - p)) + nrm[i] * 5.4   # prior overshoots
        arrow(ax, tuple(p), tuple(off), color="#6d28d9", lw=1.7)
        arrow(ax, tuple(off), tuple(q), color="#be123c", lw=1.7)
        ax.plot(*p, "o", ms=5.0, color="#b45309", zorder=5)
        # ESTIMATE happens exactly HERE -- at the vertex between the two arrows, on the image the
        # prior just produced. It moves theta, not x, which is why it is a POINT and not a step.
        ax.plot(*off, "o", ms=6.2, mfc="#15803d", mec="white", mew=1.2, zorder=6)
    ax.plot(*pts[-1], "o", ms=6.5, color=CLEAN[1], zorder=5)

    # ---- what each force alone would do ----------------------------------------------------
    p_f = pts[2]
    arrow(ax, tuple(p_f), (69.0, 63.0), color="#94a3b8", lw=1.6, ls=(0, (5, 3)), rad=-0.18)
    ax.text(66.0, 70.5, "prior alone:\na plausible image that does NOT\nexplain the measurement",
            ha="left", va="center", fontsize=8.2 * FS, color="#64748b", fontstyle="italic",
            linespacing=1.5, zorder=3)
    arrow(ax, (34.0, 14.0), (96.0, 11.5), color="#94a3b8", lw=1.6, ls=(0, (5, 3)), rad=0.06)
    ax.text(99.0, 11.2, "data step alone, at the geometry we started with ($\\hat\\theta=0$): "
            "perfectly consistent — and still streaked",
            ha="left", va="center", fontsize=8.2 * FS, color="#64748b", fontstyle="italic", zorder=3)

    # ---- the legend IS the loop: one step, three moves --------------------------------------
    box(ax, 4.0, 51.5, 58.0, 28.5, "ONE STEP OF THE LOOP  (x50)", "",
        ("#f8fafc", "#334155"), ts=10.2)
    # The green row is a POINT, not an arrow: ESTIMATE sits at the vertex BETWEEN the purple and
    # the red arrow, and it moves theta, not x -- so it has no length in this picture.
    rows = ((72.5, "#6d28d9", "arrow", "PREDICT — the prior steps toward "
                                       "$\\mathcal{M}_{clean}$\n"
                                       "it knows the direction: it was TRAINED on this path"),
            (65.0, "#15803d", "dot", "ESTIMATE — at the vertex between the two arrows:\n"
                                     "$\\hat\\theta$ is refit on the image the prior just made.\n"
                                     "It moves $\\theta$, not $x$ — so $\\mathcal{C}(\\hat"
                                     "\\theta)$ (dashed) re-aims instead."),
            (56.5, "#be123c", "arrow", "CORRECT — the data step pulls it back onto\n"
                                       "$\\mathcal{C}(\\hat\\theta)$; the TV corrector keeps it "
                                       "regular"))
    for yr, col, glyph, txt in rows:
        if glyph == "dot":
            ax.plot(10.25, yr, "o", ms=6.2 * FS, mfc=col, mec="white", mew=1.2, zorder=5)
        else:
            arrow(ax, (7.5, yr), (13.0, yr), color=col, lw=1.9)
        ax.text(15.0, yr, txt, ha="left", va="center", fontsize=8.4 * FS, color="#111827",
                linespacing=1.5, zorder=3)
    box(ax, 4.0, 3.0, 74.0, 6.4, "", "the three moves alternate $N=50$ times — no single one "
        "reaches $\\mathcal{M}_{clean}$", ("#f8fafc", "#94a3b8"), bs=8.4)
    ax.text(116.0, 48.0, "landing:  $\\bar\\theta$  and the carried  $x_t$", ha="center",
            va="center", fontsize=9.2 * FS, color=CLEAN[1], fontweight="bold", zorder=3)
    arrow(ax, (119.0, 50.5), (123.0, 61.0), color=CLEAN[1], lw=1.3, rad=-0.2)


# ==========================================================================================
def save(fig, path_noext, dpi, formats):
    for ext in formats:
        p = f"{path_noext}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.15, facecolor="white")
        print(f"wrote {p}")


def main():
    global FS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--formats", nargs="+", default=["png", "pdf", "svg"])
    ap.add_argument("--font_scale", type=float, default=1.0,
                    help="multiply every font size (and the arrowheads). The layout is "
                         "unchanged, so large values eventually overflow the boxes; 1.3 is the "
                         "gated presentation setting (--suffix _big).")
    ap.add_argument("--suffix", default="",
                    help="appended to every output filename, e.g. --suffix _big")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    FS = args.font_scale
    sfx = args.suffix

    for name, draw in (("workflow_blind", draw_blind),
                       ("workflow_manifold", draw_manifold),
                       ("workflow_training", draw_training),
                       ("workflow_inference", draw_inference)):
        fig, ax = plt.subplots(figsize=(16, 9))
        draw(ax)
        fig.subplots_adjust(0, 0, 1, 1)
        save(fig, os.path.join(args.out, name + sfx), args.dpi, args.formats)
        plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(16, 36))
    draw_blind(axes[0])
    draw_manifold(axes[1])
    draw_training(axes[2])
    draw_inference(axes[3])
    fig.subplots_adjust(0, 0, 1, 1, hspace=0.04)
    save(fig, os.path.join(args.out, "workflow_overview" + sfx), args.dpi,
         args.formats)
    plt.close(fig)


if __name__ == "__main__":
    main()
