"""Diagnose the STATIC FDK: is the SF forward operator putting streaks into it?

WHY. The user spotted streaks in the static-FDK reference panel of the SF-era validation
montages (it 10000) that the ray-march era did not show at the same 360 views. The static FDK is

    static = FDK( A(vol; P_nom), P_nom )

i.e. it involves NO motion, NO network, NO training -- so it is fully deterministic. Since
2026-07-30 the FDK backprojects through LEAP's modular VD kernel too
(`leap_projector.leap_fdk_backproject`; our algorithm, LEAP's operator -- and note LEAP's
`ADJOINT_MODE` still never enters an FDK: `leap_fdk_backproject` forces the VD backprojector
with our 1/w^2 weighting regardless of that flag). That makes this a clean, cheap A/B.

WHAT IT COMPARES, all with the SAME FDK backprojection path, varying only A:
  * `leap_native` -- THE DEPLOYED SCHEME: y simulated by LEAP at the NATIVE grid du*SOD/SDD
                     (612^3 @ 0.4187 mm), inverted on the reconstruction grid (256^3 @ 1 mm).
                     This is what `gen.simulate()` does with its defaults.
  * `leap_1mm`    -- the same operator projecting the COARSE volume, i.e. the inverse-crime
                     ablation `sim_native=False`. Kept because the native/1 mm difference is the
                     whole reason the simulation grid was split off.
  * `gridsample`  -- the independent torch ray-march reference at 1 mm (exact autograd, the role
                     Siddon plays in the sibling). Slow, opt-in via --gridsample; compare it to
                     `leap_1mm`, which is the variant it shares a grid with.
(The SF-era `sf` / `sf_fpv4` modes were removed 2026-07-29: the forward is LEAP's, so patching
`triton_sf._launch` no longer changes anything and the two modes were silently identical.)

Saves, per patient: a 3-plane montage of every variant plus the GT, DIFFERENCE montages against
the reference variant at a tight window (streaks are low-amplitude and invisible at the HU
window), a radial/angular profile of the difference (a streak is a RIDGE in the angular
direction), and aligned metrics vs the GT volume.

    python scripts/diag_static_fdk.py --out data/diag_static_fdk --patients 3
    python scripts/diag_static_fdk.py --out data/diag_static_fdk --patients 1 --gridsample
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fm3d.dataset_cq500 import CQ500Generator
from fm3d.geometry_3d import ConeBeam3DConfig, measured_region_mask
from fm3d.filters import DEFAULT_RAMP_WINDOW
from fm3d.phantom import MU_WATER, hu_to_mu
from fm3d.leap_projector import ADJOINT_MODE
from fm3d.projector_3d import reference_project_3d_batched
from val_fm3d import montage


def panels_png(path, panels, title, lo=0.0, hi=0.045, cmap="gray", dpi=110):
    """3 planes x n panels. Separate from val_fm3d.montage so the window is controllable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(panels)
    fig, ax = plt.subplots(3, n, figsize=(3.4 * n, 9.8), squeeze=False)
    for j, (vol, ttl) in enumerate(panels):
        D, H, W = vol.shape
        for i, sl in enumerate([vol[D // 2], vol[:, H // 2], vol[:, :, W // 2]]):
            a = ax[i][j]
            a.imshow(sl.detach().cpu().float().numpy(), cmap=cmap, vmin=lo, vmax=hi,
                     aspect="auto" if i else "equal", origin="lower")
            a.set_title(ttl if i == 0 else "", fontsize=9)
            a.set_xticks([]); a.set_yticks([])
    for i, nm in enumerate(["axial", "coronal", "sagittal"]):
        ax[i][0].set_ylabel(nm, fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def angular_profile(d: torch.Tensor, mask: torch.Tensor, nbin: int = 180) -> np.ndarray:
    """Mean |difference| binned by in-plane ANGLE on the mid-axial slice.

    A streak artefact is coherent along a direction, so it shows up as a periodic ridge here
    while quadrature noise is flat. Binning by angle (not radius) is the discriminator.
    """
    D, H, W = d.shape
    sl = d[D // 2].abs()
    m = mask[D // 2] if mask is not None else torch.ones_like(sl, dtype=torch.bool)
    yy, xx = torch.meshgrid(torch.arange(H, device=d.device) - (H - 1) / 2,
                            torch.arange(W, device=d.device) - (W - 1) / 2, indexing="ij")
    ang = torch.atan2(yy, xx)                                   # [-pi, pi]
    rad = torch.sqrt(yy ** 2 + xx ** 2)
    keep = m & (rad > 0.15 * W / 2) & (rad < 0.95 * W / 2)
    idx = ((ang + np.pi) / (2 * np.pi) * nbin).long().clamp(0, nbin - 1)
    out = np.zeros(nbin)
    for b in range(nbin):
        sel = keep & (idx == b)
        out[b] = float(sl[sel].mean()) if sel.any() else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/CQ500")
    ap.add_argument("--out", default="data/diag_static_fdk")
    ap.add_argument("--split", default="val")
    ap.add_argument("--patients", type=int, default=3)
    ap.add_argument("--views", type=int, default=360)
    ap.add_argument("--shape", type=int, nargs=3, default=[256, 256, 256])
    ap.add_argument("--gridsample", action="store_true",
                    help="also reconstruct through the torch ray-march reference forward (SLOW)")
    ap.add_argument("--window", default=None,
                    help="ramp apodization; default = filters.DEFAULT_RAMP_WINDOW")
    ap.add_argument("--brain_window", default="40,80",
                    help="HU centre,width for the NARROW montage (default 40,80 = brain)")
    args = ap.parse_args()

    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)
    cfg = ConeBeam3DConfig.thies(n_views=args.views)
    gen = CQ500Generator(args.root, cfg, device=dev, shape=tuple(args.shape), split=args.split)
    meas = measured_region_mask(tuple(args.shape), (gen.dz, gen.dy, gen.dx), cfg,
                               device=dev).bool()

    fkw = {} if args.window is None else dict(window=args.window)

    _ycache: dict = {}

    def static_fdk(idx, vol, variant):
        """FDK(A(vol; P_nom), P_nom). `variant` = "<forward mode>/<ramp window>"; the sinogram
        is cached per mode so two windows cost one forward projection."""
        mode, window = variant.split("/")
        with torch.no_grad():
            if mode not in _ycache:
                if mode == "gridsample":       # the gates' reference pair, not a backend
                    _ycache[mode] = reference_project_3d_batched(
                        vol, gen.P_nom[None], gen.u_coords, gen.v_coords,
                        dx=gen.dx, dy=gen.dy, dz=gen.dz, n_samples=256)
                elif mode == "leap_native":
                    _ycache[mode] = gen.simulate(idx, gen.P_nom[None])
                else:                          # leap_1mm: the inverse-crime ablation
                    _ycache[mode] = gen.project(vol, gen.P_nom[None])
            return gen.fdk(_ycache[mode], gen.P_nom[None], window=window)[0]

    # VARIANTS are (forward mode, ramp window). The deployed scheme is first and is the
    # reference every difference panel is taken against.
    win0 = args.window or DEFAULT_RAMP_WINDOW
    modes = [f"leap_native/{win0}", f"leap_native/hann", f"leap_1mm/{win0}"]
    if args.gridsample:
        modes.append(f"gridsample/{win0}")
    print(f"geometry: SOD {cfg.SOD} SDD {cfg.SDD} | {args.views} views | panel {cfg.nv}x{cfg.nu} "
          f"@ {cfg.du:g} mm | recon grid {args.shape} @ {gen.dx:g} mm")
    print(f"simulation grid: {gen.sim_shape_dhw} @ {gen.sim_voxel_mm:.5f} mm "
          f"(native = du*SOD/SDD)")
    print(f"THE SCHEME, all defaults: forward/adjoint = LEAP modular-beam | "
          f"adjoint mode = {ADJOINT_MODE} | FDK = our algorithm on LEAP's VD backprojector "
          f"(leap_fdk_backproject; ADJOINT_MODE does NOT enter it)")
    print(f"                          ramp window = {args.window or DEFAULT_RAMP_WINDOW} "
          f"(= LEAP ord2 x FBPlowpass(2.0)) | Voronoi angular weight = {gen.angle_weight} | "
          f"sim_native = {gen.sim_native}")
    print(f"modes: {modes}\n")

    rows = []
    for i in range(args.patients):
        pid = gen.records[i]["patient"]
        vol = gen.volume(i)
        gt = vol[0, 0]
        recs = {}
        _ycache.clear()
        # EXCESS SD IN A HOMOGENEOUS BRAIN ROI -- the texture metric that motivated splitting the
        # simulation grid off in the first place (the user saw crosshatch in the SF-era montages;
        # SF-era anchors, same metric: 1 mm sim 21.3 HU under bare ram-lak / 2.9 HU under
        # shepphann, NATIVE sim 1.9 / 0.0). RMSE-vs-GT cannot see it: it is dominated by the
        # bone edges, which is also why `leap_1mm` scores BETTER there -- it projects the very
        # volume it is compared against (the inverse crime), so a lower RMSE is expected and
        # is not a quality win.
        brain = meas & (gt > hu_to_mu(10.0)) & (gt < hu_to_mu(60.0))
        gx = torch.zeros_like(gt)
        gx[:, :, 1:-1] += (gt[:, :, 2:] - gt[:, :, :-2]).abs()
        gx[:, 1:-1] += (gt[:, 2:] - gt[:, :-2]).abs()
        gx[1:-1] += (gt[2:] - gt[:-2]).abs()
        # LOCALIZED to a central box. Spanning the whole brain instead makes the number report
        # the FDK's cone-beam artefacts (cupping, bone streaks), which swamp the voxel-grid
        # TEXTURE this metric exists to see -- measured 23-32 HU whole-brain against ~2 HU in
        # the box, on the same reconstruction.
        box = torch.zeros_like(meas)
        D2, H2, W2 = (n // 2 for n in gt.shape)
        box[D2 - 10:D2 + 10, H2 - 30:H2 + 30, W2 - 30:W2 + 30] = True
        roi = brain & box & (gx < hu_to_mu(-990.0))    # flat GT only: |grad| < 10 HU
        for m in modes:
            recs[m] = static_fdk(i, vol, m)
            r = recs[m]
            dif = (r - gt)[meas]
            rmse = float(dif.pow(2).mean().sqrt())
            sd_r = float(r[roi].std())
            sd_g = float(gt[roi].std())
            ex_hu = (max(sd_r ** 2 - sd_g ** 2, 0.0) ** 0.5) / MU_WATER * 1000.0
            print(f"  p{pid:3d} {m:22s}  mu mean {float(r[meas].mean()):.5f}  "
                  f"rmse-vs-GT {rmse:.4e}  excess sd in flat brain {ex_hu:6.2f} HU "
                  f"(n={int(roi.sum())})")
            rows.append(dict(patient=pid, mode=m, rmse_vs_gt=rmse, excess_sd_hu=ex_hu,
                             mu_mean=float(r[meas].mean())))

        panels_png(os.path.join(args.out, f"static_p{i}_recon.png"),
                   [(gt, "GT volume")] + [(recs[m], f"static FDK [{m}]") for m in modes],
                   f"CQ500 {args.split} p{pid} | STATIC FDK (no motion, no network) | "
                   f"window mu 0-0.045 | ramp {args.window or DEFAULT_RAMP_WINDOW}")

        # ---- NARROW-WINDOW montage. The mu 0-0.045 panel above spans -1000..+1250 HU, which
        # is a bone window: every soft tissue lands in one grey and the brain reads as flat. The
        # texture and the low-contrast anatomy this diagnostic is actually about only appear in a
        # CLINICAL BRAIN WINDOW, so that panel is emitted too, at higher dpi.
        bc, bw = (float(t) for t in args.brain_window.split(","))
        panels_png(os.path.join(args.out, f"static_p{i}_recon_brain.png"),
                   [(gt, "GT volume")] + [(recs[m], f"static FDK [{m}]") for m in modes],
                   f"CQ500 {args.split} p{pid} | STATIC FDK | BRAIN WINDOW "
                   f"{bc:g}/{bw:g} HU | ramp {args.window or DEFAULT_RAMP_WINDOW}",
                   lo=hu_to_mu(bc - 0.5 * bw), hi=hu_to_mu(bc + 0.5 * bw), dpi=170)

        # ---- SCROLLABLE RAWS. The montage is three fixed slices; a streak or a texture that
        # only lives on some slices does not survive that, so every variant also lands on disk
        # as a plain float32 volume for a viewer. Written in HU, not mu: a CT viewer's windows
        # are HU, and mu numbers (~0.02) are unreadable in one. Little-endian, z-major, no
        # header -- the raw_dir/README.txt records the import recipe.
        raw_dir = os.path.join(args.out, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        for name, volm in [("gt", gt)] + [(m.replace("/", "_"), recs[m]) for m in modes]:
            hu = (volm / MU_WATER - 1.0) * 1000.0
            D_, H_, W_ = hu.shape
            hu.detach().cpu().numpy().astype("<f4").tofile(
                os.path.join(raw_dir, f"p{pid}_{name}_{W_}x{H_}x{D_}_float32_HU.raw"))

        # TIGHT-window difference panels: streaks are ~1e-4 mu and invisible at the HU window
        refname = modes[0]                            # the deployed scheme is the reference
        ref = recs[refname]
        dpanels, prof = [], {}
        for m in modes:
            if m == refname:
                continue
            d = recs[m] - ref
            sc = float(d[meas].abs().quantile(0.999)) or 1e-6
            dpanels.append((d, f"{m} - {refname}\np99.9 |d| = {sc:.2e}"))
            prof[m] = angular_profile(d, meas)
        if dpanels:
            sc = max(float(p[0][meas].abs().quantile(0.999)) for p in dpanels) or 1e-6
            panels_png(os.path.join(args.out, f"static_p{i}_diff.png"), dpanels,
                       f"CQ500 p{pid} | static-FDK DIFFERENCE vs [{refname}] | "
                       f"symmetric window +-{sc:.2e} mu", lo=-sc, hi=sc, cmap="RdBu_r")

        if prof:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, a = plt.subplots(figsize=(9, 3.4))
            for m, p in prof.items():
                a.plot(np.linspace(-180, 180, len(p)), p, lw=1.2, label=f"{m} - {refname}")
            a.set_xlabel("in-plane angle [deg]")
            a.set_ylabel("mean |difference| [1/mm]")
            a.set_title(f"p{pid}: ANGULAR profile of the static-FDK difference "
                        f"(a streak is a periodic ridge; quadrature noise is flat)")
            a.legend(fontsize=8)
            a.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(args.out, f"static_p{i}_angular.png"), dpi=115)
            plt.close(fig)

        torch.save({m: recs[m].half().cpu() for m in modes} | {"gt": gt.half().cpu(),
                                                               "patient": pid},
                   os.path.join(args.out, f"static_p{i}.pt"))

    with open(os.path.join(args.out, "raw", "README.txt"), "w") as fh:
        D_, H_, W_ = args.shape
        fh.write(
            "float32 HU volumes, no header, little-endian, z-major (slice = one x-y plane).\n"
            f"filenames encode WxHxD = {W_}x{H_}x{D_}; voxel {gen.dx:g} mm isotropic.\n\n"
            "Fiji/ImageJ:  File > Import > Raw...\n"
            f"  Image type   32-bit Real\n  Width {W_}   Height {H_}   Number of images {D_}\n"
            "  Offset 0   Gap 0   Little-endian byte order CHECKED\n"
            "  then Image > Adjust > Brightness/Contrast, e.g. brain 0/80 HU, bone -200/2000\n\n"
            "python:\n"
            f"  import numpy as np; v = np.fromfile(p, '<f4').reshape({D_}, {H_}, {W_})\n\n"
            "variants (see the script docstring):\n"
            "  gt                        the CQ500 volume the sinogram was simulated FROM\n"
            "  leap_native_<window>      THE DEPLOYED SCHEME (native-grid simulation)\n"
            "  leap_1mm_<window>         the inverse-crime ablation, same operator at 1 mm\n")
    torch.save({"rows": rows, "args": vars(args), "adjoint_mode": ADJOINT_MODE,
                "window": args.window or DEFAULT_RAMP_WINDOW,
                "sim_grid": gen.sim_shape_dhw, "sim_voxel_mm": gen.sim_voxel_mm},
               os.path.join(args.out, "summary.pt"))
    print(f"\n-> {args.out}/  (static_p*_recon.png, _diff.png, _angular.png, *.pt)")


if __name__ == "__main__":
    main()
