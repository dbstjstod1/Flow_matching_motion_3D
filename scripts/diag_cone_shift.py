"""Is the dark horizontal banding at the SKULL BASE a cone-beam effect? (standalone probe)

WHY. In `scripts/diag_static_fdk.py`'s brain-window montage (`data/diag_static_fdk_leap/
static_p*_recon_brain.png`) the coronal and sagittal panels carry dark horizontal bands through
the skull base that the GT panel does not have. They are IDENTICAL in all three variants there
(leap_native/shepphann, leap_native/hann, leap_1mm) -- so they are neither the ramp window nor
the simulation grid. The user's hypothesis: it is the CONE ANGLE, i.e. lift the anatomy toward
the focal plane and the bands should go.

WHERE THEY ARE, measured on the stored p216 reconstruction (mean signed error in brain voxels,
per axial slice, high-passed with an 11-slice boxcar to separate banding from cupping):

    z band [mm]     -90..-80   -80..-70   -70..-60   -40..+40   +60..+70
    mean bias [HU]      -441       -114        +9      -2..-7        -23
    HP RMS   [HU]        223         26        16      < 1.1         19

So the defect is NOT global: it is flat to under 1 HU across the middle of the volume and blows
up below z = -60 mm, which is 60-90% of the way out to the measured barrel's half height
(half_v*(SOD-r)/SDD = 105 mm on axis, 97 mm at r = 60 mm). Both candidate causes live exactly
there -- the Feldkamp approximation degrades with cone angle, AND the object runs off the panel
in v (the 256 mm box is taller than the 209 mm axial coverage), so this probe has to separate
them rather than just confirm "cone".

THE TRICK -- SHIFT THE GEOMETRY, NOT THE VOLUME. Right-multiply an SE(3) translation into P:

    P'(dz) = P_nom @ T(+dz * z_hat)

`(P @ T) x = P(x + dz z_hat)`, so projecting the ORIGINAL volume through P' yields exactly the
sinogram of that volume translated UP by dz, and `FDK(y, P')` -- a consistent object/geometry
pair -- returns the reconstruction at the ORIGINAL voxel indices. No roll, no interpolation, no
anatomy pushed out of the box, GT untouched, every variant comparable slice by slice. The
object-frame plane z = -dz lands on the cone MIDPLANE, so `--dz_mm 70` puts the streak band at
zero cone angle. (`leap_projector.decompose_P`: a right SE(3) preserves P's form, so LEAP needs
no special case; and a pure z translation leaves the source's beta untouched, so the Voronoi
angular weights stay uniform and cannot confound the A/B.)

VARIANTS -- the headline plus the three controls that make it conclusive. There are only three
candidate causes for a z-local defect in a static FDK, and each gets its own ablation:
  base           the deployed geometry, bit-identical to `diag_static_fdk`'s leap_native
  up<dz>         object lifted dz mm: the streak band moves toward the midplane. Costs axial
                 coverage at the TOP of the head (the barrel moves down with the object), which
                 is why the metrics also report a common mask.
  down<dz>       object LOWERED: the negative control. If the cone angle is the cause the defect
                 must get WORSE here, not just better under `up`.
  tall<k>        panel k x taller in v (same pitch), object left where it is: removes the axial
                 truncation while KEEPING the cone angle -- separates missing data from Feldkamp.
  crop<c>        object zeroed outside |z| <= c mm on the deployed panel: tests whether the
                 contamination is the material the panel never sees.
  vblur<s>       sinogram Gaussian-blurred along v by s detector pixels before FDK. The recon
                 grid samples z at 1 mm while the panel samples it at 0.419 mm at the isocentre,
                 so sharp bone structure in v is ALIASED into z bands by the backprojection.
                 s = dz*M/2 = 1.2 px band-limits v to the recon's own z Nyquist -- this is the
                 row filter clinical FDK applies for exactly this reason.

THE ANSWER (p216, val, 360 views, shepphann, 2026-07-29). IT IS THE CONE ANGLE. Banding metric =
RMS of the high-passed per-slice bias over z in [-75, -45] mm, the band the montage's streaks
live in:

    variant     HP-RMS   HP p2p   bias    brain RMSE (band)      what it changes
    down20      16.35     50.0   -14.75      55.46              further from the midplane
    base        16.18     62.8    -7.00      44.63              deployed
    up35         8.83     51.2    -3.32      35.35              halfway to the midplane
    up70         2.78     13.5    -4.79      34.72              band ON the midplane
    tall2       16.19     62.8    -7.01      44.64              no axial truncation
    crop100     16.18     62.8    -7.00      44.63              no unseen material
    vblur1.2    15.78     61.1    -6.44      46.50              v band-limited to z Nyquist

Monotone in the shift and it reverses under the negative control -- 5.8x less banding once the
band sits on the midplane -- while the three non-cone ablations are NO-OPS to 4 significant
figures. And the defect follows the GANTRY, not the anatomy: `up70`'s own midplane band (|z| <=
40 mm, which the lift moved OFF the midplane) gets WORSE, 2.53 -> 3.83 HU. In the deployed
geometry the z dependence is

    z [mm]   0..20   -20..0   -40..-20   -60..-40   -80..-60   -100..-80
    HP-RMS    0.25     0.58       4.70       6.36      33.39      252.05   HU

i.e. the classic circular-orbit FDK cone artefact off dense bone: exact only in the midplane,
and the skull base is both the densest structure and 60-90 mm below it. Not a bug in our
operator -- `base` reproduces the deployed static FDK bit-for-bit (rel max 0.0e0).

NOTE ON THE METRIC BAND. p216's head ENDS at z = -84 mm (the series stops at the neck and
`_centre_fit` pads air below it), and that hard air/tissue face carries a -530 HU per-slice error
of its own in EVERY variant, `up70` included. It is the object's own boundary, not the artefact
under study, so the banding metric runs over `--band_mm` (default -75..-45, inside the anatomy)
and never averages the boundary slices in.

Reads nothing from the posterior loop and writes only under `--out`; imports fm3d read-only.

    CUDA_VISIBLE_DEVICES=1 python scripts/diag_cone_shift.py --out data/diag_cone_shift
    CUDA_VISIBLE_DEVICES=1 python scripts/diag_cone_shift.py --dz_mm 70 --nv_scale 2 \
        --crop_z_mm 100 --patients 1
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fm3d.dataset_cq500 import CQ500Generator
from fm3d.filters import DEFAULT_RAMP_WINDOW
from fm3d.geometry_3d import (ConeBeam3DConfig, detector_coords_3d, view_angular_weights)
from fm3d.leap_projector import ADJOINT_MODE
from fm3d.phantom import MU_WATER, hu_to_mu
from fm3d.projector_3d import fdk_conebeam_3d_batched, forward_project_3d_batched


def to_hu(mu: torch.Tensor) -> torch.Tensor:
    return (mu / MU_WATER - 1.0) * 1000.0


def z_translation(dz_mm: float, device) -> torch.Tensor:
    """T in SE(3) that moves a WORLD point up by dz along +z (the volume's slice axis)."""
    T = torch.eye(4, device=device, dtype=torch.float32)
    T[2, 3] = float(dz_mm)
    return T


def barrel_mask(shape, spacing, cfg: ConeBeam3DConfig, dz_mm: float, device="cpu"):
    """`geometry_3d.measured_region_mask` for an object translated UP by `dz_mm`.

    The barrel is fixed in the GANTRY frame, so in the OBJECT frame it moves DOWN by dz:
    a voxel at object z is measured iff z + dz is inside the barrel. Identical to
    `measured_region_mask` at dz = 0 (same expression, same voxel-centre convention).
    """
    D, H, W = shape
    dx, dy, dz = spacing
    R = 0.5 * cfg.fov_diameter_mm()
    half_v = 0.5 * cfg.nv * cfg.dv
    v_lo = cfg.det_offset_v_mm - half_v
    v_hi = cfg.det_offset_v_mm + half_v
    zs = (torch.arange(D, device=device, dtype=torch.float32) - (D - 1) / 2.0) * dz
    ys = (torch.arange(H, device=device, dtype=torch.float32) - (H - 1) / 2.0) * dy
    xs = (torch.arange(W, device=device, dtype=torch.float32) - (W - 1) / 2.0) * dx
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
    r = torch.sqrt(xx ** 2 + yy ** 2)
    mag = (cfg.SOD - r).clamp_min(1e-6) / cfg.SDD
    zg = zz + float(dz_mm)                                  # gantry-frame height
    return (r <= R) & (zg >= v_lo * mag) & (zg <= v_hi * mag)


def blur_v(sino: torch.Tensor, sigma_px: float) -> torch.Tensor:
    """Gaussian blur along the DETECTOR ROW axis v (dim -2) of (B,V,nv,nu), reflect-padded.

    The recon grid samples z at dz while the panel samples it at dv*SOD/SDD at the isocentre
    (1 mm vs 0.419 mm here), so v content above the recon's z Nyquist is aliased into z bands by
    the backprojection. sigma_px = dz * M / 2 band-limits v to that Nyquist.
    """
    if sigma_px <= 0:
        return sino
    r = max(1, int(round(3.0 * sigma_px)))
    x = torch.arange(-r, r + 1, device=sino.device, dtype=sino.dtype)
    k = torch.exp(-0.5 * (x / sigma_px) ** 2)
    k = k / k.sum()
    B, V, nv, nu = sino.shape
    s = sino.reshape(B * V, 1, nv, nu)
    s = torch.nn.functional.pad(s, (0, 0, r, r), mode="reflect")
    s = torch.nn.functional.conv2d(s, k.view(1, 1, -1, 1))
    return s.reshape(B, V, nv, nu)


def slice_bias(err_hu: torch.Tensor, mask: torch.Tensor, min_vox: int = 200,
               boxcar: int = 11):
    """Per-slice mean signed error in HU, and its high-pass (bias - boxcar(bias)).

    The banding IS a per-slice DC offset, so this is the artefact's own coordinate. The
    high-pass separates it from the FDK's smooth axial cupping, which is not what we are after.
    """
    D = err_hu.shape[0]
    bias = np.full(D, np.nan, dtype=np.float64)
    nvox = np.zeros(D, dtype=np.int64)
    for z in range(D):
        m = mask[z]
        n = int(m.sum())
        nvox[z] = n
        if n >= min_vox:
            bias[z] = float(err_hu[z][m].mean())
    filled = np.nan_to_num(bias)
    k = int(boxcar)
    pad = np.pad(filled, (k // 2, k // 2), mode="edge")
    smooth = np.convolve(pad, np.ones(k) / k, mode="valid")
    hp = bias - smooth
    return bias, hp, nvox


def panels_png(path, panels, title, *, z_slices, lo, hi, zoom=None, cmap="gray", dpi=170):
    """Rows = [axial @ z_probe, coronal (full z), sagittal (zoomed in z)]; cols = panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    za, zlo, zhi = z_slices
    n = len(panels)
    fig, ax = plt.subplots(3, n, figsize=(3.4 * n, 10.4), squeeze=False)
    for j, (vol, ttl) in enumerate(panels):
        D, H, W = vol.shape
        rows = [(vol[za], f"axial z={za}"),
                (vol[:, H // 2], "coronal"),
                (vol[zlo:zhi, :, W // 2], f"sagittal z[{zlo}:{zhi}]")]
        for i, (sl, nm) in enumerate(rows):
            a = ax[i][j]
            a.imshow(sl.detach().cpu().float().numpy(), cmap=cmap, vmin=lo, vmax=hi,
                     aspect="auto" if i else "equal", origin="lower")
            a.set_title(ttl if i == 0 else "", fontsize=9)
            a.set_xticks([]); a.set_yticks([])
            if j == 0:
                a.set_ylabel(nm, fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/CQ500")
    ap.add_argument("--out", default="data/diag_cone_shift")
    ap.add_argument("--split", default="val")
    ap.add_argument("--patients", type=int, default=1)
    ap.add_argument("--views", type=int, default=360)
    ap.add_argument("--shape", type=int, nargs=3, default=[256, 256, 256])
    ap.add_argument("--dz_mm", type=float, nargs="*", default=[-20.0, 35.0, 70.0],
                    help="object lift [mm], negative = lower (the control); object-frame "
                         "z = -dz lands on the cone midplane")
    ap.add_argument("--nv_scale", type=int, nargs="*", default=[2],
                    help="taller-panel ablation: det_nv *= k, same pitch, object unmoved")
    ap.add_argument("--crop_z_mm", type=float, nargs="*", default=[100.0],
                    help="zero the simulated object outside |z| <= c mm (deployed panel)")
    ap.add_argument("--v_blur_px", type=float, nargs="*", default=[1.2],
                    help="Gaussian sigma [detector px] along v before FDK; dz*M/2 = 1.2 px is "
                         "the recon grid's own z Nyquist")
    ap.add_argument("--band_mm", type=float, nargs=2, default=[-75.0, -45.0],
                    help="z window the banding metric is measured over; keep it INSIDE the "
                         "anatomy (the object's bottom face has a -530 HU error of its own)")
    ap.add_argument("--z_probe_mm", type=float, default=-70.0,
                    help="object-frame z the montage's axial slice and the zoom centre on")
    ap.add_argument("--zoom_mm", type=float, default=60.0)
    ap.add_argument("--window", default=None, help="ramp apodization; default = filters default")
    ap.add_argument("--brain_window", default="40,80")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    window = args.window or DEFAULT_RAMP_WINDOW
    D, H, W = (int(s) for s in args.shape)

    cfg = ConeBeam3DConfig.thies(n_views=args.views)
    gen = CQ500Generator(args.root, cfg, device=dev, shape=(D, H, W), split=args.split)
    svm = gen.sim_voxel_mm
    zs_mm = (np.arange(D) - (D - 1) / 2.0) * gen.dz          # object-frame z of each slice
    z_idx = lambda z_mm: int(round(z_mm / gen.dz + (D - 1) / 2.0))

    # ---- the variant table -------------------------------------------------------------
    # (name, dz_mm, nv_scale, crop_z_mm, v_blur_px); `base` first -- the difference reference.
    variants = [("base", 0.0, 1, None, 0.0)]
    variants += [(f"{'up' if dz > 0 else 'down'}{abs(dz):g}", float(dz), 1, None, 0.0)
                 for dz in args.dz_mm if dz != 0.0]
    variants += [(f"tall{k}", 0.0, int(k), None, 0.0) for k in args.nv_scale if int(k) != 1]
    variants += [(f"crop{c:g}", 0.0, 1, float(c), 0.0) for c in args.crop_z_mm]
    variants += [(f"vblur{s:g}", 0.0, 1, None, float(s)) for s in args.v_blur_px if s > 0]

    print(f"geometry: SOD {cfg.SOD} SDD {cfg.SDD} | {args.views} views | panel {cfg.nv}x{cfg.nu} "
          f"@ {cfg.du:g} mm | recon {D}x{H}x{W} @ {gen.dx:g} mm | ramp {window} | "
          f"adjoint {ADJOINT_MODE}")
    print(f"axial coverage on-axis {cfg.axial_coverage_mm():.1f} mm "
          f"(barrel half-height {0.5 * cfg.nv * cfg.dv * cfg.SOD / cfg.SDD:.1f} mm on axis, "
          f"{0.5 * cfg.nv * cfg.dv * (cfg.SOD - 60) / cfg.SDD:.1f} mm at r=60 mm) vs a "
          f"{D * gen.dx:.0f} mm box -> the box is TALLER than the cone")
    print(f"variants: {[v[0] for v in variants]}\n")

    all_rows, all_prof = [], {}
    for pi in range(args.patients):
        pid = gen.records[pi]["patient"]
        gt = gen.volume(pi)[0, 0]
        gt_hu = to_hu(gt)
        vol_fine = gen.volume_fine(pi)                       # (1,1,Ds,Hs,Ws), the sim object
        Ds = vol_fine.shape[2]
        zf_mm = (torch.arange(Ds, device=dev, dtype=torch.float32) - (Ds - 1) / 2.0) * svm

        recs, masks = {}, {}
        for name, dz, k, crop, vb in variants:
            cfg_x = cfg if k == 1 else dataclasses.replace(cfg, det_nv=cfg.det_nv * k)
            u_c, v_c = detector_coords_3d(cfg_x, device=dev)
            P = gen.P_nom @ z_translation(dz, dev)           # (V,3,4); exact identity at dz=0
            src = vol_fine
            if crop is not None:                             # air outside the kept z band
                src = vol_fine.clone()
                src[:, :, zf_mm.abs() > float(crop)] = 0.0
            with torch.no_grad():
                y = forward_project_3d_batched(src, P[None], u_c, v_c,
                                               dx=svm, dy=svm, dz=svm)
                y = blur_v(y, vb)
                rec = fdk_conebeam_3d_batched(
                    y, P[None], u_c, v_c, cfg_x, D=D, H=H, W=W,
                    dx=gen.dx, dy=gen.dy, dz=gen.dz, window=window, view_chunk=8,
                    view_weight=view_angular_weights(P[None]))[0]
            del y
            if crop is not None:
                del src
            recs[name] = rec
            masks[name] = barrel_mask((D, H, W), (gen.dx, gen.dy, gen.dz), cfg_x, dz,
                                      device=dev)
            print(f"  p{pid} {name:8s} dz {dz:+6.1f} mm  nv {cfg_x.nv:4d}  "
                  f"crop {'-' if crop is None else f'{crop:g} mm':>7s}  "
                  f"vblur {vb:.2f} px  measured {int(masks[name].sum()) / 1e6:.2f} M vox")
        del vol_fine
        torch.cuda.empty_cache()

        # SANITY: `base` must reproduce the deployed static FDK exactly (T = I, k = 1).
        with torch.no_grad():
            ref = gen.fdk(gen.simulate(pi, gen.P_nom[None]), gen.P_nom[None],
                          window=window)[0]
        rel = float((recs["base"] - ref).abs().max() / ref.abs().max())
        print(f"  [check] base vs gen.fdk(gen.simulate(...)): rel max {rel:.2e} "
              f"({'OK' if rel < 1e-6 else 'MISMATCH'})")
        del ref

        # ---- metrics. Two masks: each variant's OWN measured barrel, and the COMMON one
        # (their intersection) -- the only set on which the numbers are a fair comparison,
        # because lifting the object takes coverage away from the top of the head.
        common = masks["base"].clone()
        for m in masks.values():
            common &= m
        brain_gt = (gt > hu_to_mu(10.0)) & (gt < hu_to_mu(60.0))
        brain_c = common & brain_gt
        print(f"  common measured region {int(common.sum()) / 1e6:.2f} M vox "
              f"(brain {int(brain_c.sum()) / 1e6:.2f} M), z in "
              f"[{zs_mm[common.any(-1).any(-1).cpu().numpy()].min():+.1f}, "
              f"{zs_mm[common.any(-1).any(-1).cpu().numpy()].max():+.1f}] mm")

        band = (zs_mm >= min(args.band_mm)) & (zs_mm <= max(args.band_mm))
        mid = np.abs(zs_mm) <= 40.0
        print(f"  banding metric over z in [{min(args.band_mm):+.0f}, {max(args.band_mm):+.0f}] "
              f"mm ({int(band.sum())} slices), reference band |z| <= 40 mm")
        prof = {}
        for name, dz, k, crop, vb in variants:
            err = to_hu(recs[name]) - gt_hu
            bias, hp, nvox = slice_bias(err, brain_c)
            prof[name] = dict(bias=bias, hp=hp, nvox=nvox)
            hp_band = float(np.sqrt(np.nanmean(hp[band] ** 2)))
            hp_p2p = float(np.nanmax(hp[band]) - np.nanmin(hp[band]))
            hp_mid = float(np.sqrt(np.nanmean(hp[mid] ** 2)))
            bias_band = float(np.nanmean(bias[band]))
            bm = brain_c & torch.as_tensor(band, device=dev)[:, None, None]
            rmse_b = float(err[bm].pow(2).mean().sqrt())
            rmse_c = float(err[brain_c].pow(2).mean().sqrt())
            print(f"  p{pid} {name:8s}  HP-RMS {hp_band:6.2f} HU  HP p2p {hp_p2p:6.2f}  "
                  f"(mid {hp_mid:5.2f})  |  bias {bias_band:+8.2f} HU  |  brain RMSE band "
                  f"{rmse_b:6.2f}  all {rmse_c:6.2f} HU")
            all_rows.append(dict(patient=pid, variant=name, dz_mm=dz, nv_scale=k,
                                 crop_z_mm=crop, v_blur_px=vb, hp_rms_band=hp_band,
                                 hp_p2p_band=hp_p2p, hp_rms_mid=hp_mid, bias_band=bias_band,
                                 rmse_brain_band=rmse_b, rmse_brain_common=rmse_c))
        all_prof[pid] = prof

        # The z dependence of the defect in the DEPLOYED geometry, on the record: the cone-angle
        # hypothesis predicts it grows monotonically with |z|, and this is the curve to quote.
        print(f"  p{pid} base, per 20 mm band (common brain mask):")
        for zlo in range(-120, 120, 20):
            sel = (zs_mm >= zlo) & (zs_mm < zlo + 20) & ~np.isnan(prof["base"]["bias"])
            if sel.sum():
                print(f"    z {zlo:+5d}..{zlo + 20:+5d}  bias "
                      f"{np.nanmean(prof['base']['bias'][sel]):+8.2f}  HP-RMS "
                      f"{np.sqrt(np.nanmean(prof['base']['hp'][sel] ** 2)):6.2f} HU  "
                      f"({int(sel.sum())} slices)")

        # ---- figures ---------------------------------------------------------------------
        bc, bw = (float(t) for t in args.brain_window.split(","))
        za = max(0, min(D - 1, z_idx(args.z_probe_mm)))
        zlo = max(0, z_idx(args.z_probe_mm - args.zoom_mm))
        zhi = min(D, z_idx(args.z_probe_mm + args.zoom_mm))
        zsl = (za, zlo, zhi)
        panels_png(os.path.join(args.out, f"cone_p{pi}_recon_brain.png"),
                   [(gt, "GT volume")] + [(recs[n], n) for n, *_ in variants],
                   f"CQ500 {args.split} p{pid} | STATIC FDK, geometry-shift probe | BRAIN "
                   f"{bc:g}/{bw:g} HU | ramp {window} | axial slice at z="
                   f"{zs_mm[za]:+.0f} mm",
                   z_slices=zsl, lo=hu_to_mu(bc - 0.5 * bw), hi=hu_to_mu(bc + 0.5 * bw))
        panels_png(os.path.join(args.out, f"cone_p{pi}_err.png"),
                   [(to_hu(recs[n]) - gt_hu, f"{n} - GT") for n, *_ in variants],
                   f"CQ500 p{pid} | SIGNED ERROR vs GT [HU], +-150 HU | the banding is a "
                   f"per-slice DC offset, so it reads directly here",
                   z_slices=zsl, lo=-150.0, hi=150.0, cmap="RdBu_r")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axx = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for n, *_ in variants:
            axx[0].plot(zs_mm, prof[n]["bias"], lw=1.3, label=n)
            axx[1].plot(zs_mm, prof[n]["hp"], lw=1.3, label=n)
        # The object's bottom face (z ~ -84 mm) errs by -530 HU in every variant and would
        # squash the artefact under study flat, so the y range is taken from z >= band-5 only.
        keep = zs_mm >= min(args.band_mm)
        for a, key, t in zip(axx, ["bias", "hp"],
                             ["mean signed error in brain voxels [HU]",
                              "high-passed (11-slice boxcar) = THE BANDING [HU]"]):
            lim = max(np.nanmax(np.abs(prof[n][key][keep])) for n, *_ in variants)
            a.set_ylim(-1.15 * lim, 1.15 * lim)
            a.axhline(0, color="k", lw=0.6)
            a.axvspan(min(args.band_mm), max(args.band_mm), color="orange", alpha=0.15)
            a.set_ylabel(t, fontsize=9)
            a.grid(alpha=0.3)
            a.legend(fontsize=8, ncol=len(variants) + 1)
        axx[1].set_xlabel("object-frame z [mm]  (0 = volume centre = cone midplane at dz=0)")
        axx[0].set_title(f"p{pid}: axial error profile, common brain mask (shaded = metric band; "
                         f"y clipped -- the object's bottom face runs off scale in EVERY variant)",
                         fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"cone_p{pi}_zprofile.png"), dpi=120)
        plt.close(fig)

        raw_dir = os.path.join(args.out, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        for nm, volm in [("gt", gt)] + [(n, recs[n]) for n, *_ in variants]:
            to_hu(volm).detach().cpu().numpy().astype("<f4").tofile(
                os.path.join(raw_dir, f"p{pid}_{nm}_{W}x{H}x{D}_float32_HU.raw"))
        torch.save({n: recs[n].half().cpu() for n, *_ in variants} |
                   {"gt": gt.half().cpu(), "patient": pid},
                   os.path.join(args.out, f"cone_p{pi}.pt"))
        del recs, masks, common
        torch.cuda.empty_cache()

    torch.save({"rows": all_rows, "profiles": all_prof, "args": vars(args),
                "window": window, "adjoint_mode": ADJOINT_MODE,
                "sim_voxel_mm": svm, "sim_shape": gen.sim_shape_dhw},
               os.path.join(args.out, "summary.pt"))
    print(f"\n-> {args.out}/  (cone_p*_recon_brain.png, _err.png, _zprofile.png, raw/, *.pt)")


if __name__ == "__main__":
    main()
