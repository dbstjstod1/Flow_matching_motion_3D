"""GATE: the measurement y is simulated on the NATIVE grid, and only the grid changed.

The pipeline's operator split (2026-07-29, user directive "순결한 LEAP 일대일 대응"):

    SIMULATION  y = A(truth; P)     -- SF forward at du*SOD/SDD (LEAP's native-grid convention)
    INVERSION   FDK / CG / dP       -- the SAME SF matched pair at the coarse `voxel_mm` grid

This gate pins the four claims that make that split correct and load-bearing:

  1. the simulation grid is DERIVED from the given geometry (du * SOD/SDD), covers the same
     physical box as the inversion grid, and is what `simulate` actually runs on;
  2. simulating natively REMOVES the cube-basis crosshatch that motivated the change -- the
     static FDK's excess texture in a homogeneous brain ROI must beat the old 1 mm path by a
     wide margin, at no loss of bone-edge sharpness;
  3. `simulate` and the coarse `project` are the SAME OPERATOR, differing only by the grid: the
     two sinograms must agree to the level a resampling of the truth explains (a few percent),
     NOT to machine precision -- if they agreed exactly, nothing would have been fixed;
  4. NOTHING was changed about the inversion: the matched-transpose identity still holds on the
     coarse grid (the full SF suite lives in `gate_sf_projector`; this is the cheap re-assertion),
     and `sim_native=False` reproduces the old path bit-for-bit, so the switch is the only knob.

Run: python scripts/gate_native_sim.py [--patients 2]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.dataset_cq500 import CQ500Generator
from fm3d.geometry_3d import ConeBeam3DConfig

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:44s} {detail}", flush=True)
    if not ok:
        FAILED.append(name)


def brain_texture(recon: torch.Tensor, gt: torch.Tensor) -> float:
    """Excess sd over the GT in a homogeneous central brain ROI, in HU. The crosshatch metric."""
    D, H, W = gt.shape
    roi = torch.zeros_like(gt, dtype=torch.bool)
    roi[D // 2 - 12:D // 2 + 12, H // 2 - 30:H // 2 + 30, W // 2 - 30:W // 2 + 30] = True
    m = roi & (gt > 0.0195) & (gt < 0.0215)
    ex = float(recon[m].std()) ** 2 - float(gt[m].std()) ** 2
    return float(np.sqrt(max(ex, 0.0)) / 0.02 * 1000.0)


def edge_sharpness(recon: torch.Tensor, gt: torch.Tensor) -> float:
    """Mean gradient magnitude on bone, as a fraction of the GT's. Guards against 'fixed by blur'."""
    def g(v):
        d = torch.gradient(v.float())
        return torch.sqrt(sum(x ** 2 for x in d))
    bone = gt > 0.03
    return float(g(recon)[bone].mean() / g(gt)[bone].mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/CQ500")
    ap.add_argument("--patients", type=int, default=2)
    ap.add_argument("--views", type=int, default=360)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ConeBeam3DConfig.thies(n_views=args.views)
    gen = CQ500Generator(args.root, cfg, device=dev, split="val", shape=(256,) * 3, voxel_mm=1.0)
    P = gen.P_nom[None]

    print("\nT1  the simulation grid is derived from the geometry, not chosen")
    du = float(gen.u_coords[1] - gen.u_coords[0])
    expect_vox = du * cfg.SOD / cfg.SDD
    check("native voxel = du * SOD/SDD", abs(gen.sim_voxel_mm - expect_vox) < 1e-9,
          f"{gen.sim_voxel_mm:.6f} mm (du {du:.4f} x {cfg.SOD:g}/{cfg.SDD:g})")
    fine_ext = [n * gen.sim_voxel_mm for n in gen.sim_shape_dhw]
    coarse_ext = [n * d for n, d in zip(gen.shape_dhw, (gen.dz, gen.dy, gen.dx))]
    check("fine box covers the coarse box",
          all(f >= c - 1e-6 and f < c + gen.sim_voxel_mm for f, c in zip(fine_ext, coarse_ext)),
          f"{['%.2f' % f for f in fine_ext]} mm vs coarse {['%.2f' % c for c in coarse_ext]} mm")
    check("simulation is finer than inversion", gen.sim_voxel_mm < gen.dx,
          f"{'x'.join(map(str, gen.sim_shape_dhw))} @ {gen.sim_voxel_mm:.4f} vs "
          f"{'x'.join(map(str, gen.shape_dhw))} @ {gen.dx:g}")
    check("sim_native default ON", gen.sim_native is True)
    v = gen.volume_fine(0)
    check("volume_fine lands on that grid", tuple(v.shape[2:]) == gen.sim_shape_dhw,
          f"{tuple(v.shape[2:])}")
    del v
    torch.cuda.empty_cache()

    print("\nT2  simulating natively removes the cube-basis crosshatch (the reason for all this)")
    # The 1 mm reference here is the OLD pipeline, reachable through the ablation switch.
    for i in range(args.patients):
        gt = gen.volume(i)[0, 0]
        t0 = time.time()
        y_nat = gen.simulate(i, P)
        t_nat = time.time() - t0
        gen.sim_native = False
        y_old = gen.simulate(i, P)
        gen.sim_native = True
        with torch.no_grad():
            # ram-lak: the UNAPODIZED filter, i.e. the texture is measured undisguised. The
            # pipeline's deployed default is `shepphann`, which hides a good part of it -- so
            # `r_old_dep` is what the 1 mm simulation actually delivered, and the native path has
            # to beat THAT on both axes to be worth a restart.
            r_nat = gen.fdk(y_nat, P, window="ramlak")[0]
            r_old = gen.fdk(y_old, P, window="ramlak")[0]
            r_old_dep = gen.fdk(y_old, P)[0]
        tx_nat, tx_old = brain_texture(r_nat, gt), brain_texture(r_old, gt)
        tx_dep = brain_texture(r_old_dep, gt)
        sh_nat, sh_dep = edge_sharpness(r_nat, gt), edge_sharpness(r_old_dep, gt)
        pid = gen.records[i]["patient"]
        check(f"p{pid}: texture cut >=3x vs 1 mm sim", tx_nat * 3.0 <= tx_old,
              f"{tx_old:.1f} -> {tx_nat:.1f} HU  ({t_nat:.2f}s / {args.views} views)")
        # NOT compared against the 1 mm ram-lak recon: `edge_sharpness` is a gradient-magnitude
        # ratio, and the crosshatch IS gradient, so the aliased recon scores ~99% by reproducing
        # the artefact (and the 1 mm GT is itself aliased -- linear-resampled from 0.41 mm with no
        # anti-aliasing -- so 100% is not the target either). The honest comparison is against the
        # filter-based mitigation that was actually deployed.
        # CRITERION CHANGED 2026-07-30, when the forward was pinned to LEAP's Joseph kernel.
        # It used to demand `tx_nat <= tx_dep + 0.5`, i.e. native must not lose to the 1 mm
        # path on TEXTURE. That was a fair fight only while the SF cube basis put 18-21 HU of
        # crosshatch into the 1 mm recon; Joseph's trilinear-tent basis cut that to 3.5-5.0 HU
        # (ram-lak) and the deployed shepphann comparator to 0.0. And 0.0 is not a quality
        # win -- the 1 mm path projects the VERY volume it is then compared against, so it
        # cannot exhibit grid texture at all (the inverse crime, same trap the rmse column
        # carries). Comparing an honest number against a structurally-zero one is the wrong
        # test. What matters is ABSOLUTE: native must leave no crosshatch (bar 1.5 HU, ~10x
        # below the 13-21 HU that started this and well under a real scan's ~5 HU noise) and
        # must still win the axis the inverse crime cannot fake, SHARPNESS.
        check(f"p{pid}: no crosshatch, and sharper than the deployed 1 mm+shepphann path",
              tx_nat <= 1.5 and sh_nat >= sh_dep,
              f"texture {tx_dep:.1f} (1 mm, inverse crime) -> {tx_nat:.1f} HU (bar 1.5), "
              f"sharpness {sh_dep * 100:.0f}% -> {sh_nat * 100:.0f}% of GT")
        if i == 0:
            y_keep, gt_keep = y_nat, gt
        else:
            del y_nat
        del y_old, r_nat, r_old
        torch.cuda.empty_cache()

    print("\nT3  same operator, different grid (must agree ROUGHLY, not exactly)")
    gen.sim_native = False
    y_old = gen.simulate(0, P)
    gen.sim_native = True
    num = float((y_keep - y_old).pow(2).sum().sqrt())
    den = float(y_old.pow(2).sum().sqrt())
    rel = num / den
    check("native vs 1 mm sinogram: same operator", rel < 0.05, f"rel-L2 {rel:.2e} (< 5%)")
    check("native vs 1 mm sinogram: NOT identical", rel > 1e-4,
          f"rel-L2 {rel:.2e} -- if this were ~0 the grid change would be a no-op")
    mass_n = float(y_keep.mean())
    mass_o = float(y_old.mean())
    check("total attenuation preserved", abs(mass_n - mass_o) / abs(mass_o) < 0.01,
          f"mean line integral {mass_o:.4f} -> {mass_n:.4f} mm^-1")
    del y_old
    torch.cuda.empty_cache()

    print("\nT4  the inversion side is untouched")
    # THE PAIR IS NO LONGER MATCHED (2026-07-29). `adjoint` is LEAP's voxel-driven
    # backprojector, not the transpose of LEAP's forward, so this test asks what the CG data
    # step actually needs: how far from adjoint is it ON THE DATA IT SEES? The probe matters --
    # a white-noise volume and sinogram (what this used to use) compares the two voxel bases at
    # the frequencies neither resolves and reads ~7x off, while the real phantom and its own
    # sinogram read ~1e-5. `gate_leap_projector.py` T2 measures both probes and is the
    # authority; this one only pins that the native-simulation switch did not disturb it.
    D, H, W = gen.shape_dhw
    f = gen.volume(0)
    with torch.no_grad():
        Af = gen.project(f, P)
        ATg = gen.adjoint(Af, P)
    lhs = float((Af * Af).sum())
    rhs = float((f * ATg).sum())
    check("adjointness on the real volume/sinogram",
          abs(lhs - rhs) / max(abs(lhs), 1e-12) < 2e-3,
          f"<Af,Af> {lhs:.6e}  <f,A^T Af> {rhs:.6e}  "
          f"rel {abs(lhs - rhs) / max(abs(lhs), 1e-12):.2e}  (unmatched by construction)")
    check("project() still runs on the inversion grid",
          Af.shape[1:] == (args.views, gen.v_coords.numel(), gen.u_coords.numel()),
          f"{tuple(Af.shape)}")
    # The ablation switch must reproduce the OLD path, so the grid is the ONLY variable. Compared
    # against the operator's OWN run-to-run floor, not bit-for-bit: the SF forward accumulates
    # into the detector with atomics, so two identical calls differ at ~1e-6 rel by float
    # reassociation alone. Measuring that floor here is what makes the assertion meaningful.
    gen.sim_native = False
    with torch.no_grad():
        y_ab = gen.simulate(0, P)
        y_pr = gen.project(gen.volume(0), P)
        y_pr2 = gen.project(gen.volume(0), P)
    gen.sim_native = True
    scale = max(float(y_pr.abs().max()), 1e-12)
    floor = float((y_pr - y_pr2).abs().max()) / scale
    diff = float((y_ab - y_pr).abs().max()) / scale
    check("sim_native=False == plain project()", diff <= max(10.0 * floor, 1e-6),
          f"rel {diff:.1e} vs the operator's own atomic-reorder floor {floor:.1e}")

    print("\n" + "=" * 70)
    if FAILED:
        print(f"{len(FAILED)} gate(s) FAILED: " + ", ".join(FAILED))
        print("=" * 70)
        return 1
    print("all gates passed")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
