"""LEAP (LLNL leapct) one-to-one cross-check of OUR operator pair `fm3d/triton_sf.py`.

WHY (user, 2026-07-29): the repo claims a "LEAP-class" separable-footprint matched pair. This
script closes the loop against the REAL toolkit installed in the `flow_matching` env
(site-packages/leapctype.py + libleapct.so, built 2026-07-28 from source): same volume, same
geometry, their kernels vs ours -- model, numbers, and wall clock.

CODE-LEVEL CORRESPONDENCE (read out of leapctype.py / the .so symbol table before writing this):

  * FORWARD MODEL. `set_projector` docstring: "all forward projectors use the modified
    separable footprint model; this function only changes the BACKprojection model" ('SF'
    matched-class / 'VD' voxel-driven, "faster, but less accurate"). So LEAP = SF forward +
    {SF | VD} backward; ours = SF forward + its EXACT transpose. Same class, and ours is the
    stricter choice.
  * GEOMETRY. LEAP has two ways to express our per-view `P`:
      - `set_conebeam(...phis, sod, sdd)` -- a CIRCULAR ORBIT parameterized by one angle per
        view. It CANNOT express rigid patient motion (6 DoF per view). Kernels: project_SF /
        backproject_SF (+ _eSF, _vox).
      - `set_modularbeam(sourcePositions, moduleCenters, rowVectors, colVectors)` -- arbitrary
        per-view source + detector pose, which IS our motion-applied `P`. Kernels:
        modularBeamProjectorKernel_SF / ..._Joseph_... The SF path is taken only when
        `modularbeamIsAxiallyAligned` (symbol present in the .so); a general 6-DoF motion
        breaks that and LEAP silently falls back to a JOSEPH ray-driven projector, i.e. a
        DIFFERENT operator model. This script measures both regimes.
  * VOLUME. Cell-centred, square in-plane `voxelWidth` + separate `voxelHeight`, default array
    order ZYX == our (D,H,W), origin at the centre of rotation == our centred box.
  * DETECTOR. Flat panel, (numAngles, numRows, numCols) == our (V, nv, nu).
  * GEOMETRY GRADIENT. `leaptorch.ProjectorFunctionGPU.backward` returns `(vol, None, None,
    None)`: the volume gradient only. The geometry lives in C++ `parameters*` state, not in a
    tensor, so d loss/dP does not exist in LEAP at all. That is why `triton_sf` was written.

WHAT THIS RUNS (all on the GPU, deploy scale = thies 360 x 500 x 700 @ 0.64 mm, 256^3 @ 1 mm):
  [1] CONVENTION LOCK, analytically: decompose our P into (source, detector-centre, u/v axes)
      and compare against LEAP's OWN readback (`convert_to_modularbeam` + get_sourcePositions /
      get_moduleCenters / get_rowVectors / get_colVectors). No correlation search, no guessing.
  [2] NOMINAL A/B: forward + backprojection, ours vs LEAP cone-SF vs LEAP cone-VD vs LEAP
      modular, with per-pair adjointness <Af,g> = <f,A^T g> and ms/view.
  [3] MOTION A/B: P_mot = params_to_Pmot(akima 5mm/5deg p2p) -- LEAP modular is the only path
      that can hold it; plus an in-plane-only motion, which keeps LEAP on its SF kernel.
  [4] RECON: our FDK(shepphann) vs LEAP fbp(rampFilter 2 + FBPlowpass 2.0) on the same
      sinogram, in the measured barrel.

    CUDA_VISIBLE_DEVICES=1 python scripts/diag_leap_crosscheck.py [--native]

Outputs -> data/diag/leap_crosscheck/{summary.json, sino_compare.png, recon_compare.png}.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.geometry_3d import (ConeBeam3DConfig, build_conebeam_orbit, detector_coords_3d,
                              measured_region_mask)
from fm3d.projector_3d import (forward_project_3d_batched, adjoint_project_3d_batched,
                               fdk_conebeam_3d_batched)
from fm3d.rigid_motion import akima_motion, params_to_Pmot
from fm3d.phantom import head_phantom

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "data", "diag", "leap_crosscheck")


# --------------------------------------------------------------------------------------- utils
def tsync():
    torch.cuda.synchronize()
    return time.time()


def corr(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def decompose_P(P: torch.Tensor):
    """(V,3,4) -> (C, e_u, e_v, e_n, sdd) in WORLD mm, exactly as the SF kernel reads them.

    P = K [R | -R C] with K = diag(SDD, SDD, 1) and principal point (0,0) (geometry_3d), so
    M := P[:, :3] has rows (SDD*e_u, SDD*e_v, e_n) and P[:, 3] = -M C.  Rigid motion
    right-multiplies an SE(3) into P and preserves that form, so this decomposition is valid
    for the MOTION-APPLIED matrices too -- which is the whole point.
    """
    Pm = P.detach().double().cpu()
    M, p4 = Pm[:, :, :3], Pm[:, :, 3]
    C = torch.linalg.solve(M, -p4[..., None])[..., 0]              # source (V,3)
    sdd = M[:, 0].norm(dim=-1)                                     # (V,)
    e_u = M[:, 0] / sdd[:, None]
    e_v = M[:, 1] / sdd[:, None]
    e_n = M[:, 2] / M[:, 2].norm(dim=-1, keepdim=True)
    return C, e_u, e_v, e_n, sdd


def modular_from_P(P: torch.Tensor, u_coords: torch.Tensor, v_coords: torch.Tensor):
    """Our P -> LEAP `set_modularbeam` arrays (float32 numpy, (V,3) each).

    moduleCenter = the world position of the CENTRE of the detector array, i.e. detector
    coordinate (u, v) = (u_coords centre, v_coords centre) -- which is the panel offset, NOT
    the central ray, when the panel is offset (half-fan).
    """
    C, e_u, e_v, e_n, sdd = decompose_P(P)
    u_c = 0.5 * float(u_coords[0] + u_coords[-1])
    v_c = 0.5 * float(v_coords[0] + v_coords[-1])
    mod = C + sdd[:, None] * e_n + u_c * e_u + v_c * e_v
    f32 = lambda t: np.ascontiguousarray(t.numpy().astype(np.float32))
    return f32(C), f32(mod), f32(e_v), f32(e_u)      # rowVectors = e_v, colVectors = e_u


def leap_set_modular(leap, P, u, v, D, H, W, dx, dz):
    src, mod, rowv, colv = modular_from_P(P, u, v)
    du = float(u[1] - u[0])
    dv = float(v[1] - v[0])
    ok = leap.set_modularbeam(P.shape[0], len(v), len(u), dv, du, src, mod, rowv, colv)
    assert ok, "set_modularbeam rejected the geometry"
    leap.set_volume(W, H, D, dx, dz)
    leap.set_diameterFOV(1.0e5)          # kill LEAP's cylindrical mask; our SF has none
    return du, dv


def leap_set_cone(leap, cfg, phis, D, H, W, dx, dz, u, v):
    du, dv = float(u[1] - u[0]), float(v[1] - v[0])
    cC = (cfg.nu - 1) / 2.0 - float(cfg.det_offset_u_mm) / du
    cR = (cfg.nv - 1) / 2.0 - float(cfg.det_offset_v_mm) / dv
    ok = leap.set_conebeam(cfg.n_views, cfg.nv, cfg.nu, dv, du, cR, cC,
                           np.ascontiguousarray(phis.astype(np.float32)), cfg.SOD, cfg.SDD)
    assert ok, "set_conebeam rejected the geometry"
    leap.set_volume(W, H, D, dx, dz)
    leap.set_diameterFOV(1.0e5)
    return cC, cR


def bench(fn, n=1):
    fn()                                   # warm-up (JIT, LEAP's internal allocations)
    t0 = tsync()
    for _ in range(n):
        out = fn()
    dt = (tsync() - t0) / n
    return out, dt


# ----------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", type=int, default=360)
    ap.add_argument("--vox", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=256, help="volume side (D=H=W)")
    ap.add_argument("--native", action="store_true",
                    help="also bench the native simulation grid (du*SOD/SDD)")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda"
    res = {}

    from leapctype import tomographicModels
    leap = tomographicModels()
    leap.set_gpu(args.gpu)
    ver = leap.version()
    print(f"LEAP version {ver} | torch {torch.__version__} | "
          f"{torch.cuda.get_device_name(0)}", flush=True)
    res["leap_version"] = ver

    cfg = ConeBeam3DConfig.thies(n_views=args.views)
    P = build_conebeam_orbit(cfg, device=dev)
    u, v = detector_coords_3d(cfg, device=dev)
    du, dv = float(u[1] - u[0]), float(v[1] - v[0])
    N = args.n
    D = H = W = N
    dx = dy = dz = args.vox
    print(f"geometry: {cfg.describe()}\nvolume: {D}x{H}x{W} @ {dx} mm", flush=True)
    res["config"] = dict(views=cfg.n_views, nu=cfg.nu, nv=cfg.nv, du=du, dv=dv,
                         SOD=cfg.SOD, SDD=cfg.SDD, vol=[D, H, W], vox=dx)

    vol = head_phantom((D, H, W), (dz, dy, dx), device=dev)[None, None]      # (1,1,D,H,W)
    f_t = vol[0, 0].contiguous()

    # ================================================================== [1] CONVENTION LOCK ====
    # Ask LEAP what IT thinks the geometry is, in world coordinates, and compare to our own
    # decomposition of P. Both sides are then in the same units -- no image correlation needed.
    C_o, eu_o, ev_o, en_o, sdd_o = decompose_P(P)
    betas = torch.atan2(C_o[:, 1], C_o[:, 0]).numpy()
    phis_deg = np.degrees(np.unwrap(betas))
    best = None
    for sgn in (+1, -1):
        for off in (0.0, 90.0, 180.0, 270.0):
            phis = (sgn * phis_deg + off).astype(np.float32)
            leap_set_cone(leap, cfg, phis, D, H, W, dx, dz, u, v)
            leap.convert_to_modularbeam()
            s_l = leap.get_sourcePositions()
            m_l = leap.get_moduleCenters()
            r_l = leap.get_rowVectors()
            c_l = leap.get_colVectors()
            src, mod, rowv, colv = modular_from_P(P, u, v)
            e = max(float(np.abs(s_l - src).max()), float(np.abs(m_l - mod).max()),
                    float(np.abs(r_l - rowv).max()) * cfg.SOD,
                    float(np.abs(c_l - colv).max()) * cfg.SOD)
            if best is None or e < best[0]:
                best = (e, sgn, off, (s_l, m_l, r_l, c_l))
    err, sgn, off, (s_l, m_l, r_l, c_l) = best
    print(f"[1] convention: LEAP phis = {sgn:+d} * deg(atan2(Cy,Cx)) + {off:g}  ->  worst "
          f"disagreement over source / module-centre / axes: {err:.3e} mm", flush=True)
    src, mod, rowv, colv = modular_from_P(P, u, v)
    print(f"    source     ours {src[0]}  LEAP {s_l[0]}")
    print(f"    moduleCtr  ours {mod[0]}  LEAP {m_l[0]}")
    print(f"    rowVec     ours {rowv[0]}  LEAP {r_l[0]}")
    print(f"    colVec     ours {colv[0]}  LEAP {c_l[0]}", flush=True)
    res["convention"] = dict(sign=sgn, offset_deg=off, worst_mm=err)
    assert err < 1e-2, "geometry conventions do not line up -- everything below is meaningless"
    phis = (sgn * phis_deg + off).astype(np.float32)
    # LEAP'S PARAMETER OBJECT LEAKS STATE ACROSS GEOMETRY CHANGES. After the
    # `convert_to_modularbeam()` calls above, a later `set_conebeam()` restores the
    # projector but NOT the fbp path: measured on this very geometry, fbp-vs-our-FDK went
    # 3.1e-2 (fresh object) -> 2.9e-1 (same object, post-convert). Every measurement below
    # therefore runs on a FRESH instance. Do not reuse one across geometry types.
    leap = tomographicModels()
    leap.set_gpu(args.gpu)

    # =============================================================== [2] NOMINAL FORWARD/BACK ==
    print("\n[2] NOMINAL geometry (circular orbit, theta = 0)", flush=True)
    fw, bw, adj, sino, bp = {}, {}, {}, {}, {}

    with torch.no_grad():
        y_ours, t = bench(lambda: forward_project_3d_batched(
            vol, P[None], u, v, dx=dx, dy=dy, dz=dz)[0])
    sino["ours_SF"], fw["ours_SF"] = y_ours, t

    g_t = torch.zeros(cfg.n_views, cfg.nv, cfg.nu, device=dev, dtype=torch.float32)

    def leap_fwd():
        g_t.zero_()
        leap.project_gpu(g_t, f_t)
        return g_t

    leap_set_cone(leap, cfg, phis, D, H, W, dx, dz, u, v)
    _, t = bench(leap_fwd)
    sino["leap_cone_SF"], fw["leap_cone_SF"] = g_t.clone(), t

    leap_set_modular(leap, P, u, v, D, H, W, dx, dz)
    _, t = bench(leap_fwd)
    sino["leap_modular"], fw["leap_modular"] = g_t.clone(), t

    ref = sino["ours_SF"]
    print("    forward:", flush=True)
    for k, s in sino.items():
        rel = float((s - ref).norm() / ref.norm())
        print(f"      {k:14s} {1e3 * fw[k] / cfg.n_views:7.2f} ms/view  ({fw[k]:6.3f} s)  "
              f"rel-L2 vs ours {rel:.4e}  corr {corr(s.cpu().numpy(), ref.cpu().numpy()):.6f}",
              flush=True)
        res[f"fwd_{k}"] = dict(sec=fw[k], ms_per_view=1e3 * fw[k] / cfg.n_views, rel=rel)
    # LEAP against ITSELF: how much of the modular gap is LEAP's own cone-vs-modular kernel
    # difference rather than a disagreement with us? (modular takes a different code path and
    # only uses the SF kernel when `modularbeamIsAxiallyAligned`.)
    rel_ll = float((sino["leap_modular"] - sino["leap_cone_SF"]).norm()
                   / sino["leap_cone_SF"].norm())
    print(f"      [LEAP internal] modular vs cone-SF: rel-L2 {rel_ll:.4e}", flush=True)
    res["leap_modular_vs_cone"] = rel_ll

    # ---- backprojection of the SAME sinogram (ours), + each operator's own adjointness -------
    s_t = ref.contiguous()
    with torch.no_grad():
        b_ours, t = bench(lambda: adjoint_project_3d_batched(
            s_t[None], P[None], u, v, D=D, H=H, W=W, dx=dx, dy=dy, dz=dz)[0, 0])
    bp["ours_SF"], bw["ours_SF"] = b_ours, t
    adj["ours_SF"] = abs(float((sino["ours_SF"] * s_t).sum())
                         - float((f_t * b_ours).sum())) / abs(float((sino["ours_SF"] * s_t).sum()))

    bpv = torch.zeros_like(f_t)

    def leap_bp():
        bpv.zero_()
        leap.backproject_gpu(s_t, bpv)
        return bpv

    for geom, which in (("cone", "SF"), ("cone", "VD"), ("modular", "SF")):
        if geom == "cone":
            leap_set_cone(leap, cfg, phis, D, H, W, dx, dz, u, v)
        else:
            leap_set_modular(leap, P, u, v, D, H, W, dx, dz)
        leap.set_projector(which)
        _, t = bench(leap_bp)
        key = f"leap_{geom}_{which}"
        bp[key], bw[key] = bpv.clone(), t
        # own adjointness: their forward against their backward
        g_t.zero_()
        leap.project_gpu(g_t, f_t)
        lhs = float((g_t * s_t).sum())
        adj[key] = abs(lhs - float((f_t * bp[key]).sum())) / abs(lhs)
    leap.set_projector("SF")

    print("    backprojection of OUR sinogram:", flush=True)
    rb = bp["ours_SF"].flatten()
    for k in bp:
        o = bp[k].flatten()
        cs = float((o @ rb) / (o.norm() * rb.norm()))
        print(f"      {k:16s} {1e3 * bw[k] / cfg.n_views:7.2f} ms/view  ({bw[k]:6.3f} s)  "
              f"cos vs ours {cs:.6f}  norm ratio {float(o.norm() / rb.norm()):.4f}  "
              f"own adjointness {adj[k]:.2e}", flush=True)
        res[f"bwd_{k}"] = dict(sec=bw[k], ms_per_view=1e3 * bw[k] / cfg.n_views, cos=cs,
                               norm_ratio=float(o.norm() / rb.norm()), adjointness=adj[k])

    # =========================================================================== [3] MOTION ====
    print("\n[3] MOTION (per-view rigid, LEAP modular-beam is the only representation)",
          flush=True)
    res["motion"] = {}
    for name, th in (("akima_5mm5deg_p2p",
                      akima_motion(cfg.n_views, trans_mm=5.0, rot_deg=5.0, seed=0,
                                   device=dev)),
                     ("inplane_only", None)):
        if th is None:                       # tx, ty, wz only -> LEAP stays axially aligned
            th = akima_motion(cfg.n_views, trans_mm=5.0, rot_deg=5.0, seed=1, device=dev)
            th = th * torch.tensor([1., 1., 0., 0., 0., 1.], device=dev)
        Pm = params_to_Pmot(th, P).detach()
        with torch.no_grad():
            y_o, t_o = bench(lambda: forward_project_3d_batched(
                vol, Pm[None], u, v, dx=dx, dy=dy, dz=dz)[0])
        leap_set_modular(leap, Pm, u, v, D, H, W, dx, dz)
        _, t_l = bench(leap_fwd)
        y_l = g_t.clone()
        rel = float((y_l - y_o).norm() / y_o.norm())
        c = corr(y_l.cpu().numpy(), y_o.cpu().numpy())
        # backward
        with torch.no_grad():
            b_o, tb_o = bench(lambda: adjoint_project_3d_batched(
                y_o[None], Pm[None], u, v, D=D, H=H, W=W, dx=dx, dy=dy, dz=dz)[0, 0])
        s_t2 = y_o.contiguous()

        def leap_bp2():
            bpv.zero_()
            leap.backproject_gpu(s_t2, bpv)
            return bpv
        _, tb_l = bench(leap_bp2)
        b_l = bpv.clone()
        cs = float((b_l.flatten() @ b_o.flatten()) / (b_l.norm() * b_o.norm()))
        adj_l = abs(float((y_l * s_t2).sum()) - float((f_t * b_l).sum())) \
            / abs(float((y_l * s_t2).sum()))
        adj_o = abs(float((y_o * s_t2).sum()) - float((f_t * b_o).sum())) \
            / abs(float((y_o * s_t2).sum()))
        print(f"    {name}: max |theta| = {float(th[:, :3].abs().max()):.2f} mm / "
              f"{float(th[:, 3:].abs().max()) * 180 / np.pi:.2f} deg", flush=True)
        print(f"      forward  ours {1e3 * t_o / cfg.n_views:6.2f} ms/view | LEAP modular "
              f"{1e3 * t_l / cfg.n_views:6.2f} ms/view | rel-L2 {rel:.4e} corr {c:.6f}",
              flush=True)
        print(f"      backward ours {1e3 * tb_o / cfg.n_views:6.2f} ms/view (adj {adj_o:.1e}) | "
              f"LEAP {1e3 * tb_l / cfg.n_views:6.2f} ms/view (adj {adj_l:.1e}) | cos {cs:.6f}",
              flush=True)
        res["motion"][name] = dict(fwd_ours=t_o, fwd_leap=t_l, rel=rel, corr=c,
                                   bwd_ours=tb_o, bwd_leap=tb_l, cos=cs,
                                   adj_ours=adj_o, adj_leap=adj_l)
        if name == "akima_5mm5deg_p2p":
            sino["motion_ours"], sino["motion_leap"] = y_o, y_l

    # ============================================================================ [4] RECON ====
    print("\n[4] RECONSTRUCTION from the SAME (nominal) sinogram", flush=True)
    with torch.no_grad():
        r_ours, t_fdk = bench(lambda: fdk_conebeam_3d_batched(
            ref[None], P[None], u, v, cfg, D=D, H=H, W=W, dx=dx, dy=dy, dz=dz,
            window="shepphann")[0])
    leap = tomographicModels()                       # fresh: see the state-leak note in [1]
    leap.set_gpu(args.gpu)
    val = measured_region_mask((D, H, W), (dz, dy, dx), cfg, device=dev)
    # OUR FDK backprojects VOXEL-DRIVEN (bilinear detector interpolation per voxel), so the
    # apples-to-apples LEAP setting is `set_projector('VD')`; its 'SF' fbp integrates the voxel
    # FOOTPRINT instead, which is a visibly smoother image. Both are reported.
    recons = {}
    for which in ("VD", "SF"):
        leap_set_cone(leap, cfg, phis, D, H, W, dx, dz, u, v)
        leap.set_projector(which)
        leap.set_rampFilter(2)
        leap.set_FBPlowpass(2.0)
        # `leap.fbp` -> `filterProjections(g, g_out=None)` sets g_out = g: IT FILTERS IN PLACE.
        # Feeding it the same tensor twice (as `bench` does) ramp-filters an already-filtered
        # sinogram and silently returns garbage. Hand it a fresh copy every call.
        fl, t_leapfbp = bench(lambda: leap.fbp(ref.clone().contiguous()))
        rl = fl.clone() if isinstance(fl, torch.Tensor) else \
            torch.as_tensor(np.asarray(fl), device=dev)
        aa, b = rl[val].double(), r_ours[val].double()
        sc = float((aa * b).sum() / (aa * aa).sum().clamp_min(1e-30))
        rr = float((sc * aa - b).norm() / b.norm())
        cc = corr(aa.cpu().numpy(), b.cpu().numpy())
        print(f"    ours FDK shepphann {t_fdk:.2f} s | LEAP fbp[{which}] "
              f"(ord2+lowpass2.0) {t_leapfbp:.2f} s | in-barrel corr {cc:.6f} | "
              f"LS scale LEAP->ours {sc:.4f} | rel after scale {rr:.4e}", flush=True)
        res[f"recon_{which}"] = dict(ours_sec=t_fdk, leap_sec=t_leapfbp, ls_scale=sc,
                                     rel_after_scale=rr, corr=cc)
        recons[which] = rl
    r_leap = recons["VD"]
    a = r_leap[val].double()
    scale = res["recon_VD"]["ls_scale"]
    # Which of OUR apodizations does LEAP's fbp actually correspond to at this grid? The
    # backprojector is shared (2 above), so any residual here is the FILTER, not the operator.
    res["recon_window_sweep"] = {}
    for win in ("ramlak", "shepp", "hann", "shepphann"):
        with torch.no_grad():
            rw = fdk_conebeam_3d_batched(ref[None], P[None], u, v, cfg, D=D, H=H, W=W,
                                         dx=dx, dy=dy, dz=dz, window=win)[0]
        bw_ = rw[val].double()
        sc = float((a * bw_).sum() / (a * a).sum().clamp_min(1e-30))
        rr = float((sc * a - bw_).norm() / bw_.norm())
        print(f"      ours window {win:10s}: LS scale {sc:.4f}  rel vs LEAP fbp {rr:.4e}",
              flush=True)
        res["recon_window_sweep"][win] = dict(ls_scale=sc, rel=rr)
    # Split the FBP residual into its two stages: (a) the FILTER alone, compared on the
    # SINOGRAM against LEAP's `filterProjections` (the exact input its `weightedBackproject`
    # consumes), and (b) the WEIGHTED BACKPROJECTION, by feeding our filtered sinogram to
    # LEAP's weighted backprojector and comparing against LEAP's own fbp.
    with torch.no_grad():
        g_of, _ = fdk_conebeam_3d_batched(ref[None], P[None], u, v, cfg, D=D, H=H, W=W,
                                          dx=dx, dy=dy, dz=dz, window="shepphann",
                                          _return_filtered=True)
    g_lf = leap.filterProjections(ref.clone().contiguous())
    g_lf = g_lf if isinstance(g_lf, torch.Tensor) else torch.as_tensor(np.asarray(g_lf),
                                                                       device=dev)
    af, bf = g_lf.double(), g_of[0].double()
    scf = float((af * bf).sum() / (af * af).sum())
    relf = float((scf * af - bf).norm() / bf.norm())
    r_x = torch.zeros_like(r_ours)
    leap.set_projector("VD")           # same backprojector as the fbp[VD] we compare against
    leap.weightedBackproject(g_of[0].contiguous(), r_x)
    ax = r_x[val].double()
    scx = float((ax * a).sum() / (ax * ax).sum().clamp_min(1e-30))
    relx = float((scx * ax - a).norm() / a.norm())
    print(f"    stage split: FILTER (our shepphann vs LEAP ord2+lowpass2.0, on the sinogram) "
          f"rel {relf:.4e} corr {corr(af.cpu().numpy(), bf.cpu().numpy()):.6f}", flush=True)
    print(f"                 our filtered sinogram -> LEAP weightedBackproject, vs LEAP fbp: "
          f"rel {relx:.4e}", flush=True)
    res["stage_split"] = dict(filter_rel=relf, our_filter_into_leap_wbp_rel=relx)

    # ----------------------------------------------------------------------------- figures ----
    kv = cfg.n_views // 4
    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    panels = [("ours SF", sino["ours_SF"][kv]), ("LEAP cone SF", sino["leap_cone_SF"][kv]),
              ("diff x50", 50 * (sino["leap_cone_SF"] - sino["ours_SF"])[kv]),
              ("ours SF (motion)", sino["motion_ours"][kv]),
              ("LEAP modular (motion)", sino["motion_leap"][kv]),
              ("diff x50", 50 * (sino["motion_leap"] - sino["motion_ours"])[kv])]
    vmax = float(sino["ours_SF"][kv].max())
    for j, (t_, im) in enumerate(panels):
        a_ = ax[j // 3, j % 3]
        a_.imshow(im.detach().cpu(), cmap="gray", vmin=0 if "diff" not in t_ else -vmax,
                  vmax=vmax)
        a_.set_title(f"view {kv}: {t_}", fontsize=9)
        a_.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sino_compare.png"), dpi=120)
    plt.close(fig)

    zc = D // 2
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.6))
    vm = float(r_ours[zc].max())
    for j, (ttl, im) in enumerate([("ours FDK shepphann", r_ours[zc]),
                                   (f"LEAP fbp x{scale:.3f}", scale * r_leap[zc]),
                                   ("diff x20", 20 * (scale * r_leap - r_ours)[zc])]):
        ax[j].imshow(im.detach().cpu(), cmap="gray", vmin=-0.1 * vm if j == 2 else 0,
                     vmax=vm if j < 2 else 0.1 * vm)
        ax[j].set_title(ttl, fontsize=9)
        ax[j].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "recon_compare.png"), dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------------- [5] native grid ------
    if args.native:
        print("\n[5] NATIVE simulation grid (du * SOD/SDD)", flush=True)
        vx = du * cfg.SOD / cfg.SDD
        Nn = int(round(N * dx / vx))
        Nn += Nn % 2
        voln = head_phantom((Nn, Nn, Nn), (vx, vx, vx), device=dev)[None, None]
        fn_t = voln[0, 0].contiguous()
        with torch.no_grad():
            yn, tn_o = bench(lambda: forward_project_3d_batched(
                voln, P[None], u, v, dx=vx, dy=vx, dz=vx)[0])
        leap_set_cone(leap, cfg, phis, Nn, Nn, Nn, vx, vx, u, v)

        def leap_fwd_n():
            g_t.zero_()
            leap.project_gpu(g_t, fn_t)
            return g_t
        _, tn_l = bench(leap_fwd_n)
        rel = float((g_t - yn).norm() / yn.norm())
        print(f"    {Nn}^3 @ {vx:.4f} mm: ours {tn_o:.2f} s | LEAP cone SF {tn_l:.2f} s | "
              f"rel-L2 {rel:.4e}", flush=True)
        res["native"] = dict(n=Nn, vox=vx, ours_sec=tn_o, leap_sec=tn_l, rel=rel)
        # FBP on the NATIVE grid: if [4]'s residual is LEAP's coarse-grid anti-alias low-pass
        # (voxelWidth > du*SOD/SDD) rather than the operator, it must collapse here.
        cfg_n = ConeBeam3DConfig.thies(n_views=args.views)
        with torch.no_grad():
            rn_o = fdk_conebeam_3d_batched(yn[None], P[None], u, v, cfg_n, D=Nn, H=Nn, W=Nn,
                                           dx=vx, dy=vx, dz=vx, window="shepphann")[0]
        leap.set_projector("VD")                      # match our voxel-driven FDK: see [4]
        leap.set_rampFilter(2)
        leap.set_FBPlowpass(2.0)
        rn_l = leap.fbp(yn.clone().contiguous())      # in-place filter: see [4]
        rn_l = rn_l if isinstance(rn_l, torch.Tensor) else torch.as_tensor(np.asarray(rn_l),
                                                                          device=dev)
        valn = measured_region_mask((Nn, Nn, Nn), (vx, vx, vx), cfg_n, device=dev)
        an, bn = rn_l[valn].double(), rn_o[valn].double()
        scn = float((an * bn).sum() / (an * an).sum().clamp_min(1e-30))
        reln = float((scn * an - bn).norm() / bn.norm())
        print(f"    FBP on the native grid: corr {corr(an.cpu().numpy(), bn.cpu().numpy()):.6f}"
              f" | LS scale {scn:.4f} | rel after scale {reln:.4e}", flush=True)
        res["native_fbp"] = dict(ls_scale=scn, rel_after_scale=reln,
                                 corr=corr(an.cpu().numpy(), bn.cpu().numpy()))

    with open(os.path.join(OUT, "summary.json"), "w") as fj:
        json.dump(res, fj, indent=2)
    print(f"\noutputs -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
