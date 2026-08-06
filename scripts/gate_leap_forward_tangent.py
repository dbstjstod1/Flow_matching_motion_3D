"""Gate `triton_leap_grad.leap_forward_tangent` -- the EXACT s-derivative of LEAP's pinned
Joseph FORWARD projection, written 2026-08-05 for the data bridge's velocity target.

WHY THE REFERENCE IS float64 TORCH AND NOT A FINITE DIFFERENCE. The whole reason this kernel
exists is that an fp32 central difference of the forward projection bottoms out at ~2% of the
target (truncation above, cancellation below, amplified through the FDK's ramp -- measured in
scripts/diag_bridge_data_tangent.py). A gate built on that difference could never certify
better than the thing the kernel replaced. So the reference here is a float64 TORCH
transcription of the same Joseph maths, differentiated by autograd's jvp: exact to fp64, and
comparing like with like (same model, same a.e. conventions) rather than across models -- the
mistake the 2026-07-30 ledger records for the backprojection tangent.

  F1  value parity     torch64 Joseph == leap_forward_model (the Triton value path)
  F2  value parity     leap_forward_tangent's value == leap_forward_model's
  F3  THE TANGENT      leap_forward_tangent's dy/ds == autograd jvp of the torch64 model,
                       through the full chain (s -> P(s) -> modular arrays -> projection)
  F4  fp64 FD sanity   the torch64 model's own central difference agrees with its jvp, so F3's
                       reference is not itself an artifact of autograd
  F5  fp32 FD is worse: the deployed fd path (--data_tangent fd) sits further from the truth
                       than the analytic one does. This is the claim that justifies the kernel.

    python scripts/gate_leap_forward_tangent.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.geometry_3d import ConeBeam3DConfig, build_conebeam_orbit
from fm3d.rigid_motion import bridge_P_and_dP, params_to_Pmot, random_motion
from fm3d.triton_leap_grad import (leap_forward_model, leap_forward_tangent,
                                   modular_arrays_torch)

n_fail = 0
# small on purpose: the reference marches NP planes in python, and a gate wants to run in
# seconds. The maths is per-ray, so size proves nothing extra.
NPX, NV, NU, NVIEW = 64, 24, 32, 4
DXV = 2.0                     # mm, isotropic (LEAP's Joseph kernel requires it)


def check(name, ok, detail=""):
    global n_fail
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}   {detail}")
    if not ok:
        n_fail += 1


def rel(a, b):
    return float(torch.linalg.vector_norm(a.double() - b.double())
                 / (torch.linalg.vector_norm(b.double()) + 1e-30))


def joseph_torch64(vol, arrs, *, nv, nu, dx, dz, du, dv):
    """float64 torch transcription of `_leap_joseph_kernel`'s VALUE path. vol (D,H,W).

    Line-for-line with the Triton kernel, including the frozen dominant-axis choice, the
    entry-plane 0.5 weight, and BORDER addressing (out-of-range taps read zero). Differentiable
    in `arrs`, which is the point: autograd then supplies the exact jvp to compare against.
    """
    src, mod, rowv, colv = arrs
    D, H, W = vol.shape
    NP = W
    u0g = -0.5 * (nu - 1) * du
    v0g = -0.5 * (nv - 1) * dv
    b0 = -0.5 * (W - 1) * dx
    z0 = -0.5 * (D - 1) * dz

    dev, dt = vol.device, vol.dtype
    t_r = (torch.arange(nv, device=dev, dtype=dt) * dv + v0g)[:, None]        # (nv,1)
    s_c = (torch.arange(nu, device=dev, dtype=dt) * du + u0g)[None, :]        # (1,nu)

    p = src[:, None, None, :]                                                # (V,1,1,3)
    det = mod[:, None, None, :] + colv[:, None, None, :] * s_c[..., None] \
        + rowv[:, None, None, :] * t_r[..., None]                            # (V,nv,nu,3)
    r = det - p
    rx, ry, rz = r[..., 0], r[..., 1], r[..., 2]
    px = p[..., 0].expand_as(rx)
    py = p[..., 1].expand_as(rx)
    pz = p[..., 2].expand_as(rx)

    ydom = ry.abs() > rx.abs()
    r_a = torch.where(ydom, ry, rx)
    r_b = torch.where(ydom, rx, ry)
    p_a = torch.where(ydom, py, px)
    p_b = torch.where(ydom, px, py)
    inv_ra = 1.0 / r_a
    j0 = torch.where(r_a > 0, torch.zeros_like(r_a), torch.full_like(r_a, NP - 1))

    nrm = torch.sqrt(rx * rx + ry * ry + rz * rz)
    L = dx * nrm / r_a.abs()

    volf = vol.reshape(-1)
    HW = H * W
    S_tot = torch.zeros_like(r_a)
    for j in range(NP):
        w_j = j * dx + b0
        lam = (w_j - p_a) * inv_ra
        cb = (p_b + lam * r_b - b0) / dx
        cz = (pz + lam * rz - z0) / dz
        ibf, izf = torch.floor(cb), torch.floor(cz)
        wb, wz = cb - ibf, cz - izf
        ib, iz = ibf.long(), izf.long()

        def tap(di, dj):
            i2, z2 = ib + di, iz + dj
            ok = (i2 >= 0) & (i2 < NP) & (z2 >= 0) & (z2 < D)
            # the kernel's flat index: iz*HW + (ydom ? j*NP : j) + ib*(ydom ? 1 : NP)
            base = torch.where(ydom, torch.full_like(i2, j * NP), torch.full_like(i2, j))
            step = torch.where(ydom, torch.ones_like(i2), torch.full_like(i2, NP))
            idx = (z2.clamp(0, D - 1) * HW + base + i2.clamp(0, NP - 1) * step)
            return torch.where(ok, volf[idx], torch.zeros_like(wb))

        S = (1 - wb) * (1 - wz) * tap(0, 0) + wb * (1 - wz) * tap(1, 0) \
            + (1 - wb) * wz * tap(0, 1) + wb * wz * tap(1, 1)
        S_tot = S_tot + torch.where(j0 == j, 0.5, 1.0) * S
    return L * S_tot


def main():
    torch.manual_seed(0)
    dev = "cuda"
    cfg = ConeBeam3DConfig.thies(n_views=NVIEW)
    P_nom = build_conebeam_orbit(cfg, device=dev).to(torch.float64)[:NVIEW]
    du, dv = float(cfg.du), float(cfg.dv)

    # a phantom with real edges: bilinear taps are where a tangent can go wrong
    zz, yy, xx = torch.meshgrid(*[torch.arange(NPX, device=dev, dtype=torch.float64)
                                  - (NPX - 1) / 2] * 3, indexing="ij")
    vol = torch.zeros((NPX, NPX, NPX), device=dev, dtype=torch.float64)
    vol[(xx ** 2 + yy ** 2 + zz ** 2) < (0.30 * NPX) ** 2] = 0.02
    vol[((xx - 6) ** 2 + (yy + 4) ** 2 + (zz - 3) ** 2) < (0.10 * NPX) ** 2] = 0.045
    vol32 = vol.to(torch.float32)

    theta = random_motion(NVIEW, trans_mm=15.0, rot_deg=20.0, amp_mode="thies",
                          device=dev).to(torch.float64)
    s0 = 0.5
    P_s, Pdot_s = bridge_P_and_dP(theta, P_nom, s0)

    print("\nF1/F2  value parity (the tangent kernel must not drift from the value path)")
    y_tri = leap_forward_model(vol32[None], P_s[None].to(torch.float32), nv=NV, nu=NU,
                               dx=DXV, dy=DXV, dz=DXV, du=du, dv=dv)[0]
    with torch.no_grad():
        arrs0 = modular_arrays_torch(P_s, 0.0, 0.0)
        y_ref = joseph_torch64(vol, arrs0, nv=NV, nu=NU, dx=DXV, dz=DXV, du=du, dv=dv)
    check("torch64 Joseph == leap_forward_model", rel(y_tri, y_ref) < 1e-5,
          f"rel {rel(y_tri, y_ref):.3e}")

    y_tan, dy_tan = leap_forward_tangent(vol32[None], P_s[None].to(torch.float32),
                                         Pdot_s[None].to(torch.float32),
                                         nv=NV, nu=NU, dx=DXV, dy=DXV, dz=DXV, du=du, dv=dv)
    check("leap_forward_tangent's value == leap_forward_model", rel(y_tan[0], y_tri) < 1e-6,
          f"rel {rel(y_tan[0], y_tri):.3e}")

    print("\nF3  THE TANGENT vs the float64 autograd jvp of the same model")

    def model_at(sv):
        P = params_to_Pmot((sv * theta.reshape(-1)).reshape(theta.shape), P_nom)
        return joseph_torch64(vol, modular_arrays_torch(P, 0.0, 0.0),
                              nv=NV, nu=NU, dx=DXV, dz=DXV, du=du, dv=dv)

    s_leaf = torch.tensor(s0, device=dev, dtype=torch.float64)
    _, dy_jvp = torch.autograd.functional.jvp(
        model_at, s_leaf, torch.tensor(1.0, device=dev, dtype=torch.float64))
    r3 = rel(dy_tan[0], dy_jvp)
    c3 = float(torch.sum(dy_tan[0].double() * dy_jvp)
               / (torch.linalg.vector_norm(dy_tan[0].double())
                  * torch.linalg.vector_norm(dy_jvp) + 1e-30))
    check("dy/ds == fp64 jvp (magnitude)", r3 < 2e-3, f"rel {r3:.3e}")
    check("dy/ds == fp64 jvp (direction)", c3 > 0.99999, f"cos {c3:.8f}")

    print("\nF4  the reference is not an autograd artifact: fp64 central difference of itself")
    e = 1e-5
    with torch.no_grad():
        fd64 = (model_at(s0 + e) - model_at(s0 - e)) / (2 * e)
    check("fp64 FD == fp64 jvp", rel(fd64, dy_jvp) < 1e-6, f"rel {rel(fd64, dy_jvp):.3e}")

    print("\nF5  and the fp32 central difference really is worse (why the kernel exists)")
    with torch.no_grad():
        for d in (0.02, 0.005):
            def sim32(sv):
                P = params_to_Pmot((sv * theta.reshape(-1)).reshape(theta.shape),
                                   P_nom).to(torch.float32)
                return leap_forward_model(vol32[None], P[None], nv=NV, nu=NU,
                                          dx=DXV, dy=DXV, dz=DXV, du=du, dv=dv)[0]
            fd32 = (sim32(s0 + d) - sim32(s0 - d)) / (2 * d)
            rfd = rel(fd32, dy_jvp)
            check(f"analytic beats fd(delta={d})", r3 < rfd,
                  f"fd rel {rfd:.3e}  vs  analytic {r3:.3e}")

    print(f"\n{'ALL GATES PASS' if n_fail == 0 else f'{n_fail} GATE(S) FAILED'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
