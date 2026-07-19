"""Gate the Triton ray-march's adjoint w.r.t. the projection matrices.

The Triton kernel inherited from the 4DCT project differentiated only the VOLUME; the gradient in
the ray constants (A, Bk, step) -- the only route by which Pmat enters -- was dropped, and
autograd reads a dropped gradient as a ZERO. That is silent: motion estimation would converge
to theta = 0 and report nothing wrong. `triton_raymarch._bwd_kernel` now carries it.

This gate is what makes that safe to rely on. grid_sample is the reference: it is plain torch and
autograd differentiates it correctly by construction, so agreement with it -- to a tight
tolerance, on the quantity the estimator actually descends -- is the claim.

  T1  forward parity           the two backends compute the same line integrals
  T2  d(loss)/d(theta)         the motion gradient, the whole point. Direction (cosine) AND
                               magnitude, per DoF block.
  T2b d(loss)/d(theta), MOVED  the same parity at theta = 0.5*theta_true, so the adjoint is
                               tested off the nominal orbit, not only at P_nom.
  T3  d(loss)/d(volume)        unchanged by the new kernel -- a regression check, since the
                               backward now also loads the eight corner values it used to only
                               scatter into.
  T4  BOTH AT ONCE             a graph that differentiates the volume and Pmat together, which is
                               what a joint update would do. The two halves must not corrupt
                               each other.
  T5  speed                    what this was for.

    python scripts/gate_triton_adjoint.py
"""

from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.geometry_3d import ConeBeam3DConfig, build_conebeam_orbit, detector_coords_3d
from fm3d.phantom import head_phantom
from fm3d.projector_3d import forward_project_3d_batched
from fm3d.rigid_motion import params_to_Pmot

DEV = "cuda"
CFG = ConeBeam3DConfig(det_bin=4, n_views=64)
SHAPE = (96, 128, 128)
SPACING = (1.5, 1.5, 1.5)

_fails: list[str] = []


def check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:34s} {detail}")
    if not ok:
        _fails.append(name)


def cos(a, b):
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-20))


def main():
    dz, dy, dx = SPACING
    P_nom = build_conebeam_orbit(CFG, device=DEV)
    u, v = detector_coords_3d(CFG, device=DEV)
    vol = head_phantom(SHAPE, SPACING, device=DEV)
    kw = dict(dx=dx, dy=dy, dz=dz, n_samples=256, view_chunk=8, row_chunk=64)

    torch.manual_seed(0)
    th_true = torch.randn(CFG.n_views, 6, device=DEV) * torch.tensor(
        [2., 2., 2., .02, .02, .02], device=DEV)
    with torch.no_grad():
        y = forward_project_3d_batched(vol[None, None], params_to_Pmot(th_true, P_nom)[None],
                                       u, v, backend="gridsample", **kw)

    def run(backend, theta=None, volume=None):
        th = torch.zeros(CFG.n_views, 6, device=DEV) if theta is None else theta
        x = vol if volume is None else volume
        pred = forward_project_3d_batched(
            x[None, None], params_to_Pmot(th, P_nom)[None], u, v, backend=backend, **kw)
        return pred, 0.5 * ((pred - y) ** 2).mean()

    # ---- T1 forward parity
    print("T1  forward parity")
    with torch.no_grad():
        pg, _ = run("gridsample")
        pt, _ = run("triton")
    rel = float((pg - pt).norm() / pg.norm())
    check("line integrals agree", rel < 1e-5, f"rel diff = {rel:.2e}")

    # ---- T2 d/d theta  -- the reason this kernel exists
    print("\nT2  d(loss)/d(theta)   [the motion gradient]")
    grads = {}
    for backend in ("gridsample", "triton"):
        th = torch.zeros(CFG.n_views, 6, device=DEV, requires_grad=True)
        run(backend, theta=th)[1].backward()
        grads[backend] = th.grad.clone()
    gg, gt = grads["gridsample"], grads["triton"]
    check("gradient is not zero", float(gt.norm()) > 0,
          f"|g_triton| = {gt.norm():.4e}  (the old kernel returned exactly 0 here)")
    for name, sl in [("translation", slice(0, 3)), ("rotation", slice(3, 6))]:
        a, b = gg[:, sl], gt[:, sl]
        c = cos(a, b)
        r = float((a - b).norm() / a.norm())
        check(f"{name} block", c > 0.9999 and r < 2e-3,
              f"cos = {c:.6f}  rel = {r:.2e}  |grid| {a.norm():.3e} |triton| {b.norm():.3e}")

    # ---- T2b the same parity at a MOVED geometry. T2 evaluates the adjoint only at
    # theta = 0, i.e. at P = P_nom exactly -- an adjoint error term that cancels on the nominal
    # source ring would pass it. Re-run mid-descent, where the estimator actually lives.
    print("\nT2b d(loss)/d(theta) at theta = 0.5*theta_true   [moved geometry]")
    grads = {}
    for backend in ("gridsample", "triton"):
        th = (0.5 * th_true).detach().clone().requires_grad_(True)
        run(backend, theta=th)[1].backward()
        grads[backend] = th.grad.clone()
    gg, gt = grads["gridsample"], grads["triton"]
    for name, sl in [("translation", slice(0, 3)), ("rotation", slice(3, 6))]:
        a, b = gg[:, sl], gt[:, sl]
        c = cos(a, b)
        r = float((a - b).norm() / a.norm())
        check(f"moved {name} block", c > 0.9999 and r < 2e-3,
              f"cos = {c:.6f}  rel = {r:.2e}  |grid| {a.norm():.3e} |triton| {b.norm():.3e}")

    # ---- T3 d/d volume -- regression
    print("\nT3  d(loss)/d(volume)   [regression: unchanged]")
    gv = {}
    for backend in ("gridsample", "triton"):
        x = vol.clone().requires_grad_(True)
        run(backend, volume=x)[1].backward()
        gv[backend] = x.grad.clone()
    r = float((gv["gridsample"] - gv["triton"]).norm() / gv["gridsample"].norm())
    check("volume gradient", cos(gv["gridsample"], gv["triton"]) > 0.9999 and r < 2e-3,
          f"cos = {cos(gv['gridsample'], gv['triton']):.6f}  rel = {r:.2e}")

    # ---- T4 both at once
    print("\nT4  volume AND theta in one graph")
    out = {}
    for backend in ("gridsample", "triton"):
        x = vol.clone().requires_grad_(True)
        th = torch.zeros(CFG.n_views, 6, device=DEV, requires_grad=True)
        run(backend, theta=th, volume=x)[1].backward()
        out[backend] = (x.grad.clone(), th.grad.clone())
    for i, name in enumerate(["volume", "theta"]):
        a, b = out["gridsample"][i], out["triton"][i]
        r = float((a - b).norm() / a.norm())
        check(f"joint: {name}", cos(a, b) > 0.9999 and r < 2e-3,
              f"cos = {cos(a, b):.6f}  rel = {r:.2e}")

    # ---- T5 speed
    print("\nT5  speed  (forward + backward through theta)")
    for backend in ("gridsample", "triton"):
        th = torch.zeros(CFG.n_views, 6, device=DEV, requires_grad=True)
        for _ in range(2):                                   # warm up / JIT
            run(backend, theta=th)[1].backward()
        th.grad = None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(5):
            th.grad = None
            run(backend, theta=th)[1].backward()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / 5
        print(f"      {backend:11s} {dt * 1e3:7.1f} ms   peak {torch.cuda.max_memory_allocated() / 2**30:5.2f} GiB")
        if backend == "gridsample":
            base, bmem = dt, torch.cuda.max_memory_allocated()
    speed = base / dt
    print(f"      -> {speed:.1f}x faster, {bmem / torch.cuda.max_memory_allocated():.1f}x less peak memory")
    check("triton is faster", speed > 1.0, f"{speed:.1f}x")

    print("\n" + "=" * 70)
    print(f"FAILED: {', '.join(_fails)}" if _fails else "all gates passed")
    print("=" * 70)
    return 1 if _fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
