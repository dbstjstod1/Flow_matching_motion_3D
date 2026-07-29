"""Does `l2si` actually do anything here, or is plain `l2` the same loss?

WHY IT IS IN DOUBT (user, 2026-07-28). `l2si` was adopted from the 2D project, where it fixed a
REAL failure: the flow-matching push and the data-residual update disagreed about the image's
overall brightness, x oscillated in scale, and a plain L2 sinogram term charged that oscillation
to the motion parameters. If the 3D loop does not have that scale drift, `l2si` is a no-op that we
are carrying for historical reasons -- and it is not free, because it makes the data term blind to
one degree of freedom.

THE ALGEBRA FIRST. With c = <p,y>/<p,p> substituted back,

    L_l2si = || c p - y ||^2 / N = ( ||y||^2 - <p,y>^2/||p||^2 ) / N = ||y||^2 sin^2(angle(p,y)) / N

so `l2si` is a PURELY ANGULAR loss: it depends on the DIRECTION of the rendered sinogram and on
nothing else. `l2` sees direction AND magnitude. The two therefore differ exactly to the extent
that the rendered sinogram's magnitude is wrong, i.e. exactly to the extent that c != 1.

WHAT THIS SCRIPT MEASURES, on the reference images the loop actually hands the estimator (the
carried x_t of a finished run, step by step, plus the cold FDK it starts from and the GT ceiling):

  1. c itself, globally and over the 24-view minibatches the estimator really uses -- the
     minibatch SPREAD matters as much as the mean, because a c that jitters per batch injects
     gradient noise that `l2` would not have.
  2. the geometry of the two gradients w.r.t. theta at the SAME point and on the SAME batch:
     cosine similarity (do they point the same way?) and norm ratio (how much of an lr rescale
     would a switch to `l2` require?). If cos ~ 1 and the ratio is ~constant, `l2` at a rescaled
     lr is the same optimizer and the in-loop A/B can only find noise.

    python scripts/exp_loss_l2_geom.py --from data/runs/akima55/c2f_v0
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.dataset_cq500 import CQ500Generator                            # noqa: E402
from fm3d.geometry_3d import ConeBeam3DConfig, detector_coords_3d        # noqa: E402
from fm3d.motion_estimation import sinogram_data_loss                    # noqa: E402
from fm3d.projector_3d import forward_project_3d_batched                 # noqa: E402
from fm3d.rigid_motion import (amp_from_run_args, make_motion,           # noqa: E402
                                params_to_Pmot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="logs/fm3d_cq500/ckpt_iter500000.pth")
    ap.add_argument("--from", dest="src", default="data/runs/akima55/c2f_v0",
                    help="a finished run whose snapshots supply the loop's own reference images")
    ap.add_argument("--views", type=int, default=24, help="the estimator's minibatch size")
    ap.add_argument("--batches", type=int, default=24, help="minibatches averaged per row")
    ap.add_argument("--coarse", type=int, default=2, help="estimation grid, as deployed")
    args = ap.parse_args()

    dev = "cuda"
    meta = torch.load(os.path.join(args.src, "snaps", "meta.pt"), map_location="cpu",
                      weights_only=False)["args"]
    run, seed = meta["run"], meta["seed"]
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    ca = ck["args"]
    cfg = ConeBeam3DConfig.thies(n_views=ca["views"])
    gen = CQ500Generator(ca["root"], cfg=cfg, device=dev, split=meta.get("split", "val"),
                         shape=tuple(ca["shape"]), voxel_mm=1.0)
    gt = gen.volume(run)
    th_true = make_motion(meta["motion_kind"], cfg.n_views, device=dev, seed=seed,
                          trans_mm=amp_from_run_args(meta)[0], rot_deg=amp_from_run_args(meta)[1])
    with torch.no_grad():
        y = gen.simulate(run, params_to_Pmot(th_true, gen.P_nom)[None])[0]
        cold = gen.fdk(y[None], gen.P_nom[None])[0]

    # the estimator's grid, exactly as run_posterior3d builds it (det_bin x coarse, half voxels)
    c = args.coarse
    est_cfg = ConeBeam3DConfig.thies(n_views=cfg.n_views, det_bin=cfg.det_bin * c) if c > 1 else cfg
    eu, ev = detector_coords_3d(est_cfg, device=dev) if c > 1 else (gen.u_coords, gen.v_coords)
    y_e = F.avg_pool2d(y[None], c)[0] if c > 1 else y
    vox = float(c) * gen.dx

    def down(v):
        return F.avg_pool3d(v[None, None], c)[0, 0] if c > 1 else v

    # the loop's own reference images, in order of the ODE
    refs = [("cold FDK (step 0 input)", cold, torch.zeros_like(th_true))]
    snaps = sorted(f for f in os.listdir(os.path.join(args.src, "snaps")) if f.startswith("step"))
    for f in snaps:
        s = torch.load(os.path.join(args.src, "snaps", f), map_location=dev, weights_only=False)
        refs.append((f"x_t step {s['step']:3d} (t={s['t']:.2f})", s["x_t"].float(),
                     s["theta"].float().to(dev)))
    refs.append(("ground truth", gt[0, 0], th_true))

    V = cfg.n_views
    g = torch.Generator(device=dev).manual_seed(0)
    print(f"val {run} seed {seed} | grid {vox:g} mm, panel {est_cfg.nv}x{est_cfg.nu} | "
          f"minibatch {args.views}/{V} views x {args.batches}\n")
    print(f"{'reference image':30} {'c global':>9} {'c batch mean+-sd':>19} {'[min,max]':>15} "
          f"{'cos(g_l2,g_l2si)':>17} {'|g_l2|/|g_l2si|':>16}")
    for name, img, th in refs:
        im = down(img.to(dev))

        def proj(theta, views=None):
            P = params_to_Pmot(theta if views is None else theta[views],
                               gen.P_nom if views is None else gen.P_nom[views])
            return forward_project_3d_batched(im[None, None], P[None], eu, ev, dx=vox, dy=vox,
                                              dz=vox)[0]

        with torch.no_grad():
            p = proj(th)
            c_glob = float((p * y_e).sum() / (p * p).sum().clamp_min(1e-12))
        cs, cos, rat = [], [], []
        for _ in range(args.batches):
            vw = torch.randperm(V, device=dev, generator=g)[: args.views]
            with torch.no_grad():
                pv, yv = p[vw], y_e[vw]
                cs.append(float((pv * yv).sum() / (pv * pv).sum().clamp_min(1e-12)))
            gs = []
            for kind in ("l2", "l2si"):
                t = th.clone().requires_grad_(True)
                loss = sinogram_data_loss(proj(t, vw), y_e[vw], kind, du=est_cfg.du)
                gs.append(torch.autograd.grad(loss, t)[0].flatten())
            cos.append(float(F.cosine_similarity(gs[0], gs[1], dim=0)))
            rat.append(float(gs[0].norm() / gs[1].norm().clamp_min(1e-30)))
        cs = torch.tensor(cs)
        print(f"{name:30} {c_glob:9.4f} {cs.mean():10.4f} +-{cs.std():6.4f} "
              f"[{cs.min():.3f},{cs.max():.3f}] {torch.tensor(cos).mean():17.4f} "
              f"{torch.tensor(rat).mean():16.4f}")

    print("\nc == 1 means the rendered sinogram already has the right magnitude, and l2si -- which "
          "is\nblind to magnitude by construction -- can only differ from l2 through second-order "
          "terms.\ncos ~ 1 with a stable norm ratio means `l2` at lr x (that ratio) IS the same "
          "optimizer.")


if __name__ == "__main__":
    main()
