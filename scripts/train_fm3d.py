"""Train the 3D flow-matching prior on the geometry bridge (Sec. II.B of the paper).

The bridge is the family of FDK reconstructions of the same scan with the motion attenuated
linearly to zero:

    x_t = FDK( A(x; P_nom @ T((1-t) theta)), P_nom ),      t in [0, 1]

so x_0 is the uncorrected reconstruction the inference loop starts from and x_1 is the static
(motion-free) FDK, both by construction. Because FDK is linear in the sinogram and its geometry
does not depend on t, the velocity target is available in closed form,

    dx_t/dt = FDK( dy_t/dt, P_nom ),   dy_t/dt = -(dA/dP)[x; P((1-t) theta)] . P_nom Tdot theta,

with dA/dP the exact geometry derivative of the (LEAP, Joseph) forward projector
(`fm3d/triton_leap_grad.leap_forward_tangent`). The network regresses v_phi(x_t, t) onto that
tangent with t ~ U[0, 1].

`--bridge linear` trains the paper's bridge ablation: the pixel-space line between the same two
endpoints, x_t = (1-t) x_0 + t x_1, whose velocity target is the constant x_1 - x_0.

The network only sees `--patch`^3 patches (32^3 = volume/8, the rule of arXiv:2512.18161) plus
four conditioning channels (the whole x_t downsampled to the patch grid and the patch's absolute
z/y/x coordinates); the operator runs on the full volume under no_grad. Bridge draws are
expensive, so a rolling cache of `--cache` whole-volume draws is refreshed every `--refresh`
steps and each batch mixes patches from several draws.

The defaults reproduce the deployed run exactly (CQ500 256^3 @ 1 mm, patch 32 / batch 64,
cache 8 / refresh 12, 500k iterations, cosine lr 1e-4 -> 1e-6, EMA 0.999, fp16 AMP, native
612^3 simulation grid, Thies training amplitudes):

    python scripts/train_fm3d.py --out logs/fm3d_databridge
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # for val_fm3d

from fm3d.dataset_cq500 import CQ500Generator
from fm3d.geometry_3d import ConeBeam3DConfig, measured_region_mask
from fm3d.prior_patch import make_tile_inputs, volume_context
from fm3d.rigid_motion import (AMP_UNITS, bridge_P_and_dP, params_to_Pmot,
                               random_motion)
from fm3d.unet_3d import UNet3D
from val_fm3d import run_validation


def sample_t(n: int, device) -> torch.Tensor:
    """t ~ U[0,1], as in arXiv:2512.18161 and the bridge models this prior belongs to."""
    return torch.rand(n, device=device)


@torch.no_grad()
def bridge_pair_data(gen, idx: int, t: torch.Tensor, theta, delta: float = 0.005,
                     mode: str = "analytic"):
    """(x_t, dx_t/dt) in NET space for the geometry bridge, one volume `idx`, scalar `t`.

        x_t = FDK( A(x; P_nom @ T((1-t) theta)), P_nom )

    The motion decays in the MEASUREMENT while the reconstruction geometry stays nominal, so
    x_0 is the inference cold start and x_1 the static FDK exactly, and the tangent carries no
    angular-weight term. mode="analytic" (default) takes dy/ds from the exact s-derivative of
    the Joseph forward kernel (`gen.simulate_tangent`); mode="fd" is a central difference kept
    as the gate's counterparty (it bottoms out at ~2% of the target and is not used to train).
    """
    tv = float(t)
    s = 1.0 - tv
    if mode == "analytic":
        P_s, Pdot_s = bridge_P_and_dP(theta, gen.P_nom, s)
        x_t = gen.to_net(gen.fdk(gen.simulate(idx, P_s[None]), gen.P_nom[None])[0])
        _, dy_ds = gen.simulate_tangent(idx, P_s[None], Pdot_s[None])
    elif mode == "fd":
        def sim(sv: float):
            return gen.simulate(idx, params_to_Pmot(sv * theta, gen.P_nom)[None])

        x_t = gen.to_net(gen.fdk(sim(s), gen.P_nom[None])[0])
        sp, sm = min(s + delta, 1.0), max(s - delta, 0.0)
        dy_ds = (sim(sp) - sim(sm)) / (sp - sm)
    else:
        raise ValueError(f"unknown data-bridge tangent mode: {mode!r}")
    # d/dt = -d/ds, and the FDK is linear: one backprojection of the sinogram derivative.
    dx = -gen.to_net_tangent(gen.fdk(dy_ds, gen.P_nom[None])[0])
    return x_t, dx


@torch.no_grad()
def bridge_pair_linear(gen, idx: int, t: torch.Tensor, y):
    """(x_t, dx_t/dt) in NET space for the pixel-linear bridge (the ablation arm):

        x_t = (1-t) x_0 + t x_1,   x_0 = FDK(y_theta, P_nom),   x_1 = the static FDK

    Same endpoints as the geometry bridge (x_1 is the memoized static FDK, which IS the geometry
    bridge's t=1 image), so the two arms differ only in the path. The tangent x_1 - x_0 is exact
    by definition; the affine offset of `to_net` cancels in the difference.
    """
    x0 = gen.to_net(gen.fdk(y, gen.P_nom[None])[0])
    x1 = gen.static_anchor_net(idx)
    dx = x1 - x0
    return x0 + float(t) * dx, dx


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="cq500", choices=["cq500"])
    ap.add_argument("--root", default="data/CQ500")
    ap.add_argument("--split", default="train")
    ap.add_argument("--shape", type=int, nargs=3, default=(256, 256, 256), help="grid @ 1 mm")
    ap.add_argument("--out", default="logs/fm3d_a")
    ap.add_argument("--iters", type=int, default=500000)
    ap.add_argument("--batch", type=int, default=64, help="patches per step")
    ap.add_argument("--patch", type=int, default=32,
                    help="patch edge [voxels]; 32 = volume/8 at 256^3 (arXiv:2512.18161)")
    ap.add_argument("--cache", type=int, default=8, help="whole-volume bridge draws held at once")
    ap.add_argument("--refresh", type=int, default=12, help="steps between refreshing one draw")
    ap.add_argument("--views", type=int, default=360)
    ap.add_argument("--base", type=int, default=32, help="U-Net base width (mults 1,2,4)")
    ap.add_argument("--context", default="global", choices=["global", "none"],
                    help="global = the four conditioning channels (in_ch=5); none = bare patch")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_final", type=float, default=1e-6,
                    help="cosine-decay the lr from --lr to this value at --iters (None = constant)")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--amp", action="store_true", default=True, help="fp16 AMP (--no-amp: fp32)")
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--amp_dtype", default="float16", choices=["float16", "bfloat16"])
    # ---- motion. ALL AMPLITUDES ARE PEAK-TO-PEAK (fm3d/rigid_motion) ------------------------
    ap.add_argument("--trans_mm", type=float, default=10.0,
                    help="EVALUATION translation amplitude [mm] (inline validation only)")
    ap.add_argument("--rot_deg", type=float, default=10.0,
                    help="EVALUATION rotation amplitude [deg] (inline validation only)")
    ap.add_argument("--motion_amp", default="thies", choices=["fixed", "thies"],
                    help="training amplitudes: thies = per-DoF u~U(0,1) fraction of the --train_* "
                         "maxima (Thies et al., Sec. II-B); fixed = every DoF at --trans_mm/--rot_deg")
    ap.add_argument("--train_trans_mm", type=float, default=15.0,
                    help="max TRAINING translation [mm] (--motion_amp thies)")
    ap.add_argument("--train_rot_deg", type=float, default=20.0,
                    help="max TRAINING rotation [deg] (--motion_amp thies)")
    # ---- the bridge ---------------------------------------------------------------------------
    ap.add_argument("--bridge", default="data", choices=["data", "linear"],
                    help="data = the geometry bridge (the method); linear = the pixel-linear "
                         "bridge between the same endpoints (the ablation)")
    ap.add_argument("--data_tangent", default="analytic", choices=["analytic", "fd"],
                    help="velocity target of the geometry bridge: analytic = exact geometry "
                         "derivative of the Joseph forward (default); fd = central difference "
                         "(gate counterparty only)")
    ap.add_argument("--bridge_delta", type=float, default=0.02,
                    help="central-difference step in s for --data_tangent fd")
    ap.add_argument("--sim_grid", default="native", choices=["native", "coarse"],
                    help="native = simulate y from the 612^3 @ 0.42 mm volume (no inverse crime); "
                         "coarse = simulate on the 1 mm reconstruction grid")
    # ---- bookkeeping --------------------------------------------------------------------------
    ap.add_argument("--resume", default=None,
                    help="a ckpt .pth or a run dir (uses ckpt_last.pth): restore model+EMA+"
                         "optimizer+RNG and continue to --iters")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--keep_ckpts", type=int, default=0,
                    help="keep only the newest N numbered checkpoints (0 = keep all)")
    ap.add_argument("--val_every", type=int, default=10000,
                    help="inline prior-only validation every N iterations (0 = off)")
    ap.add_argument("--val_patients", type=int, default=3)
    ap.add_argument("--val_ode_steps", type=int, default=50)
    ap.add_argument("--no_tb", action="store_true", help="disable the tensorboard writer")
    ap.add_argument("--compile", action="store_true", default=True,
                    help="torch.compile the velocity net (--no-compile to disable)")
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    args = ap.parse_args()

    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)

    motion_gen = None
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        motion_gen = torch.Generator().manual_seed(args.seed + 1)
        print(f"seeded torch+numpy with {args.seed} (motion generator: {args.seed + 1})")

    cfg = ConeBeam3DConfig.thies(n_views=args.views)
    gen = CQ500Generator(args.root, cfg, device=dev, split=args.split,
                         shape=tuple(args.shape), voxel_mm=1.0,
                         sim_native=(args.sim_grid == "native"))
    print(f"CQ500 '{args.split}': {gen.n_slabs} patients | grid {gen.shape} @ 1 mm")
    print(f"detector {cfg.nv}x{cfg.nu} @ {cfg.du:.3f} mm | FOV {cfg.fov_diameter_mm():.0f} mm | "
          f"{cfg.n_views} views")

    # Patches are drawn only from the measured region (a barrel that narrows with radius):
    # training on never-measured voxels would teach the prior to hallucinate where inference has
    # no data to correct it.
    meas = measured_region_mask(gen.shape, (gen.dz, gen.dy, gen.dx), cfg, device=dev)
    p = args.patch
    D, H, W = gen.shape
    ok = []
    for z in range(0, D - p + 1, 8):
        for yy in range(0, H - p + 1, 16):
            for xx in range(0, W - p + 1, 16):
                if meas[z:z + p, yy:yy + p, xx:xx + p].float().mean() > 0.9:
                    ok.append((z, yy, xx))
    if not ok:
        raise RuntimeError("no patch fits inside the measured region; shrink --patch")
    print(f"valid patch origins: {len(ok)}")

    in_ch = 5 if args.context == "global" else 1
    model = UNet3D(in_ch=in_ch, base=args.base).to(dev)
    ema = UNet3D(in_ch=in_ch, base=args.base).to(dev)
    ema.load_state_dict(model.state_dict())
    for q in ema.parameters():
        q.requires_grad_(False)
    n_par = sum(q.numel() for q in model.parameters())
    print(f"UNet3D in_ch={in_ch} (context={args.context}) base={args.base}: "
          f"{n_par / 1e6:.2f} M params")

    # torch.compile shares the parameters, so `model` stays the canonical module for the
    # optimizer, the EMA and the checkpoints (plain state_dict keys); `net` is the compiled forward.
    net = torch.compile(model) if args.compile else model
    if args.compile:
        print("torch.compile: ON (net fwd/bwd only). --no-compile to disable.")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    def lr_at(it):
        """Cosine schedule as a pure function of `it` (nothing to save for --resume)."""
        if args.lr_final is None:
            return args.lr
        prog = min(it, args.iters) / max(args.iters, 1)
        return args.lr_final + 0.5 * (args.lr - args.lr_final) * (1 + math.cos(math.pi * prog))

    # ---- resume: model + EMA + optimizer + RNG streams ---------------------------------------
    start_it = 0
    if args.resume:
        ck_path = (os.path.join(args.resume, "ckpt_last.pth")
                   if os.path.isdir(args.resume) else args.resume)
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        prev = ck.get("args", {})

        def _cmp(v):
            return tuple(v) if isinstance(v, (list, tuple)) else v

        for k in ("base", "patch", "context", "bridge", "shape", "views", "dataset",
                  "data_tangent", "trans_mm", "rot_deg", "sim_grid",
                  "motion_amp", "train_trans_mm", "train_rot_deg"):
            if k in prev and k in vars(args) and _cmp(prev[k]) != _cmp(vars(args)[k]):
                raise SystemExit(f"--resume mismatch on '{k}': checkpoint has {prev[k]!r}, "
                                 f"this run asks for {vars(args)[k]!r}")
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        if ck.get("scaler") is not None:
            scaler.load_state_dict(ck["scaler"])
        rng = ck.get("rng")
        if rng is None:
            print("resume: checkpoint carries no RNG state -- continuing with fresh RNG")
        else:
            torch.set_rng_state(rng["torch"].cpu())
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
            np.random.set_state(rng["numpy"])
            if rng.get("motion") is not None:
                if motion_gen is None:
                    motion_gen = torch.Generator()
                motion_gen.set_state(rng["motion"].cpu())
        loss_log_prev = ck.get("loss_log", [])
        start_it = int(ck.get("iter", 0))
        print(f"resumed {ck_path} at iter {start_it} -> training to {args.iters}")
        if start_it >= args.iters:
            raise SystemExit(f"--iters {args.iters} is not beyond the checkpoint's {start_it}")
    else:
        loss_log_prev = []

    # Training and evaluation amplitudes are different objects (Thies II-B vs IV).
    mot_trans = args.train_trans_mm if args.motion_amp == "thies" else args.trans_mm
    mot_rot = args.train_rot_deg if args.motion_amp == "thies" else args.rot_deg
    _mx = "max " if args.motion_amp == "thies" else ""
    print(f"motion (peak-to-peak): train={args.motion_amp} {_mx}{mot_trans:g} mm / "
          f"{mot_rot:g} deg  |  val=fixed {args.trans_mm:g} mm / {args.rot_deg:g} deg")
    if args.bridge == "data":
        print(f"bridge: geometry  x_t = FDK(A(x; P((1-t)theta)), P_nom)  tangent={args.data_tangent}")
    else:
        print("bridge: pixel-linear  x_t = (1-t) x_0 + t x_1  (ablation arm)")

    next_idx = [None]        # the next patient, sampled one draw ahead so its native-grid
                             # volume can prefetch on a worker thread under the training steps

    def draw():
        """One bridge sample, whole volume: (x_t (1,1,D,H,W), dx (D,H,W), t, ctx).

        `ctx` is the global-context channel (x_t on the patch grid), a property of the draw
        shared by every patch cropped from it; `predict_x1_patched` rebuilds it the same way
        from the evolving x_t at inference."""
        idx = next_idx[0] if next_idx[0] is not None \
            else int(torch.randint(gen.n_slabs, (1,)).item())
        next_idx[0] = int(torch.randint(gen.n_slabs, (1,)).item())
        gen.prefetch_fine(next_idx[0])
        th = random_motion(gen.cfg.n_views, trans_mm=mot_trans, rot_deg=mot_rot,
                           amp_mode=args.motion_amp, device=dev, generator=motion_gen)[None]
        t = sample_t(1, dev)[0]
        if args.bridge == "data":
            x_t, dx = bridge_pair_data(gen, idx, t, th[0], delta=args.bridge_delta,
                                       mode=args.data_tangent)
        else:
            y = gen.simulate(idx, params_to_Pmot(th[0], gen.P_nom)[None])
            x_t, dx = bridge_pair_linear(gen, idx, t, y)
        x_t = x_t[None, None]                                    # (1,1,D,H,W)
        ctx = volume_context(x_t, (p, p, p)) if in_ch == 5 else None
        return x_t, dx, t, ctx

    cache = [draw() for _ in range(args.cache)]
    print(f"bridge cache warm ({args.cache} draws)")

    # ---- inline validation on the held-out VAL split (prior-only ODE, val_fm3d.run_validation)
    val_gen = None
    if args.val_every > 0:
        val_gen = CQ500Generator(args.root, cfg, device=dev, split="val",
                                 shape=tuple(args.shape), voxel_mm=1.0, verbose=False,
                                 sim_native=(args.sim_grid == "native"))
    val_dir = os.path.join(args.out, "val")
    os.makedirs(val_dir, exist_ok=True)
    writer = None
    if not args.no_tb:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(args.out, "tb"))
        print(f"tensorboard: tensorboard --logdir {os.path.join(args.out, 'tb')}")

    t0 = time.time()
    loss_log = list(loss_log_prev)          # (iter, loss) per step, saved into every checkpoint
    pending: list[tuple[int, torch.Tensor]] = []   # detached GPU losses awaiting ONE host sync
    n_nonfinite = 0

    def flush_losses():
        nonlocal n_nonfinite
        if not pending:
            return
        vals = torch.stack([v for _, v in pending]).cpu()
        nf = int((~torch.isfinite(vals)).sum())
        if nf:
            n_nonfinite += nf
            print(f"it {pending[-1][0]:6d} | WARN non-finite loss x{nf} in the last "
                  f"{len(pending)} steps ({n_nonfinite} total; fp16 overflow?)", flush=True)
        loss_log.extend((i, float(v)) for (i, _), v in zip(pending, vals))
        pending.clear()

    for it in range(start_it + 1, args.iters + 1):
        if args.lr_final is not None:
            for g in opt.param_groups:
                g["lr"] = lr_at(it)

        if it % args.refresh == 0:
            cache[torch.randint(len(cache), (1,)).item()] = draw()

        # Sample (cache entry, origin) pairs, then group by entry so the tile assembly runs once
        # per distinct draw. The loss is a mean over the batch, so row order is free.
        picks = [(int(torch.randint(len(cache), (1,)).item()),
                  int(torch.randint(len(ok), (1,)).item())) for _ in range(args.batch)]
        groups: dict[int, list[int]] = {}
        for ci, oi in picks:
            groups.setdefault(ci, []).append(oi)
        xs, ds, ts = [], [], []
        for ci, ois in groups.items():
            x_t, dx, t, ctx = cache[ci]
            coords = [ok[oi] for oi in ois]
            xs.append(make_tile_inputs(x_t, coords, (p, p, p), ctx))   # (len(ois),C,p,p,p)
            ds += [dx[z:z + p, yy:yy + p, xx:xx + p] for (z, yy, xx) in coords]
            ts += [t] * len(ois)
        xb = torch.cat(xs, 0)                                   # (B,in_ch,p,p,p)
        db = torch.stack(ds)[:, None]                           # (B,1,p,p,p) -- target: ch 0 only
        tb = torch.stack(ts)

        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=getattr(torch, args.amp_dtype), enabled=args.amp):
            loss = ((net(xb, tb) - db) ** 2).mean()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        with torch.no_grad():
            d = min(args.ema, (1 + it) / (10 + it))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm, alpha=1 - d)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)

        pending.append((it, loss.detach()))
        if it % 50 == 0:
            flush_losses()
            recent = [l for _, l in loss_log[-50:]]
            ma = sum(recent) / len(recent)
            lr_now = opt.param_groups[0]["lr"]
            lr_str = f" | lr {lr_now:.2e}" if args.lr_final is not None else ""
            print(f"it {it:6d} | loss {loss_log[-1][1]:.5f} | ma50 {ma:.5f}{lr_str} | "
                  f"{(time.time() - t0) / max(it - start_it, 1):.2f}s/it", flush=True)
            if writer is not None:
                writer.add_scalar("train/fm_loss", loss_log[-1][1], it)
                writer.add_scalar("train/fm_loss_ma50", ma, it)
                writer.add_scalar("train/lr", lr_now, it)

        if val_gen is not None and it % args.val_every == 0:
            ema.eval()
            run_validation(ema, val_gen, meas, val_dir, it=it, patients=args.val_patients,
                           patch=args.patch, ode_steps=args.val_ode_steps,
                           trans_mm=args.trans_mm, rot_deg=args.rot_deg, writer=writer, dev=dev)
            ema.train()

        if it % args.save_every == 0 or it == args.iters:
            flush_losses()
            ck = {"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                  "scaler": scaler.state_dict() if args.amp else None,
                  "iter": it, "args": {**vars(args), "amp_units": AMP_UNITS},
                  "rng": {"torch": torch.get_rng_state(),
                          "cuda": torch.cuda.get_rng_state_all(),
                          "numpy": np.random.get_state(),
                          "motion": motion_gen.get_state() if motion_gen is not None else None},
                  "loss_log": loss_log}
            torch.save(ck, os.path.join(args.out, f"ckpt_iter{it:06d}.pth"))
            torch.save(ck, os.path.join(args.out, "ckpt_last.pth"))
            print(f"saved ckpt_iter{it:06d}.pth")
            if args.keep_ckpts > 0:
                numbered = sorted(glob.glob(os.path.join(args.out, "ckpt_iter*.pth")))
                for f in numbered[:-args.keep_ckpts]:
                    os.remove(f)
                    print(f"pruned {os.path.basename(f)}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
