"""Train the 3D flow-matching prior on the GEOMETRY BRIDGE.

THE BRIDGE. Instead of interpolating between a corrupted image and a clean one in pixel space,
every intermediate state is a genuine FDK reconstruction under a PARTIALLY CORRECTED geometry:

    x_t  = FDK(y, P_nom @ T(t * theta))        t = 0 -> the uncorrected recon; t = 1 -> the clean one
    dx_t = d/dt x_t                            EXACT, fused into the FDK's backprojection pass
                                               (--tangent fd restores the old central difference)

so the prior is trained on exactly the manifold the inference loop walks: the set of FDK images
reachable by some geometry. That is the point -- a linear pixel-space bridge passes through images
that no geometry produces, and the prior then spends its capacity on states inference never visits.

`t * theta` is a geodesic because the rotation is an axis-angle vector (`rigid_motion.so3_exp`);
with Euler angles it would not be, and the velocity target would quietly stop pointing where the
inference ODE travels.

MEMORY. The network only ever sees 64^3 PATCHES; the operator (one fused FDK+tangent pass per
sample, under `no_grad`) runs on the full slab. That split is the whole design -- it is what lets a 3D prior
train on a 24 GB card. Bridge draws are expensive, so a rolling cache of volume pairs is refreshed
every `--refresh` steps and each batch mixes patches from several cached draws (so `t` varies
within a batch). Both tricks are from the 4DCT project. `t` itself is sampled UNIFORMLY
(`sample_t`), matching arXiv:2512.18161 and the bridge-model default.

    python scripts/train_fm3d.py --iters 20000 --out logs/fm3d_a
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # for val_fm3d

from fm3d.dataset_cq500 import CQ500Generator
from fm3d.dataset_slab import AAPMSlabGenerator
from fm3d.geometry_3d import ConeBeam3DConfig, measured_region_mask
from fm3d.prior_patch import make_tile_inputs, volume_context
from fm3d.rigid_motion import bridge_P_and_dP, params_to_Pmot, random_motion
from fm3d.unet_3d import UNet3D
from val_fm3d import run_validation

DATA = "/home/mirlab/Desktop/Flow_matching_motion/data/AAPM_head_data"


def sample_t(n: int, device) -> torch.Tensor:
    """t ~ U[0,1], uniform.

    This is the STANDARD for this model family and matches our own references: the UNet
    paper we follow (arXiv:2512.18161) samples t ~ Uniform with no timestep weighting, and
    the bridge models this prior belongs to (I2SB; InDI, Delbracio & Milanfar 2023
    arXiv:2303.11435; Direct Diffusion Bridges) all default to uniform t.

    An earlier version pulled 30% of draws into [0,0.15] to over-sample the cold-start (t=0)
    endpoint where the inference ODE begins. That endpoint bias is a *validated* restoration
    trick (InDI reports biasing toward the degraded endpoint beats uniform), and the smooth,
    citable way to do it would be logit-normal sampling with negative location (SD3 / Esser
    et al. 2024, arXiv:2403.03206). It was dropped here in favour of the reference-faithful
    uniform sampler; restore it via a logit-normal draw if the cold-start region needs more
    capacity.
    """
    return torch.rand(n, device=device)


@torch.no_grad()
def bridge_pair(gen, t: torch.Tensor, y, theta, dlt, delta: float = 0.02,
                mode: str = "analytic"):
    """(x_t, dx_t) in NET space, for one volume. t: scalar tensor.

    THE ANCHORED GEOMETRY BRIDGE:

        x_t = FDK(y, P_nom @ T(t*theta))  +  t * Delta,
        Delta = x_anchor - FDK(y, P_nom @ T(theta))                  [computed once, in `draw`]

    WHY THE ANCHOR EXISTS. The bare geometry bridge's endpoint is NOT a clean image. Handing FDK
    the TRUE theta does not reproduce a static scan: FDK is an analytic inverse derived for a
    CIRCULAR, EQUIANGULAR orbit, and per-view motion breaks that. Measured on CQ500 with the
    literature's own motion model (Akima, 10 nodes, 5 mm / 5 deg), FDK(y, P(theta_true)) sits
    **1-3 dB below the static scan** even after `view_angular_weights` recovers the gantry-axis
    part. So a prior trained on the bare bridge learns FDK's residual motion artefact AS ITS
    TARGET -- it would faithfully reproduce, at t=1, an image that is not clean.

    The anchor is a first-order detrend that costs nothing and pins the endpoint:
      * at t=0 the Delta term vanishes -> x_0 = FDK(y, P_nom), the inference cold start, to the bit
      * at t=1 -> x_1 = x_anchor, the clean image. NOT bit-exact in the analytic mode: Delta is
        computed with the batched FDK kernel (`gen.fdk`, in `draw`) while x_t comes from the fused
        Triton tangent kernel (`gen.fdk_tangent`), so x_1 = x_anchor holds to the two kernels'
        gated agreement (scripts/gate_fdk_tangent.py), not by construction. mode="fd" restores
        the single-kernel, exact-cancellation path.
      * in between the geometry term still dominates, so the path stays the manifold of
        partially-corrected reconstructions -- which is what the inference loop actually walks as
        theta_hat converges. (The alternative -- attenuating the motion in the DATA,
        y_t = A(x; P((1-t)theta)) -- also lands clean, but its intermediate images are those of a
        patient who moved LESS, which inference never sees.)
      * Delta is constant in t, so the velocity target is just  d/dt FDK(y,P(t*theta)) + Delta.
        No extra reconstructions.

    This is the image-domain twin of Flowmatching-4DCT's `t1_anchor` detrend (which pins its
    sinogram bridge to a REAL static scan). `dlt=None` restores the bare bridge.

    dx_t (mode="analytic", the default) is the EXACT forward-mode derivative w.r.t. Pmat:
    one fused reconstruction pass (`fdk_conebeam_3d_tangent` -- filter once, then a Triton
    kernel that accumulates FDK and d/ds FDK from the same detector taps), replacing the old
    central difference (mode="fd"): three full FDKs per sample that agreed with the exact
    derivative only to ~1e-3. The old comment here guessed forward-mode would cost ~3x; with
    the tangent fused into the backprojection it costs ~0.5x. Gated against float64 central
    differences in scripts/gate_fdk_tangent.py; "fd" is kept as the gate's counterparty and
    an escape hatch.
    """
    tv = float(t)
    if mode == "analytic":
        P, Pdot = bridge_P_and_dP(theta, gen.P_nom, tv)
        x_mu, dx_mu = gen.fdk_tangent(y, P[None], Pdot[None])
        x_t = gen.to_net(x_mu[0])                                    # (D,H,W)
        dx = gen.to_net_tangent(dx_mu[0])                            # affine gain, no shift
    elif mode == "fd":
        def fdk_at(s: float):
            P = params_to_Pmot(s * theta, gen.P_nom)[None]
            return gen.to_net(gen.fdk(y, P)[0])                      # (D,H,W)

        sp, sm = min(tv + delta, 1.0), max(tv - delta, 0.0)
        x_t = fdk_at(tv)
        dx = (fdk_at(sp) - fdk_at(sm)) / (sp - sm)
    else:
        raise ValueError(f"unknown bridge tangent mode: {mode!r}")
    if dlt is not None:
        x_t = x_t + tv * dlt
        dx = dx + dlt
    return x_t, dx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cq500", choices=["cq500", "aapm"],
                    help="cq500 = the literature's dataset in the literature's geometry "
                         "(SID 785 / SDD 1200); aapm = the old stacked-slice stand-in")
    ap.add_argument("--root", default="data/CQ500")
    ap.add_argument("--split", default="train")
    ap.add_argument("--shape", type=int, nargs=3, default=(256, 256, 256))   # cq500, @ 1 mm
    ap.add_argument("--data", default=DATA)                                  # aapm only
    ap.add_argument("--out", default="logs/fm3d_a")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8)          # patches per step
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--cache", type=int, default=6)          # bridge draws held at once
    ap.add_argument("--refresh", type=int, default=12)       # steps between refreshing one draw
    ap.add_argument("--slab", type=int, default=64)          # aapm only
    ap.add_argument("--in_plane", type=int, default=256)     # aapm only
    ap.add_argument("--views", type=int, default=360)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--context", default="global", choices=["global", "none"],
                    help="global = the arXiv:2512.18161 conditioning (in_ch=5); "
                         "none = the bare-patch prior (in_ch=1)")
    ap.add_argument("--lr", type=float, default=1e-4)          # the 2D sibling's vanilla-FM lr
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--amp", action="store_true", default=True,
                    help="fp16 AMP, like the 2D sibling. --no-amp for fp32.")
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--amp_dtype", default="float16", choices=["float16", "bfloat16"],
                    help="float16 matches the 2D sibling; bfloat16 is safer but not what it used")
    ap.add_argument("--trans_mm", type=float, default=5.0)
    ap.add_argument("--rot_deg", type=float, default=5.0)   # the literature's amplitude
    ap.add_argument("--anchor", default="static", choices=["static", "gt", "none"],
                    help="what the bridge's t=1 endpoint IS. static (DEFAULT) = the MOTION-FREE "
                         "FDK -- the same reconstruction operator, the same scan, no motion. It "
                         "is what this scanner can actually produce of a still patient, and it is "
                         "where BOTH sibling projects anchor. Without it the endpoint is FDK(y, "
                         "P(theta_true)), which still sits 1-3 dB below a static scan, so the "
                         "prior would learn FDK's residual MOTION artefact as its target. "
                         "gt = the volume itself (also removes FDK's cone-beam floor, but asks "
                         "the net to invert the operator's own defect). none = the bare bridge.")
    ap.add_argument("--tangent", default="analytic", choices=["analytic", "fd"],
                    help="how bridge_pair gets the velocity target dx_t. analytic (DEFAULT) = "
                         "the exact d/ds FDK(y, P(s*theta)) in one fused pass "
                         "(fdk_conebeam_3d_tangent); fd = the old central difference, three "
                         "full FDKs, ~1e-3 off -- kept as an escape hatch and gate counterparty")
    ap.add_argument("--resume", default=None,
                    help="a ckpt .pth or a run dir (uses ckpt_last.pth): restore model+EMA+"
                         "optimizer+loss history+RNG and continue to --iters")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed torch+numpy at startup (default None = unseeded, the historical "
                         "behavior). Also routes the motion draw through a dedicated seeded "
                         "generator so the Akima spline nodes become reproducible too.")
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--keep_ckpts", type=int, default=0,
                    help="keep only the newest N numbered ckpt_iterNNNNNN.pth (0 = keep all, "
                         "the historical behavior). ckpt_last.pth is never pruned.")
    ap.add_argument("--val_every", type=int, default=5000)
    ap.add_argument("--val_patients", type=int, default=3)
    ap.add_argument("--val_ode_steps", type=int, default=50)   # the deploy loop's count
    ap.add_argument("--no_tb", action="store_true", help="disable the tensorboard writer")
    ap.add_argument("--compile", action="store_true", default=True,
                    help="torch.compile the velocity net (~1.30x end-to-end here: 0.54 -> 0.41 "
                         "s/it; the net fwd+bwd is 76%% of a step and compile fuses its many small "
                         "3D-conv kernels). Numerically faithful (rel-L2 1.6e-3 vs eager, within "
                         "fp16). Checkpoints save the UNDERLYING module, so resume/inference stay "
                         "plain-UNet3D compatible. --no-compile to disable.")
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    args = ap.parse_args()

    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)

    # --seed None keeps the historical unseeded behavior (and the historical RNG call sequence).
    # `motion_gen` stays None when unseeded, which is exactly what random_motion/sample_motion
    # received before it existed.
    motion_gen = None
    if args.seed is not None:
        torch.manual_seed(args.seed)                       # seeds CPU + all CUDA generators
        np.random.seed(args.seed)
        motion_gen = torch.Generator().manual_seed(args.seed + 1)
        print(f"seeded torch+numpy with {args.seed} (motion generator: {args.seed + 1})")

    if args.dataset == "cq500":
        cfg = ConeBeam3DConfig.thies(n_views=args.views)
        gen = CQ500Generator(args.root, cfg, device=dev, split=args.split,
                             shape=tuple(args.shape), voxel_mm=1.0)
        print(f"CQ500 '{args.split}': {gen.n_slabs} patients | grid {gen.shape} @ 1 mm | "
              f"fdk_scale {gen.fbp_scale:.5g}")
    else:
        cfg = ConeBeam3DConfig(det_bin=2, n_views=args.views)
        gen = AAPMSlabGenerator(args.data, cfg, device=dev, slab=args.slab,
                                in_plane=args.in_plane)
        print(f"slabs: {gen.n_slabs} from {len(gen.runs)} runs | grid {gen.shape} @ "
              f"({gen.dz}, {gen.dy}, {gen.dx}) mm | fdk_scale {gen.fbp_scale:.5g}")
    print(f"detector {cfg.nv}x{cfg.nu} @ {cfg.du:.3f} mm | FOV {cfg.fov_diameter_mm():.0f} mm | "
          f"{cfg.n_views} views")

    # Patches are drawn only from the MEASURED REGION (a barrel, not a cylinder -- it narrows
    # with radius). Training the prior on never-measured voxels teaches it to hallucinate exactly
    # where inference has no data to correct it.
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

    # GLOBAL CONTEXT (arXiv:2512.18161). A 64^3 patch of a head cannot tell whether it is
    # orbit or posterior fossa, nor what the rest of the slab looks like, so the bare-patch
    # prior can only learn LOCAL structure. Four conditioning channels close that: the whole
    # x_t resampled onto the patch grid, and the patch voxels' absolute (z,y,x) in the volume.
    # The velocity target is untouched -- the net still predicts one channel, for channel 0.
    in_ch = 5 if args.context == "global" else 1
    model = UNet3D(in_ch=in_ch, base=args.base).to(dev)
    ema = UNet3D(in_ch=in_ch, base=args.base).to(dev)
    ema.load_state_dict(model.state_dict())
    for q in ema.parameters():
        q.requires_grad_(False)
    n_par = sum(q.numel() for q in model.parameters())
    print(f"UNet3D in_ch={in_ch} (context={args.context}) base={args.base}: "
          f"{n_par / 1e6:.2f} M params")

    # torch.compile wraps the module but shares its parameters, so `model` stays the canonical
    # UNet3D for the optimizer, the EMA, and CHECKPOINTS (saving model.state_dict() keeps plain
    # keys -- no _orig_mod. prefix -- so resume and run_posterior3d/val load unchanged). `net` is
    # the compiled forward used ONLY in the training step. First few steps pay a one-off compile
    # warmup, so the printed s/it settles after ~iter 50.
    net = torch.compile(model) if args.compile else model
    if args.compile:
        print("torch.compile: ON (net fwd/bwd only; ~1.30x). --no-compile to disable.")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    # ---- RESUME ---------------------------------------------------------------------------
    # Continue a finished/killed run: restore model + EMA + OPTIMIZER STATE + the loss history,
    # and carry on from the saved iter. The optimizer state is the part that matters -- AdamW's
    # moment estimates take thousands of steps to rebuild, and dropping them makes a "resumed"
    # run behave like a fresh one with a warm init (a silent, and silently worse, restart).
    start_it = 0
    if args.resume:
        ck_path = (os.path.join(args.resume, "ckpt_last.pth")
                   if os.path.isdir(args.resume) else args.resume)
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        prev = ck.get("args", {})

        # A resumed run MUST keep the architecture and the bridge it was trained under; anything
        # else silently changes what the weights mean. `_cmp` normalizes sequences because the
        # SAME --shape arrives as a tuple when defaulted and a list when typed on the CLI.
        def _cmp(v):
            return tuple(v) if isinstance(v, (list, tuple)) else v

        for k in ("base", "patch", "context", "anchor", "shape", "views", "dataset",
                  "tangent", "trans_mm", "rot_deg"):
            if k in prev and k in vars(args) and _cmp(prev[k]) != _cmp(vars(args)[k]):
                raise SystemExit(f"--resume mismatch on '{k}': checkpoint has {prev[k]!r}, "
                                 f"this run asks for {vars(args)[k]!r}")
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        if ck.get("scaler") is not None:
            scaler.load_state_dict(ck["scaler"])
        # RNG: restore the sampling streams so a resumed run continues the data sequence the
        # interrupted one would have drawn. Old checkpoints predate these keys -- note and go on.
        rng = ck.get("rng")
        if rng is None:
            print("resume: checkpoint carries no RNG state (older format) -- continuing "
                  "with fresh RNG")
        else:
            torch.set_rng_state(rng["torch"].cpu())        # map_location may have moved these
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
            np.random.set_state(rng["numpy"])
            if rng.get("motion") is not None:
                if motion_gen is None:
                    motion_gen = torch.Generator()
                motion_gen.set_state(rng["motion"].cpu())
        loss_log_prev = ck.get("loss_log", [])
        start_it = int(ck.get("iter", 0))
        print(f"resumed {ck_path} at iter {start_it} "
              f"({len(loss_log_prev)} loss points carried over) -> training to {args.iters}")
        if start_it >= args.iters:
            raise SystemExit(f"--iters {args.iters} is not beyond the checkpoint's {start_it}")
    else:
        loss_log_prev = []

    def draw():
        """One bridge sample, held whole-volume: (x_t (1,1,D,H,W), dx (D,H,W), t, ctx).

        `ctx` is the global-context channel -- x_t on the patch grid -- and it is a property
        of the DRAW, not of the patch, so it is computed once here and shared by every patch
        cropped from this volume. At inference `predict_x1_patched` rebuilds it the same way
        from the evolving x_t, so train and infer see the same channel.

        The ANCHOR (see `bridge_pair`) is also a property of the draw. On cq500 its motion-free
        static FDK is MEMOIZED per volume (`gen.static_anchor_net`), since it depends on neither
        the motion nor t -- that removes one forward projection (~0.9 s) and one FDK (~0.18 s) from
        every draw after a volume's first, and the cache refreshes a draw only every `--refresh`
        steps. To reach the memo we sample the volume INDEX ourselves here (cq500's `volume(idx)`
        is a clean per-patient lookup); aapm slabs have no such stable key, so they keep the old
        inline path via `sample_motion`."""
        idx = None
        if args.dataset == "cq500":
            idx = int(torch.randint(gen.n_slabs, (1,)).item())
            vol = gen.volume(idx)
            th = random_motion(gen.cfg.n_views, trans_mm=args.trans_mm, rot_deg=args.rot_deg,
                               device=dev, generator=motion_gen)[None]
            with torch.no_grad():
                y = gen.project(vol, params_to_Pmot(th[0], gen.P_nom)[None])
        else:
            y, th, vol = gen.sample_motion(1, trans_mm=args.trans_mm, rot_deg=args.rot_deg,
                                           generator=motion_gen)
        dlt = None
        if args.anchor != "none":
            if args.anchor == "gt":
                # THE VOLUME ITSELF. A motion-free FDK is NOT clean -- it still carries the
                # cone-beam artefact of a circular orbit (34.3 dB from the GT here; 44.3 dB if you
                # look only at the midplane, so it IS the cone and not sampling -- quadrupling the
                # views buys 0.1 dB). That artefact is a defect of the INVERSE, not a property of
                # the data: y is the projection of the true volume, so the image the data supports
                # is the GT. Anchoring at the static FDK would teach the prior to PAINT IN cone
                # artefacts it is supposed to remove.
                x_anchor = gen.to_net(vol[0, 0])
            elif idx is not None:             # "static", cq500: the memoized motion-free recon
                x_anchor = gen.static_anchor_net(idx)
            else:                             # "static", aapm: compute inline (no stable key)
                y0 = gen.project(vol, gen.P_nom[None])
                x_anchor = gen.to_net(gen.fdk(y0, gen.P_nom[None])[0])
            x1_geo = gen.to_net(gen.fdk(y, params_to_Pmot(th[0], gen.P_nom)[None])[0])
            dlt = x_anchor - x1_geo
        t = sample_t(1, dev)[0]
        x_t, dx = bridge_pair(gen, t, y, th[0], dlt, mode=args.tangent)
        x_t = x_t[None, None]                                        # (1,1,D,H,W)
        ctx = volume_context(x_t, (p, p, p)) if in_ch == 5 else None
        return x_t, dx, t, ctx

    cache = [draw() for _ in range(args.cache)]
    print(f"bridge cache warm ({args.cache} draws)")

    # ---- validation: a held-out VAL generator + a tensorboard writer -----------------------
    # Same geometry, the VAL split (never trained on), evaluated inline every `val_every` steps.
    # Standalone `val_fm3d.py` runs the identical `run_validation`; the montages land in out/val.
    val_gen = None
    if args.dataset == "cq500" and args.val_every > 0:
        # fbp_scale is INJECTED from the train generator: _calibrate() fits it on volume 0 of the
        # generator's own split, so letting the val generator recalibrate (on val patient 0) would
        # hand validation a slightly different NET normalization than training saw. Injecting also
        # skips the calibration projection at startup.
        val_gen = CQ500Generator(args.root, cfg, device=dev, split="val",
                                 shape=tuple(args.shape), voxel_mm=1.0, verbose=False,
                                 fbp_scale=gen.fbp_scale)
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
        """Move pending losses into loss_log with a single device sync for the whole block.

        The old loop called float(loss) AND torch.isfinite(loss) every step -- two host syncs
        per iteration that serialized the CPU against the GPU. (GradScaler.step keeps its own
        internal found_inf sync; that one is inherent to fp16 AMP.) fp16 can overflow to
        inf/nan; GradScaler skips a non-finite step on its own, but a persistently non-finite
        LOSS means the run is dead and should say so, not spin silently -- the check now runs
        per flushed block instead of per step."""
        nonlocal n_nonfinite
        if not pending:
            return
        vals = torch.stack([v for _, v in pending]).cpu()          # the one sync
        nf = int((~torch.isfinite(vals)).sum())
        if nf:
            n_nonfinite += nf
            print(f"it {pending[-1][0]:6d} | WARN non-finite loss x{nf} in the last "
                  f"{len(pending)} steps ({n_nonfinite} total; fp16 overflow?)", flush=True)
        loss_log.extend((i, float(v)) for (i, _), v in zip(pending, vals))
        pending.clear()

    for it in range(start_it + 1, args.iters + 1):
        if it % args.refresh == 0:
            cache[torch.randint(len(cache), (1,)).item()] = draw()

        # Sample ALL (cache entry, origin) pairs first, in the exact per-patch RNG order the
        # old loop used (one randint over the cache, one over the origins, alternating), THEN
        # group by entry so make_tile_inputs runs once per distinct entry (<= cache size calls
        # instead of `batch`). The loss is a mean over the batch, so row order is free.
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
            # rate over THIS process's steps, not `it` -- after a resume `it` starts high and
            # dividing by it would report a nonsense s/it
            print(f"it {it:6d} | loss {loss_log[-1][1]:.5f} | ma50 {ma:.5f} | "
                  f"{(time.time() - t0) / max(it - start_it, 1):.2f}s/it", flush=True)
            if writer is not None:
                writer.add_scalar("train/fm_loss", loss_log[-1][1], it)
                writer.add_scalar("train/fm_loss_ma50", ma, it)

        if val_gen is not None and it % args.val_every == 0:
            # PRIOR-ONLY ODE from the cold start on the val split -- what the loss cannot tell us.
            # EMA weights, eval mode, then straight back to training.
            ema.eval()
            run_validation(ema, val_gen, meas, val_dir, it=it, patients=args.val_patients,
                           patch=args.patch, ode_steps=args.val_ode_steps, anchor=args.anchor,
                           trans_mm=args.trans_mm, rot_deg=args.rot_deg, writer=writer, dev=dev)
            ema.train()

        if it % args.save_every == 0 or it == args.iters:
            flush_losses()
            # loss_log rides along in the checkpoint, so the convergence curve survives even if
            # the text log is rotated or lost -- the whole per-step history, (iter, loss) pairs.
            # "rng" makes --resume continue the exact sampling streams (older ckpts lack it).
            ck = {"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                  "scaler": scaler.state_dict() if args.amp else None,
                  "iter": it, "args": vars(args), "fbp_scale": gen.fbp_scale,
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
