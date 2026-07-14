"""Train the 3D flow-matching prior on the GEOMETRY BRIDGE.

THE BRIDGE. Instead of interpolating between a corrupted image and a clean one in pixel space,
every intermediate state is a genuine FDK reconstruction under a PARTIALLY CORRECTED geometry:

    x_t  = FDK(y, P_nom @ T(t * theta))        t = 0 -> the uncorrected recon; t = 1 -> the clean one
    dx_t = d/dt x_t                            estimated by a central difference through FDK

so the prior is trained on exactly the manifold the inference loop walks: the set of FDK images
reachable by some geometry. That is the point -- a linear pixel-space bridge passes through images
that no geometry produces, and the prior then spends its capacity on states inference never visits.

`t * theta` is a geodesic because the rotation is an axis-angle vector (`rigid_motion.so3_exp`);
with Euler angles it would not be, and the velocity target would quietly stop pointing where the
inference ODE travels.

MEMORY. The network only ever sees 64^3 PATCHES; the operator (3 FDKs per sample, under
`no_grad`) runs on the full slab. That split is the whole design -- it is what lets a 3D prior
train on a 24 GB card. Bridge draws are expensive, so a rolling cache of volume pairs is refreshed
every `--refresh` steps and each batch mixes patches from several cached draws (so `t` varies
within a batch). Both tricks are from the 4DCT project.

    python scripts/train_fm3d.py --iters 20000 --out logs/fm3d_a
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.dataset_slab import AAPMSlabGenerator
from fm3d.geometry_3d import ConeBeam3DConfig, measured_region_mask
from fm3d.prior_patch import make_tile_inputs, volume_context
from fm3d.rigid_motion import params_to_Pmot
from fm3d.unet_3d import UNet3D

DATA = "/home/mirlab/Desktop/Flow_matching_motion/data/AAPM_head_data"


def sample_t(n: int, device) -> torch.Tensor:
    """t ~ U[0,1], but 30% of draws are pulled into [0, 0.15].

    The inference ODE STARTS at t=0 -- the uncorrected FDK, the most corrupted state there is --
    and that is where a uniform sampler puts the fewest examples per unit of image change. The
    4DCT project biases the same way.
    """
    t = torch.rand(n, device=device)
    lo = torch.rand(n, device=device) < 0.30
    return torch.where(lo, torch.rand(n, device=device) * 0.15, t)


@torch.no_grad()
def bridge_pair(gen, t: torch.Tensor, y, theta, dlt, delta: float = 0.02):
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

    The anchor is a first-order detrend that costs nothing and fixes the endpoint EXACTLY:
      * at t=0 the Delta term vanishes -> x_0 = FDK(y, P_nom), the inference cold start, to the bit
      * at t=1 -> x_1 = x_anchor, the clean image, BY CONSTRUCTION
      * in between the geometry term still dominates, so the path stays the manifold of
        partially-corrected reconstructions -- which is what the inference loop actually walks as
        theta_hat converges. (The alternative -- attenuating the motion in the DATA,
        y_t = A(x; P((1-t)theta)) -- also lands clean, but its intermediate images are those of a
        patient who moved LESS, which inference never sees.)
      * Delta is constant in t, so the velocity target is just  d/dt FDK(y,P(t*theta)) + Delta.
        No extra reconstructions.

    This is the image-domain twin of Flowmatching-4DCT's `t1_anchor` detrend (which pins its
    sinogram bridge to a REAL static scan). `dlt=None` restores the bare bridge.

    dx_t is a central difference through the FDK -- three reconstructions per sample. An exact
    forward-mode derivative w.r.t. Pmat is possible but ~3x the cost, and the 2D project measured
    the finite difference to agree with it to ~1e-3.
    """
    def fdk_at(s: float):
        P = params_to_Pmot(s * theta, gen.P_nom)[None]
        return gen.to_net(gen.fdk(y, P)[0])                          # (D,H,W)

    tv = float(t)
    sp, sm = min(tv + delta, 1.0), max(tv - delta, 0.0)
    x_t = fdk_at(tv)
    dx = (fdk_at(sp) - fdk_at(sm)) / (sp - sm)
    if dlt is not None:
        x_t = x_t + tv * dlt
        dx = dx + dlt
    return x_t, dx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default="logs/fm3d_a")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8)          # patches per step
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--cache", type=int, default=6)          # bridge draws held at once
    ap.add_argument("--refresh", type=int, default=12)       # steps between refreshing one draw
    ap.add_argument("--slab", type=int, default=64)
    ap.add_argument("--in_plane", type=int, default=256)
    ap.add_argument("--views", type=int, default=360)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--context", default="global", choices=["global", "none"],
                    help="global = the arXiv:2512.18161 conditioning (in_ch=5); "
                         "none = the bare-patch prior (in_ch=1)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--trans_mm", type=float, default=5.0)
    ap.add_argument("--rot_deg", type=float, default=5.0)   # the literature's amplitude
    ap.add_argument("--anchor", default="gt", choices=["gt", "static", "none"],
                    help="what the bridge's t=1 endpoint IS. gt (DEFAULT) = the ground-truth "
                         "volume: the only endpoint with no reconstruction artefact in it. "
                         "static = the motion-free FDK, which still carries FDK's CONE-BEAM "
                         "artefact (34.3 dB from the GT, and 44.3 dB at the midplane -- it is "
                         "entirely a cone effect, not sampling: 1440 views buy +0.1 dB). "
                         "none = the bare geometry bridge, whose endpoint is not clean at all")
    ap.add_argument("--save_every", type=int, default=2000)
    args = ap.parse_args()

    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)

    cfg = ConeBeam3DConfig(det_bin=2, n_views=args.views)
    gen = AAPMSlabGenerator(args.data, cfg, device=dev, slab=args.slab, in_plane=args.in_plane)
    print(f"slabs: {gen.n_slabs} from {len(gen.runs)} runs | grid {gen.shape} @ "
          f"({gen.dz}, {gen.dy}, {gen.dx}) mm | fdk_scale {gen.fbp_scale:.5g}")
    print(f"detector {cfg.nv}x{cfg.nu} @ {cfg.du:.3f} mm | FOV {cfg.fov_diameter_mm():.0f} mm")

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

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    def draw():
        """One bridge sample, held whole-volume: (x_t (1,1,D,H,W), dx (D,H,W), t, ctx).

        `ctx` is the global-context channel -- x_t on the patch grid -- and it is a property
        of the DRAW, not of the patch, so it is computed once here and shared by every patch
        cropped from this volume. At inference `predict_x1_patched` rebuilds it the same way
        from the evolving x_t, so train and infer see the same channel.

        The ANCHOR (see `bridge_pair`) is also a property of the draw: one extra static forward
        projection and two extra FDKs, PER DRAW rather than per t, and the cache refreshes a draw
        only every `--refresh` steps."""
        y, th, vol = gen.sample_motion(1, trans_mm=args.trans_mm, rot_deg=args.rot_deg)
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
            else:                             # "static": the motion-free reconstruction
                y0 = gen.project(vol, gen.P_nom[None])
                x_anchor = gen.to_net(gen.fdk(y0, gen.P_nom[None])[0])
            x1_geo = gen.to_net(gen.fdk(y, params_to_Pmot(th[0], gen.P_nom)[None])[0])
            dlt = x_anchor - x1_geo
        t = sample_t(1, dev)[0]
        x_t, dx = bridge_pair(gen, t, y, th[0], dlt)
        x_t = x_t[None, None]                                        # (1,1,D,H,W)
        ctx = volume_context(x_t, (p, p, p)) if in_ch == 5 else None
        return x_t, dx, t, ctx

    cache = [draw() for _ in range(args.cache)]
    print(f"bridge cache warm ({args.cache} draws)")

    t0 = time.time()
    for it in range(1, args.iters + 1):
        if it % args.refresh == 0:
            cache[torch.randint(len(cache), (1,)).item()] = draw()

        xs, ds, ts = [], [], []
        for _ in range(args.batch):
            x_t, dx, t, ctx = cache[torch.randint(len(cache), (1,)).item()]
            z, yy, xx = ok[torch.randint(len(ok), (1,)).item()]
            xs.append(make_tile_inputs(x_t, [(z, yy, xx)], (p, p, p), ctx))   # (1,C,p,p,p)
            ds.append(dx[z:z + p, yy:yy + p, xx:xx + p])
            ts.append(t)
        xb = torch.cat(xs, 0)                                   # (B,in_ch,p,p,p)
        db = torch.stack(ds)[:, None]                           # (B,1,p,p,p) -- target: ch 0 only
        tb = torch.stack(ts)

        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
            loss = ((model(xb, tb) - db) ** 2).mean()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        with torch.no_grad():
            d = min(args.ema, (1 + it) / (10 + it))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm, alpha=1 - d)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)

        if it % 50 == 0:
            print(f"it {it:6d} | loss {float(loss.detach()):.5f} | "
                  f"{(time.time() - t0) / it:.2f}s/it", flush=True)
        if it % args.save_every == 0 or it == args.iters:
            ck = {"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                  "iter": it, "args": vars(args), "fbp_scale": gen.fbp_scale}
            torch.save(ck, os.path.join(args.out, f"ckpt_iter{it:06d}.pth"))
            torch.save(ck, os.path.join(args.out, "ckpt_last.pth"))
            print(f"saved ckpt_iter{it:06d}.pth")


if __name__ == "__main__":
    main()
