"""Retrain the W3DM wavelet-domain diffusion prior on OUR CQ500 train split (A2, 2026-08-13).

NOT upstream code -- De Paepe et al. released inference only (run_jrm_adm.py + weights); this
training loop is OURS, reconstructed from what their inference provably assumes:

  * the model is an x0-PREDICTOR in the wavelet domain: `model_inference` does
    x0 = model(xt, t) and clips to [-1, 1] via the image domain
    (adaptative_diffusion_sampler.py) -- so the loss below regresses W(x0), not epsilon;
  * the state lives in W(image) space with the LLL band scaled by 1/3 (physics.Wavelet);
  * the noise is drawn in IMAGE space and wavelet-transformed (`sample_one_step`:
    eps = W(randn_like(image)); run_jrm_adm starts xt = W(randn_like(x))). W is linear, so
    training draws xt = W(sqrt(a)*x0 + sqrt(1-a)*eps) with eps ~ N(0,1) in image space;
  * linear betas beta_start=1e-4, beta_end=0.02, T=1000 (config/adm_jrm.yaml), intensity in
    [-1, 1] over the [-1000, 2000] HU window (dataloaders.get_minus_one_one_norm_hu_transform).

Recipe (corrected 2026-09-07 -- the 08-13 docstring wrongly said these were unpublished): the
paper states "Adam optimizer for approximately 1.2 million iterations, using rotations and
translations for data augmentation"; the released dataloaders.py still carries the training
augmentation itself (`transform_train_head`: tio.RandomAffine scales 0.9-1.1, +-15 deg,
+-10 mm, p=0.75, after the [-1,1] HU normalisation). Batch size / lr / EMA are unpublished;
their UNet is WDM's (Friedrich 2024, itself guided-diffusion), whose defaults we take: lr 1e-4
(guided-diffusion), EMA 0.9999. `--augment` switches the released augmentation on; the
08-13/08-15 run (weights_retrain/) was 300k iters, AdamW 1e-4, EMA 0.999, no augmentation.
Data: data/train_volumes/*.npy = OUR 150-patient train split exported by
scripts/export_w3dm_train.py (raw HU, their 160x192x192 superior-first orientation), so the
retrained prior has ZERO overlap with our test cohort -- the released weights' 263-volume
split is unknown and may leak into our test30, which is the whole reason this file exists.

    cd refs/jrm-adm && python train_w3dm.py --out weights_retrain [--iters 300000]
"""
import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from src.physics.physics import Wavelet
from src.utils.creator_utils import create_model
from src.utils.dataloaders import CTDataset, get_minus_one_one_norm_hu_transform
from src.utils.yaml_config import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_volumes")
    ap.add_argument("--out", default="weights_retrain")
    ap.add_argument("--iters", type=int, default=300000)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument("--amp", action="store_true", help="fp16 autocast (released model is fp32)")
    ap.add_argument("--resume", default=None, help="a train ckpt .pth to continue from")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--augment", action="store_true",
                    help="their released training augmentation (dataloaders.transform_train_head)")
    ap.add_argument("--opt", choices=["adam", "adamw"], default="adamw",
                    help="paper says Adam; the 08-13 run used AdamW")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)

    dcfg = load_yaml("config/adm_jrm.yaml")["diffusion"]
    T = int(dcfg["diffusion_timesteps"])
    beta = torch.linspace(float(dcfg["beta_start"]), float(dcfg["beta_end"]), T, device=dev)
    abar = torch.cumprod(1.0 - beta, dim=0)                       # (T,)

    if args.augment:
        from src.utils.dataloaders import transform_train_head
        tf = transform_train_head
    else:
        tf = get_minus_one_one_norm_hu_transform()
    ds = CTDataset(root_dir=args.data, transform=tf)
    assert len(ds) > 0, f"no .npy volumes under {args.data} -- run export_w3dm_train.py first"
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                    persistent_workers=True)
    print(f"{len(ds)} train volumes | T={T} | batch {args.batch} | lr {args.lr:g} | "
          f"iters {args.iters} | amp {args.amp} | opt {args.opt} | ema {args.ema} | "
          f"augment {args.augment}")

    # UNetModel.to() is overridden upstream and returns None (w3dm.py:1255) -- never chain it.
    model = create_model()
    model.to(dev)
    model.train()
    ema = create_model()
    ema.to(dev)
    ema.eval()
    opt = (torch.optim.Adam if args.opt == "adam" else torch.optim.AdamW)(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    start_it, loss_log = 0, []
    if args.resume:
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"]); start_it = ck["iter"]; loss_log = ck["loss_log"]
        print(f"resumed at iter {start_it}")
    else:
        ema.load_state_dict(model.state_dict())
    for q in ema.parameters():
        q.requires_grad_(False)

    wavelet = Wavelet()
    it, t0 = start_it, time.time()
    while it < args.iters:
        for x in dl:                                              # (B,1,160,192,192) in [-1,1]
            if it >= args.iters:
                break
            it += 1
            x = x.to(dev, non_blocking=True).float()
            t = torch.randint(0, T, (x.shape[0],), device=dev)
            a = abar[t][:, None, None, None, None]
            eps = torch.randn_like(x)
            xt = wavelet.transform(torch.sqrt(a) * x + torch.sqrt(1.0 - a) * eps)
            target = wavelet.transform(x)

            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                loss = torch.nn.functional.mse_loss(model(xt, t), target)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                d = min(args.ema, (1 + it) / (10 + it))
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(d).add_(pm, alpha=1 - d)
            loss_log.append((it, float(loss.detach())))

            if it % 100 == 0:
                ma = sum(v for _, v in loss_log[-100:]) / min(100, len(loss_log))
                print(f"it {it:6d} | loss {loss_log[-1][1]:.5f} | ma100 {ma:.5f} | "
                      f"{(time.time() - t0) / max(it - start_it, 1):.2f}s/it", flush=True)
            if it % args.save_every == 0 or it == args.iters:
                ck = {"model": model.state_dict(), "ema": ema.state_dict(),
                      "opt": opt.state_dict(), "iter": it, "loss_log": loss_log,
                      "args": vars(args)}
                torch.save(ck, os.path.join(args.out, "ckpt_last.pth"))
                # the artifact their solver loads: bare EMA weights, same file format as their
                # released model_state_dict.pth
                torch.save(ema.state_dict(),
                           os.path.join(args.out, "model_state_dict.pth"))
                print(f"saved (it {it})", flush=True)


if __name__ == "__main__":
    main()
