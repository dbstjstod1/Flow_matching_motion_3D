"""Train Thies' frozen autofocus objective: the 3D U-Net that regresses the VIF* quality map.

    THIS IS STAGE 1 OF 2. Stage 2 is `scripts/bench_thies_estimate.py`, which loads the
    checkpoint written here, freezes it, and runs the 100-iteration gradient descent.

WHAT THE PAPER PRESCRIBES (all of it is a default below)
--------------------------------------------------------
  L341-343   L1 loss, Adam, lr 1e-3, batch 16, input and output volumes 128 x 128 x 128
  L334-341   3D U-Net, feature maps 8^l for l=1..4, ReLU, final 1x1 conv, no final activation
  L479-489   a NEW random motion per sample use; 10-node splines; zero-centred per spline;
             amplitudes drawn per DoF from THEIR released clipped-half-normal sampler
             (`amp_mode="thies_hn"`), whose mass near zero is the paper's "perturb the data
             only slightly" clause
  L490-493   **the pair**: the MOTION-FREE projection data is reconstructed with the PERTURBED
             matrices ("The filtered projection data is reconstructed from these perturbed
             matrices"). The data is never simulated with motion in stage 1 -- getting this
             backwards trains the net on oracle reconstructions and voided the first run;
             see bench/thies/data.py point 0 and PROVENANCE.md "2026-08-06".
  L505-507   the reconstruction grid for motion estimation is 128^3 at 2 mm  <- so the net is
             trained on exactly the volumes it will later be asked to score
  L493-496   volumes affinely mapped to ~[0,1] with fixed, sample-independent constants
  L382-394   CQ500, sequential patient split 150 / 50 / 120

WHAT IS DELIBERATELY *NOT* THE PAPER
------------------------------------
The training amplitude LEVEL. `--amp ours` (the default) uses **15 mm / 20 deg peak-to-peak**
maxima -- the maxima our own prior was trained at
(`logs/fm3d_cq500_leap/ckpt_iter500000.pth`: train_trans_mm=15, train_rot_deg=20,
amp_units="p2p"), which is 1.5x / 1.33x the paper's 10 mm / 15 deg. Pass
`--amp thies` to train the published operating point instead. See bench/thies/PROVENANCE.md §3.
(The amplitude SHAPE is theirs under both flags: `amp_mode="thies_hn"`.)

WHAT THE PAPER SIMPLY DOES NOT SAY -- and what we chose instead
---------------------------------------------------------------
Two gaps, both filled by us, both flagged here so nobody later mistakes them for "Thies'
protocol":

  * **How long to train.** The paper gives the loss, the optimizer, the learning rate and the
    batch size (L341-343) and never states an epoch count. (The "lr 1e-4, Adam, 500 epochs" at
    L527-528 is their re-implementation of the *Huang et al. baseline*, not their own network --
    an easy line to misread.) So `--iters` is ours: 5000 x batch 16 = 80k distinct simulated
    scans, and the run is judged by the validation curve, not by the budget.
  * **What the 50-patient validation split was for.** The paper defines it at L390-394 and never
    returns to it -- no early stopping, no selection criterion. We sweep ALL of it every
    `--val_every` steps with one FIXED motion draw per patient, and keep `qmnet_best.pth`. See
    `validate()` for why the draws are fixed rather than resampled.

Neither choice can favour the baseline unfairly: a longer budget and a real checkpoint-selection
rule both make it stronger, which is the direction a baseline should err in.

COST, BEFORE YOU LAUNCH
-----------------------
Every sample costs ONE native-grid simulation (612^3 volume, 360 views) -- the same operator and
the same price our own trainer pays per draw, ~1 s. At `--batch 16` that is ~16-20 s per
optimizer step and the U-Net step itself is noise next to it. The motion-FREE half of each pair
is cached to disk on first touch (8 MB/patient), so it is paid once per patient, not per sample.

    --iters 5000 --batch 16  =  80k samples  ~=  530 epochs over the 150 training patients
                             ~=  22-24 h on one A6000.

Launch it the way every long job in this repo is launched (Bash background jobs are killed at
session teardown):

    setsid nohup .../python scripts/bench_thies_train_qm.py --out logs/bench_thies_qm \\
        </dev/null > logs/bench_thies_qm.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.thies.data import (EVAL_AMP, FINE_WORKERS, QMSampleSource,             # noqa: E402
                              THIES_TRAIN_AMP, TRAIN_AMP)
from bench.thies.motion import ThiesSplineMotion                                   # noqa: E402
from bench.thies.qmnet import QualityMetricUNet3D, THIES_F_MAPS                    # noqa: E402
from bench.thies.recon import ThiesConeRecon, VolumeGrid, to_unit                  # noqa: E402
from fm3d.geometry_3d import ConeBeam3DConfig                                      # noqa: E402
from fm3d.rigid_motion import (akima_motion, params_to_Pmot,                       # noqa: E402
                               reprojection_error, zero_centre_gauge)


def build_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # -- data / geometry (all of it mirrors scripts/train_fm3d.py so the two see one world) --
    ap.add_argument("--root", default="data/CQ500")
    ap.add_argument("--views", type=int, default=360,
                    help="TMI L455-458: 360 projections over a full 2*pi")
    ap.add_argument("--shape", type=int, default=256,
                    help="the CQ500 resampling grid the native simulation is cropped to; NOT the "
                         "reconstruction grid (that is --est_shape)")
    ap.add_argument("--split_counts", type=int, nargs=3, default=(150, 50, 120),
                    help="TMI L390-394, sequential and patient-level")

    # -- the estimation grid the net is trained on ------------------------------------------
    ap.add_argument("--est_shape", type=int, default=128, help="TMI L505-507")
    ap.add_argument("--est_voxel_mm", type=float, default=2.0, help="TMI L505-507")

    # -- Thies' reconstruction -------------------------------------------------------------
    ap.add_argument("--ramp", default="ramlak",
                    help='TMI L303-307, "a classical shift-invariant ramp filter". Our own '
                         'shepphann default is NOT part of their method.')
    ap.add_argument("--distance_weight", action="store_true",
                    help="A/B ONLY: add the 1/w^2 FDK distance weight their Eq. 3 and their "
                         "released kernel both omit. A run with this on is not 'Thies'.")

    # -- THE AMPLITUDE (see the module docstring) --------------------------------------------
    ap.add_argument("--amp", default="ours", choices=["ours", "thies"],
                    help="amplitude MAXIMA: 'ours' = 15 mm / 20 deg p2p (our deployed prior's "
                         "training maxima, the harder setting); 'thies' = the paper's 10 mm / "
                         "15 deg p2p. Both draw per-DoF amplitudes from THEIR released "
                         "clipped-half-normal sampler (amp_mode='thies_hn')")
    ap.add_argument("--sim_nodes", type=int, default=10,
                    help="TMI L482-484: 10 spline nodes for SIMULATED motion (the estimator uses "
                         "30; that lives in bench_thies_estimate.py)")

    # -- the network + optimizer, verbatim from L334-343 -------------------------------------
    ap.add_argument("--f_maps", type=int, nargs="+", default=list(THIES_F_MAPS),
                    help="feature maps per level. The default (8,16,32,64) reads '8^l, l=1..4' "
                         "as the exponential family the PUBLISHED PDF typesets and the author's "
                         "own fork of the cited plant-seg backbone implements "
                         "(number_of_features_per_level(8,4) = init*2^k). Pass '8 16 24 32' for "
                         "the flattened-text reading we ran until 2026-08-05, which under-"
                         "provisions the baseline 2x -- see bench/thies/qmnet.py's header")
    ap.add_argument("--norm", default="none", choices=["none", "group", "batch"],
                    help="the paper names no normalization; 'group' is the cited plant-seg block")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--iters", type=int, default=5000)

    # -- bookkeeping -------------------------------------------------------------------------
    ap.add_argument("--out", default="logs/bench_thies_qm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--val_patients", type=int, default=0,
                    help="0 = ALL of the validation split (the deployed setting). The paper "
                         "defines the 50-patient validation set (L390-394) and then never says "
                         "what it did with it, so this is OUR choice: a DETERMINISTIC sweep of "
                         "every val patient with one fixed motion draw each, which makes the "
                         "curve monotone-comparable across checkpoints and selects "
                         "qmnet_best.pth. A positive value truncates it, for smoke tests only.")
    ap.add_argument("--no_tb", action="store_true",
                    help="disable the tensorboard writer (same flag as scripts/train_fm3d.py)")

    # -- the RPE probe: the CONVERGENCE CRITERION (2026-08-06) --------------------------------
    ap.add_argument("--rpe_every", type=int, default=500,
                    help="every N iters, run the paper's OWN optimizer (Eq. 6, 100 GD steps) on "
                         "--rpe_patients fixed val scans against the CURRENT net and log the RPE "
                         "to tensorboard. Val L1 is a regression score on VIF*; what stage 2 "
                         "consumes is the gradient field it induces, and the two can decouple -- "
                         "RPE is the paper's headline metric and is what decides the training "
                         "budget. 0 disables. Cost ~2-3 min per probe at the defaults.")
    ap.add_argument("--rpe_patients", type=int, default=3,
                    help="val patients 0..N-1, probe motion seed RPE_SEED0+i -- the SAME "
                         "(split=val, run=i, seed=2000+i) triples the stage-2 stop-criterion "
                         "runs use, so a TB point is directly comparable to a "
                         "bench_thies_estimate run at the deployed 10/10 amplitude")
    ap.add_argument("--rpe_iters", type=int, default=100, help="Eq. 6 iterations per probe run")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", default=None)
    return ap.parse_args(argv)


def make_sources(a):
    cfg = ConeBeam3DConfig.thies(n_views=a.views)
    grid = VolumeGrid.centred(a.est_shape, a.est_voxel_mm)
    recon = ThiesConeRecon(cfg, ramp_window=a.ramp, distance_weight=a.distance_weight)
    amp = dict(TRAIN_AMP if a.amp == "ours" else THIES_TRAIN_AMP)
    common = dict(cfg=cfg, grid=grid, recon=recon, amp=amp, n_nodes=a.sim_nodes,
                  device=a.device, shape=(a.shape,) * 3,
                  split_counts=tuple(a.split_counts))
    tr = QMSampleSource(a.root, split="train", **common)
    va = QMSampleSource(a.root, split="val", verbose=False, **common)
    return cfg, grid, recon, tr, va


_NEXT_IDXS: list[int] = []          # the FOLLOWING batch's patients, already prefetching


def draw_batch(src, gen, n, device):
    """One training batch, with the next samples' native volumes loading under the current one.

    The patient indices are drawn UP FRONT so sample i can prefetch i+FINE_WORKERS before doing
    its own GPU work -- see `QMSampleSource.prefetch` for the measured stall (a native 612^3 load
    is 1.23-2.21 s, mean 1.55, against ~1.04 s of GPU work per sample after the 2026-08-04
    speedups).

    THE PREFETCH CROSSES THE BATCH BOUNDARY, and that is the point of `_NEXT_IDXS`. The first
    version drew one batch's indices, prefetched the first `FINE_WORKERS` of them, and then
    immediately consumed index 0 -- so EVERY batch began by blocking on a full cold load. At
    16 samples that is ~1.5 s of guaranteed idle per ~16.7 s step, and it is exactly what the
    duty-cycle trace showed (83%, with 5 full-zero seconds per minute). Drawing the next batch's
    indices while the current one is still being consumed means the loader is never idle and the
    consumer never waits on a cold slot.

    WHY THE DEPTH AND THE WORKER COUNT ARE THE SAME NUMBER: raw throughput was never the problem
    (2 workers already supply a sample every 0.77 s against a 1.04 s demand). The problem is
    VARIANCE -- a depth-2 queue gives a slow 2.2 s load only 2 x 1.04 = 2.09 s of slack, so it
    misses. `FINE_WORKERS` (now 4) sets both, which is why `bench/thies/data.py` owns it.

    NOTE the RNG consumption ORDER moves again: batch k+1's indices are now drawn before batch
    k's motions. Patients and motions are i.i.d. so the distribution is untouched, but a run
    resumed across this change is not a bit-continuation -- the same caveat the 2026-08-04
    up-front draw already carried.
    """
    global _NEXT_IDXS

    def _draw():
        return [int(torch.randint(0, len(src), (1,), generator=gen).item()) for _ in range(n)]

    idxs = _NEXT_IDXS or _draw()
    _NEXT_IDXS = _draw()
    look = idxs + _NEXT_IDXS         # one flat lookahead, so the spill needs no special case
    for j in range(min(FINE_WORKERS, n)):            # fill every loader before consuming any
        src.prefetch(idxs[j])
    vols, tgts = [], []
    for i, idx in enumerate(idxs):
        # look ahead inside this batch, then spill into the NEXT one so the pipeline never drains
        if i + FINE_WORKERS < len(look):
            src.prefetch(look[i + FINE_WORKERS])
        s = src.sample(idx, generator=gen)
        vols.append(s["vol"])
        tgts.append(s["target"])
    return torch.cat(vols).to(device), torch.cat(tgts).to(device)


VAL_SEED0 = 900_000        # patient i's fixed validation motion is seed VAL_SEED0 + i
RPE_SEED0 = 2_000          # probe patient i's motion seed -- MUST stay 2000+i: it is the seed
                           # base of the stage-2 stop-criterion runs (drive_qm_stopcrit.sh), and
                           # `make_motion("akima", seed=s)` == `akima_motion(seed=s)` bit for bit,
                           # so the TB rpe curve and a bench_thies_estimate run measure the SAME
                           # problem instance. (Test-cohort seeds are 1000+i -- never these.)


class RPEProbe:
    """Eq. 6 run END TO END on a few fixed val scans against the CURRENT net -> RPE to TB.

    Setup simulates each probe patient's corrupted scan once (native grid) and keeps the
    FILTERED sinogram on CPU (504 MB each); a probe then costs `iters` backprojections + net
    passes per patient (~40-70 s each at 128^3 on an idle A6000).

    The descent uses `torch.autograd.grad(f, mot.x)` rather than `f.backward()` ON PURPOSE:
    backward() would also accumulate gradients into the (trainable, non-frozen) net parameters,
    and although the training loop's `opt.zero_grad()` runs before the next `loss.backward()`,
    the probe must not depend on that ordering to be harmless.
    """

    def __init__(self, src, recon, grid, *, n_patients: int, iters: int, device):
        self.recon, self.grid, self.iters = recon, grid, int(iters)
        self.device = device
        self.P_nom = src.gen.P_nom
        self.n_views = recon.cfg.n_views
        self.worlds = []
        for i in range(n_patients):
            # bit-identical to build_world(split="val", run=i, seed=2000+i) at the deployed
            # 10/10 p2p: make_motion's akima path forwards to akima_motion with these defaults,
            # and amp_mode="fixed" consumes nothing from the RNG stream.
            theta = akima_motion(self.n_views, n_nodes=10, seed=RPE_SEED0 + i,
                                 zero_centre=True, device=device, **EVAL_AMP)
            y = src.gen.simulate(i, params_to_Pmot(theta, self.P_nom)[None])
            g_filt = recon.filter(y).cpu()
            del y
            init = reprojection_error(torch.zeros(self.n_views, 6, dtype=torch.float64,
                                                  device=device),
                                      theta.double(), self.P_nom.double())["rpe_mm"]
            self.worlds.append(dict(idx=i, theta=theta, g_filt=g_filt, rpe_init=init))
        self.rpe_init_mean = sum(w["rpe_init"] for w in self.worlds) / max(len(self.worlds), 1)

    def __call__(self, net, *, s0: float = 100.0, decay: float = 0.97) -> dict[str, float]:
        net.eval()
        out = {}
        rpes, rpes_zc = [], []
        for w in self.worlds:
            g = w["g_filt"].to(self.device)
            mot = ThiesSplineMotion(self.n_views, n_nodes=30, device=self.device)
            for n in range(self.iters):
                vol = self.recon.backproject(g, mot.Pmot(self.P_nom), self.grid)
                f = net.score(to_unit(vol)[None, None]).mean()
                mot.x.grad = torch.autograd.grad(f, mot.x)[0]
                mot.gd_step(s0 * decay ** n)
            th64 = mot.theta().detach().double()
            tt64 = w["theta"].double()
            P64 = self.P_nom.double()
            r = reprojection_error(th64, tt64, P64)["rpe_mm"]
            rz = reprojection_error(zero_centre_gauge(th64), tt64, P64)["rpe_mm"]
            out[f"rpe/p{w['idx']}"] = r
            rpes.append(r)
            rpes_zc.append(rz)
            del g
        out["rpe/mean"] = sum(rpes) / len(rpes)
        out["rpe/zero_centred_mean"] = sum(rpes_zc) / len(rpes_zc)
        net.train()
        return out


@torch.no_grad()
def validate(net, src, *, batch: int, limit: int = 0) -> float:
    """Mean L1 over EVERY validation patient, with ONE FIXED motion draw each.

    The paper is silent on how its 50-patient validation split was used (it defines the split at
    L390-394 and never returns to it), so this is our choice, made for one reason: a validation
    number that moves only because the WEIGHTS moved. Resampling the motion at each validation
    -- which the first version of this script did -- makes the curve jitter by more than the
    training signal, and a "best" checkpoint chosen off it is noise. Fixing one seed per patient
    (`VAL_SEED0 + i`) and sweeping all of them removes both the motion variance and the
    patient-sampling variance at a cost of ~50 s, against a 500-step validation interval of
    hours.

    NOTE this is a REGRESSION score (how well VIF* is predicted), not a motion-compensation
    score. Only `scripts/bench_thies_estimate.py` answers the latter.
    """
    n = len(src) if limit <= 0 else min(limit, len(src))
    net.eval()
    tot, cnt = 0.0, 0
    for j in range(min(FINE_WORKERS, n)):
        src.prefetch(j)
    for i0 in range(0, n, batch):
        vols, tgts = [], []
        for i in range(i0, min(i0 + batch, n)):
            if i + FINE_WORKERS < n:
                src.prefetch(i + FINE_WORKERS)   # same single-slot-cache problem as draw_batch
            s = src.sample(i, seed=VAL_SEED0 + i)
            vols.append(s["vol"])
            tgts.append(s["target"])
        v, t = torch.cat(vols), torch.cat(tgts)
        tot += float(torch.nn.functional.l1_loss(net(v), t)) * v.shape[0]
        cnt += v.shape[0]
    net.train()
    return tot / max(cnt, 1)


def main(argv=None):
    a = build_args(argv)
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)

    cfg, grid, recon, tr, va = make_sources(a)
    print(f"[bench-thies] geometry: SOD {cfg.SOD:g} / SDD {cfg.SDD:g} | {cfg.nv}x{cfg.nu} "
          f"@ {cfg.du:g} mm | {cfg.n_views} views / {cfg.angular_range_deg:g} deg")
    print(f"[bench-thies] recon grid {grid.shape} @ {grid.spacing[0]:g} mm | ramp={a.ramp} "
          f"| distance_weight={a.distance_weight} | mu scale {recon.scale:.6g}")
    print(f"[bench-thies] amplitude '{a.amp}' -> {tr.amp}   (peak-to-peak)")
    print(f"[bench-thies] train {len(tr)} patients | val {len(va)} patients")

    net = QualityMetricUNet3D(a.f_maps, norm=a.norm).to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"[bench-thies] U-Net f_maps={tuple(a.f_maps)} norm={a.norm} | {n_par/1e6:.2f} M params")

    probe = None
    if a.rpe_every:
        probe = RPEProbe(va, recon, grid, n_patients=a.rpe_patients, iters=a.rpe_iters,
                         device=dev)
        print(f"[bench-thies] RPE probe: {a.rpe_patients} val patients, seeds "
              f"{RPE_SEED0}..{RPE_SEED0 + a.rpe_patients - 1}, {EVAL_AMP['trans_mm']:g} mm / "
              f"{EVAL_AMP['rot_deg']:g} deg p2p, every {a.rpe_every} iters | "
              f"initial (theta=0) RPE {probe.rpe_init_mean:.3f} mm")

    opt = torch.optim.Adam(net.parameters(), lr=a.lr)     # L341-343: Adam, lr 1e-3
    it0 = 0
    resumed_best = float("inf")
    resumed_rng = None
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        it0 = int(ck["iter"])
        # RESTORE THE BEST-SO-FAR, or the first validation after a resume silently overwrites
        # qmnet_best.pth with a WORSE checkpoint (anything beats `inf`). Observed 2026-08-04:
        # val went 0.12150 at iter 1000 -> 0.12160 at iter 1500, and the worse one was written
        # as "best" purely because the run had been restarted for the prefetch fix in between.
        resumed_best = float(ck.get("val_l1", float("inf")))
        resumed_rng = ck.get("rng")          # None on pre-2026-08-06 checkpoints
        print(f"[bench-thies] resumed {a.resume} at iter {it0} "
              f"(best val L1 so far {resumed_best:.5f})")

    with open(os.path.join(a.out, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=1)

    writer = None
    if not a.no_tb:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(a.out, "tb"))

    # THE SAMPLING STREAM, AND WHY --resume MUST NOT JUST RE-SEED IT.
    # `g_tr` drives both the patient draw and the motion draw, and this line used to run
    # unconditionally after the resume block -- so a resumed run replayed the IDENTICAL sequence
    # from iteration 1. Extending 5000 -> 10000 would then show the network the same 5000 batches
    # twice instead of 5000 fresh ones, which is not a longer run, it is two epochs over the same
    # 80k draws. That directly contradicts the protocol being reimplemented ("we dynamically
    # sample a new random motion perturbation each time a sample is used for training", II-B.3):
    # the target's whole point is that a patient is never seen under the same motion twice.
    # Checkpoints written from here on carry the generator state; older ones do not, so for those
    # we derive a fresh, still-deterministic seed from the resume point and say so.
    g_tr = torch.Generator()
    if resumed_rng is not None:
        # the checkpoint is loaded with map_location=dev, which drags the saved RNG state onto
        # the GPU too -- set_state requires a CPU ByteTensor.
        g_tr.set_state(resumed_rng.cpu())
        print("[bench-thies] restored the sampling RNG from the checkpoint "
              "(the draw sequence continues rather than replaying)")
    else:
        g_tr.manual_seed(a.seed + 1 + it0)
        if it0:
            print(f"[bench-thies] checkpoint carries no RNG state (pre-2026-08-06 format): "
                  f"seeding the sampling stream with {a.seed + 1 + it0} = seed+1+iter so the "
                  f"continuation draws FRESH patients/motions instead of replaying 1..{it0}")
    best = resumed_best
    t0 = time.time()
    for it in range(it0 + 1, a.iters + 1):
        vol, tgt = draw_batch(tr, g_tr, a.batch, dev)
        vol.requires_grad_(False)
        pred = net(vol)
        loss = torch.nn.functional.l1_loss(pred, tgt)     # L341: L1 loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if it % 10 == 0 or it == it0 + 1:
            el = time.time() - t0
            print(f"it {it:6d}  L1 {loss.item():.5f}  "
                  f"| target mean {tgt.mean().item():.4f}  pred mean {pred.mean().item():.4f}  "
                  f"| {el/max(1, it-it0):.1f} s/it", flush=True)
            if writer is not None:
                writer.add_scalar("train/l1", loss.item(), it)
                writer.add_scalar("train/target_mean", tgt.mean().item(), it)
                writer.add_scalar("train/pred_mean", pred.mean().item(), it)

        def _save(tag):
            ck = dict(model=net.state_dict(), opt=opt.state_dict(), iter=it, args=vars(a),
                      f_maps=tuple(a.f_maps), norm=a.norm, val_l1=best,
                      rng=g_tr.get_state())   # so --resume continues the draw stream
            torch.save(ck, os.path.join(a.out, tag))

        if a.val_every and it % a.val_every == 0:
            v = validate(net, va, batch=a.batch, limit=a.val_patients)
            n_used = len(va) if a.val_patients <= 0 else min(a.val_patients, len(va))
            mark = ""
            if v < best:
                best = v
                _save("qmnet_best.pth")
                mark = "  <- best, saved qmnet_best.pth"
            print(f"it {it:6d}  VAL L1 {v:.5f}  (all {n_used} val patients, fixed draws){mark}",
                  flush=True)
            if writer is not None:
                writer.add_scalar("val/l1", v, it)
                writer.add_scalar("val/l1_best", best, it)

        if probe is not None and it % a.rpe_every == 0:
            t_p = time.time()
            scores = probe(net)
            per = "  ".join("p%d %.3f" % (w["idx"], scores["rpe/p%d" % w["idx"]])
                            for w in probe.worlds)
            print(f"it {it:6d}  RPE {scores['rpe/mean']:.3f} mm  (zc {scores['rpe/zero_centred_mean']:.3f}; "
                  f"init {probe.rpe_init_mean:.3f}; {per})  [{time.time()-t_p:.0f} s]",
                  flush=True)
            if writer is not None:
                for k, s in scores.items():
                    writer.add_scalar(k, s, it)
                writer.add_scalar("rpe/init_mean", probe.rpe_init_mean, it)

        if it % a.save_every == 0 or it == a.iters:
            _save(f"qmnet_iter{it:06d}.pth")
            _save("qmnet_last.pth")
            print(f"[bench-thies] saved qmnet_iter{it:06d}.pth", flush=True)

    if writer is not None:
        writer.close()
    print(f"[bench-thies] done in {(time.time()-t0)/3600:.2f} h. best VAL L1 {best:.5f}\n"
          f"  Next: scripts/bench_thies_estimate.py --qm {a.out}/qmnet_best.pth\n"
          f"  (evaluation amplitude {EVAL_AMP}; use qmnet_best.pth, not _last -- the paper is "
          f"silent on checkpoint selection so we select on the val split it defines)")


if __name__ == "__main__":
    main()
