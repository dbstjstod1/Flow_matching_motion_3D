"""Paired (motion-affected reconstruction, VIF* map) samples for the quality-metric net.

TMI L479-497:

    "To train the quality metric network, we dynamically sample a new random motion perturbation
     each time a sample is used for training and apply it to the projection matrices. The
     spline-based motion model with 10 nodes per spline is used with a maximal amplitude of
     10 mm for translation and 15 deg for rotation. ... All splines are individually zero-centered
     and translated into perturbed projection matrices using the motion model. The filtered
     projection data is reconstructed from these perturbed matrices and the training target VIF*
     is computed from the perturbed and unperturbed reconstruction."

Three things that follow from that last sentence and are easy to get wrong:

  1. the VIF reference is the **unperturbed RECONSTRUCTION**, not the ground-truth volume. Both
     sides of the pair therefore carry the same cone-beam and discretization artifacts, and the
     target isolates motion alone.
  2. the perturbation is redrawn **every time a sample is used**, so this is a generator, not a
     fixed dataset.
  3. the motion is **zero-centred per spline**, which is the gauge convention -- a constant
     offset of the whole trajectory is unobservable (see `fm3d.rigid_motion.zero_centre_gauge`
     and the blind-motion gauge note in the repo docs), so leaving it in would make the target
     depend on an unrecoverable quantity.

AMPLITUDE: ours, not theirs -- see PROVENANCE.md section 3. The defaults here are the deployed
prior's training amplitude (15 mm / 20 deg peak-to-peak, per-DoF unequal), i.e. 1.5x / 1.33x the
paper's, so the baseline is trained on the same distribution our prior was.

THE SINOGRAM IS THE ONE OUR OWN PIPELINE USES. `CQ500Generator.simulate` projects the NATIVE
0.4187 mm volume through the LEAP operator, exactly as in `scripts/train_fm3d.py` and
`scripts/run_posterior3d.py`. Only the reconstruction differs between the two methods, which is
the entire point of the benchmark.
"""

from __future__ import annotations

import os

import torch

from fm3d.dataset_cq500 import CQ500Generator
from fm3d.geometry_3d import ConeBeam3DConfig
from fm3d.rigid_motion import akima_motion, params_to_Pmot

from .recon import ThiesConeRecon, VolumeGrid, to_unit
from .vif import vif_star_map_3d

__all__ = ["QMSampleSource", "TRAIN_AMP", "THIES_TRAIN_AMP", "EVAL_AMP", "THIES_EVAL_AMP",
           "FINE_WORKERS"]


# HOW MANY NATIVE VOLUMES LOAD CONCURRENTLY, and therefore how far ahead callers must prefetch.
# The two numbers are the same number and must not drift, which is why the prefetch depth is
# read from here rather than hard-coded in the training script.
#
# Measured on this box (2026-08-04), batch 16:
#   0 workers (no prefetch at all)   36.4 s/it   45% GPU duty -- every sample blocks 1.231 s
#   1 worker,  depth 1               22.5 s/it   87% duty     -- LOADER-bound (load 1.23 s > GPU 0.9 s)
#   2 workers, depth 2               21.4 s/it                 -- then the GPU work fell (below)
#   2 workers, depth 2, faster GPU   16.6 s/it   83% duty, 5 zeros/min
#   4 workers, depth 4 + cross-batch 15.0 s/it   96% duty, 0 zeros/min
# Cost is one page-locked 612^3 volume (~0.9 GB host RAM) per worker; the box has 227 GB free
# and 128 cores at load average 11, so neither RAM nor CPU is anywhere near the constraint.
#
# RAISED 2 -> 4 ON 2026-08-04, for VARIANCE, not throughput. The same day's kernel work (fp32
# VIF, fast Thies backprojector) cut GPU work from ~1.34 to ~1.04 s/sample, and the duty cycle
# promptly fell to 83% with 5 full-zero seconds per minute. Two workers still supply a sample
# every 0.77 s -- faster than the 1.04 s demand -- so the average was never the problem: a load
# costs 1.23-2.21 s (mean 1.55) and a depth-2 queue gives a slow one only 2 x 1.04 = 2.09 s of
# slack. Depth 4 gives it 4.2 s. The other half was that the prefetch did not cross the batch
# boundary, so every batch began by blocking on a cold load; see
# `scripts/bench_thies_train_qm.draw_batch`.
#
# WHAT IT BOUGHT, MEASURED ON THE SETTLED RUN: duty 83% -> 96%, hard stalls 5 -> 0 per minute,
# step 16.6 -> 15.0 s/it (-10%). Read the SETTLED number, not the first ten iterations: the
# script prints a CUMULATIVE average (`el / (it - it0)`), so a fresh process reads ~19 s/it at
# iter 1 and only converges after ~50 iterations -- comparing two runs at iter 2040 compares two
# warmups, which is exactly the mistake made once here on 2026-08-04.
#
# There is little left. At 96% duty the residual is not prefetch-recoverable, and the remaining
# per-sample GPU work is the native 612^3 forward projection plus one Thies backprojection
# (315 ms at 128^3, itself already 1.3x faster since the fast kernels) plus the ramp filter of a
# 360x500x700 sinogram -- all three are protocol, not overhead. Do not raise this number further
# expecting throughput.
FINE_WORKERS = 4


# Peak-to-peak, `fm3d.rigid_motion.AMP_UNITS`. See PROVENANCE.md section 3 for the ledger.
TRAIN_AMP = dict(trans_mm=15.0, rot_deg=20.0, amp_mode="thies")     # OURS (deployed prior)
THIES_TRAIN_AMP = dict(trans_mm=10.0, rot_deg=15.0, amp_mode="thies")   # the paper's
EVAL_AMP = dict(trans_mm=10.0, rot_deg=10.0, amp_mode="fixed")      # OURS (2x the paper)
THIES_EVAL_AMP = dict(trans_mm=5.0, rot_deg=5.0, amp_mode="fixed")      # the paper's


class QMSampleSource:
    """Draws `(volume_unit, vif_star_map)` pairs on the fly. One patient per call.

    The motion-free reconstruction of each patient is a fixed quantity, so it is computed once
    and cached to disk (8 MB per patient at 128^3). Without that cache every sample would pay
    TWO native-grid simulations instead of one, and the static one would be identical every time.
    """

    def __init__(self, root: str, cfg: ConeBeam3DConfig, *, split: str = "train",
                 device="cuda", grid: VolumeGrid | None = None,
                 recon: ThiesConeRecon | None = None,
                 amp: dict | None = None, n_nodes: int = 10,
                 cache_dir: str = "data/bench_thies_cache",
                 shape=(256, 256, 256), split_counts=(150, 50, 120),
                 fine_workers: int = FINE_WORKERS, verbose: bool = True):
        self.gen = CQ500Generator(root, cfg, device=device, split=split,
                                  shape=tuple(shape), voxel_mm=1.0, sim_native=True,
                                  split_counts=tuple(split_counts),
                                  fine_workers=fine_workers, verbose=verbose)
        self.cfg = cfg
        self.device = device
        self.grid = grid or VolumeGrid.centred(128, 2.0)
        self.recon = recon or ThiesConeRecon(cfg)
        self.amp = dict(amp or TRAIN_AMP)
        self.n_nodes = int(n_nodes)
        self.cache_dir = os.path.join(cache_dir, f"{split}_"
                                                 f"{'x'.join(map(str, self.grid.shape))}_"
                                                 f"{self.grid.spacing[0]:g}mm")
        os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self) -> int:
        return len(self.gen.records)

    def prefetch(self, idx: int) -> None:
        """Start loading patient `idx`'s NATIVE 612^3 volume on the dataset's worker thread.

        WHY THIS IS NOT OPTIONAL HERE. `CQ500Generator.volume_fine` keeps a SINGLE-SLOT RAM
        cache, and this sampler draws a random patient every sample, so the slot misses
        essentially always (1 in 150). Measured on this box: cache hit 43 ms, **cache miss
        1.231 s**, entirely synchronous CPU/disk work with the GPU idle -- against ~0.9 s of
        actual GPU work per sample. That is the 45% duty cycle observed on the first run
        (`100 0 0 0 99 0 100 0 100 13 ...` sampled every 3 s).

        `prefetch_fine` is the fix the TRAINER already shipped for the identical symptom
        (commit 3fafd9a, "6 s-periodic GPU 0% dips"); the bench simply never called it. It is
        a no-op when the volume is cached or already in flight, and its futures live in a dict
        keyed by patient, so prefetching one ahead cannot clobber the one being consumed.
        """
        self.gen.prefetch_fine(idx)

    # -- the motion-free half, cached -----------------------------------------------------
    @torch.no_grad()
    def static_recon(self, idx: int) -> torch.Tensor:
        """(D,H,W) mu. Thies' `I_ref`, and separately the reference every metric in the repo is
        quoted against (see the metrics note in the repo docs: our numbers are vs the GT volume,
        Thies' are vs the motion-free static reconstruction)."""
        pid = self.gen.records[idx % len(self)]["patient"]
        f = os.path.join(self.cache_dir, f"static_p{pid:04d}.pt")
        if os.path.exists(f):
            return torch.load(f, map_location=self.device)
        y = self.gen.simulate(idx, self.gen.P_nom[None])
        v = self.recon(y, self.gen.P_nom, self.grid)
        torch.save(v.cpu(), f)
        return v

    # -- one training pair -----------------------------------------------------------------
    @torch.no_grad()
    def sample(self, idx: int, *, generator: torch.Generator | None = None, seed=None):
        """-> dict(vol=(1,1,D,H,W) in ~[0,1], target=(1,1,D,H,W) VIF*, theta=(V,6), idx=int)."""
        theta = akima_motion(self.cfg.n_views, n_nodes=self.n_nodes,
                             device=self.device, generator=generator, seed=seed,
                             zero_centre=True, **self.amp)
        y = self.gen.simulate(idx, params_to_Pmot(theta, self.gen.P_nom)[None])
        moving = self.recon(y, params_to_Pmot(theta, self.gen.P_nom), self.grid)
        static = self.static_recon(idx).to(self.device)

        dist = to_unit(moving)[None, None]
        ref = to_unit(static)[None, None]
        return dict(vol=dist, target=vif_star_map_3d(dist, ref), theta=theta, idx=int(idx))
