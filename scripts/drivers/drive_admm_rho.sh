#!/usr/bin/env bash
# ADMM rho sweep — the knob my first sweep held fixed, and probably the one that mattered.
#
# WHAT WENT WRONG. I set rho = 4.3e4 to "balance the blocks" of A^T A + rho D^T D (measured
# ||A^T A|| = 5.04e5, ||D^T D|| = 11.7). At that setting ADMM's carried x_t came out 3.5 dB BELOW
# the kappa-blend it was meant to replace (33.52 vs 37.00 dB, val 0). Balance is the wrong target:
# at the first outer step d = u = 0, so the x-update solves (A^T A + rho D^T D) z = A^T y, which
# with rho D^T D ~ A^T A is a HEAVILY Tikhonov-smoothed least squares. The dual claws that back
# over subsequent steps, but with one 5-iteration CG sweep per ODE step it never fully recovers.
# The TV block should be a PERTURBATION of the data block, not its equal.
#
# Reference point: DDS ships rho = 10 on ~[0,1] data. We cannot copy the number (our operator
# normalization differs -- ||A^T A|| = 5.04e5 here) but we can copy the SPIRIT: rho D^T D should
# sit well below A^T A. This sweep spans two decades below the balanced value:
#     rho 4.3e2  -> rho||D^T D|| ~ 1% of ||A^T A||
#     rho 4.3e3  -> ~10%
#     rho 4.3e4  -> ~100%  (already run: data/admm_t3e4_v0, the balanced/too-strong end)
#
# Threshold is held at 3e-4 (mu units, ~0.5% of our [0,0.06] range) so this isolates rho. The
# earlier thresh sweep at the bad rho is kept on disk but should NOT be read as a threshold curve.
#   GPU=0 bash scripts/drivers/drive_admm_rho.sh
set -u
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
CK=logs/fm3d_cq500/ckpt_iter500000.pth
cd /home/mirlab/Desktop/Flow_matching_motion_3D
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE="--ckpt $CK --split val --run 0 --seed 3 --est_band fullband --loss l2si \
      --n_steps 50 --per 50 --views_per_iter 24 --theta_avg 1 \
      --dc_op admm --cg_iters 5 --admm_thresh 3e-4"

run () {
    local out=$1; shift
    if [ -f "data/$out/result.pt" ]; then echo "SKIP $out (done)"; return; fi
    echo "=== $out  $(date) ==="
    $PY scripts/run_posterior3d.py $BASE --out "data/$out" "$@"
}

run admm_r4e2_v0 --admm_rho 4.3e2      # TV ~1% of the data block
run admm_r4e3_v0 --admm_rho 4.3e3      # ~10%

echo "=== ADMM RHO SWEEP DONE  $(date) ==="
