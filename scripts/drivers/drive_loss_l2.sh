#!/usr/bin/env bash
# IS `l2si` DOING ANYTHING AT AKIMA 5/5, OR IS PLAIN `l2` THE SAME LOSS?  (user, 2026-07-28)
#
# `l2si` was inherited from the 2D project, where it fixed a real oscillation: the flow-matching
# push and the data-residual update disagreed about the image's overall brightness, so a plain L2
# sinogram term charged that scale drift to the motion parameters. The user's doubt is that 3D may
# not have that failure mode at all.
#
# ALREADY MEASURED (scripts/exp_loss_l2_geom.py, val 0 seed 3, the deployed 2 mm estimation grid),
# on every reference image the loop actually hands the estimator -- the cold FDK, the carried x_t
# at each ODE step, and the GT:
#     optimal scale c = <p,y>/<p,p>  =  1.0007 +- 0.0001   (24-view minibatches, range 0.997-1.005)
#     cos(grad_l2, grad_l2si)        >= 0.997 for every in-loop reference
#     |grad_l2| / |grad_l2si|        =  1.00 +- 0.01
# i.e. the scale mismatch l2si exists to absorb is 0.07%, and the two losses produce the same
# gradient. The prediction is a DEAD TIE. Nothing here is expected to move; the point is to have
# measured it, at both the estimator level and end-to-end.
#
# PHASE 1 (decisive, ~3.2 h): the in-loop A/B at the DEPLOYED operating point, val 0/1/2 x seeds
# 3/7/11, everything at its new default and ONLY --loss changed. The l2si arm already exists as
# data/runs/akima55/c2f_v*, so only the l2 arm is run. 3 patients because a single run cannot
# resolve <0.003 SSIM (the loop is not run-to-run deterministic; see the memory).
# lr is held at 3e-3 for BOTH arms, which the norm ratio above licenses -- and Adam rescales per
# parameter anyway, so a global gradient-scale change is close to invisible to it.
#
# PHASE 2 (mechanism, ~2 h): the loop-free bench, `--suite loss`, sweeping lr on BOTH losses at
# the deployed budget (20,000 iterations = N50 x PER400) on the 2 mm grid. Two references: `xt`
# (what the late loop hands the estimator, where theta accuracy is actually decided) and `cold`
# (step 0). Sweeping lr is not optional -- a loss compared at one fixed lr is a confound.
set -u
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
CK=logs/fm3d_cq500/ckpt_iter500000.pth
cd /home/mirlab/Desktop/Flow_matching_motion_3D
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while pgrep -f "run_posterior3d.py|exp_est_sweep.py" > /dev/null; do sleep 60; done
echo "=== l2 vs l2si  starts $(date) ==="

# ---- PHASE 1: in-loop, everything default except the loss -----------------------------------
for i in 0 1 2; do
    case $i in 0) S=3;; 1) S=7;; 2) S=11;; esac
    OUT=data/runs/akima55/l2_v$i
    if [ -f "$OUT/result.pt" ]; then echo "skip $OUT (done)"; continue; fi
    echo "--- l2_v$i  (val $i seed $S)  $(date) ---"
    $PY scripts/run_posterior3d.py --ckpt $CK --split val --run $i --seed $S \
        --loss l2 --out $OUT > $OUT.log 2>&1
    echo "--- l2_v$i done $(date) ---"
done

$PY scripts/cmp_runs.py data/runs/akima55/c2f_v0 data/runs/akima55/c2f_v1 data/runs/akima55/c2f_v2 \
    data/runs/akima55/l2_v0 data/runs/akima55/l2_v1 data/runs/akima55/l2_v2 \
    2>&1 | tee data/runs/akima55/cmp_loss_l2.txt

# ---- PHASE 2: loop-free bench, lr swept on both losses --------------------------------------
for REF in xt cold; do
    $PY scripts/exp_est_sweep.py --suite loss --ref $REF --coarse 2 --iters_per_config 20000 \
        --chunk 250 --xt_from data/runs/akima55/c2f_v0/result.pt \
        > data/runs/akima55/est_sweep/loss_$REF.log 2>&1
    echo "--- bench ref=$REF done $(date) ---"
done

echo "=== l2 vs l2si  ALL DONE $(date) ==="
