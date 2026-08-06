#!/bin/bash
# Phase 3, 2026-08-03: 3-patient confirmation of the two leading speed configs.
# The phase-2 `noise` arm (an IDENTICAL rerun of coarse-all/per400) measured the run-to-run
# spread at 0.51 dB / 0.0017 SSIM on x_t -- LARGER than almost every gap in the phase-1/2 table,
# so nothing in the per100..per400 band is separable on one patient. val0 is already run for both
# candidates; this adds val1 and val2 so each config gets a 3-patient mean (noise ~0.3 dB).
# Baseline for comparison = data/post500k_val{0,1,2} (c2f, per400, 949s/patient).
cd /home/mirlab/Desktop/Flow_matching_motion_3D
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
export CUDA_VISIBLE_DEVICES=0
CK=logs/fm3d_cq500_leap/ckpt_iter500000.pth
run () { tag=$1; r=$2; shift 2
  echo "##### $tag val$r :: $* #####"
  $PY scripts/run_posterior3d.py --ckpt $CK --out data/conf_${tag}_val$r --run $r "$@" \
      > logs/conf_${tag}_val$r.log 2>&1
  echo "--- $tag val$r done ---"; }
for r in 1 2; do run ca_per100 $r --est_coarse_until 1.0 --per 100; done
for r in 1 2; do run ca_per200 $r --est_coarse_until 1.0 --per 200; done
echo "PHASE3 COMPLETE"
