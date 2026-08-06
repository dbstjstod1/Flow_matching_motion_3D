#!/bin/bash
# PER / coarse-grid ablation, 2026-08-03. GPU0 ONLY, strictly sequential so step times are clean.
# Baseline for comparison = data/post500k_val0 (deployed winning set, per 400, c2f 1/2 -> 1/1 @ t=0.5).
# Everything else is held at the winning set; only the listed flag changes.
cd /home/mirlab/Desktop/Flow_matching_motion_3D
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
export CUDA_VISIBLE_DEVICES=0
CK=logs/fm3d_cq500_leap/ckpt_iter500000.pth
run () {   # run <tag> <extra flags...>
  tag=$1; shift
  echo "##### $tag :: $* #####"
  $PY scripts/run_posterior3d.py --ckpt $CK --out data/persweep_$tag --run 0 "$@" \
      > logs/persweep_$tag.log 2>&1
  echo "--- $tag done ---"
  grep -E "^step  49|FINAL" logs/persweep_$tag.log
}
run oracle    --theta_oracle
run fineonly  --est_coarse 1
run coarseall --est_coarse_until 1.0
run per200    --per 200
run per100    --per 100
run per50     --per 50
run per200ramp --per 200 --per_sched ramp
echo "PERSWEEP COMPLETE"
