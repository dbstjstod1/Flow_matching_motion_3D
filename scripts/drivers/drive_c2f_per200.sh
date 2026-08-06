#!/bin/bash
# 2026-08-03, phase 6: test the user's hypothesis that c2f x per200 is the sweet spot.
# At n=1 it is unresolvable: PSNR ranks it 2nd of all arms (39.30, within 1/6 of a noise bar of
# the best) while SSIM ranks it LAST of the per100..400 group (0.9810, 2.4x the noise bar below
# c2f/per400) -- and the c2f SSIM ordering is NON-MONOTONIC in PER (400: .9853, 100: .9838,
# 200: .9810), which is itself the signature of noise domination.
# rot IS resolved and IS worse: 0.16 vs 0.06 deg at a 0.01 deg noise bar.
# val0 is rerun to size this config's own spread; val1/val2 give a 3-patient mean to set against
# the c2f/per400 3-patient baseline already in data/post500k_val*.
# Waits for phase 5 to release GPU0 first.
cd /home/mirlab/Desktop/Flow_matching_motion_3D
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
export CUDA_VISIBLE_DEVICES=0
CK=logs/fm3d_cq500_leap/ckpt_iter500000.pth
while pgrep -f drive_fine_vs_c2f.sh >/dev/null; do sleep 30; done
for spec in "val0b 0" "val1 1" "val2 2"; do
  set -- $spec; tag=$1; r=$2
  echo "##### c2f per200 $tag #####"
  $PY scripts/run_posterior3d.py --ckpt $CK --out data/c2f200_$tag --run $r --per 200 \
      > logs/c2f200_$tag.log 2>&1
  echo "--- c2f200 $tag done ---"
done
echo "PHASE6 COMPLETE"
