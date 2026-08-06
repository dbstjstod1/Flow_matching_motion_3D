#!/bin/bash
# 2026-08-03, phase 5: settle FULL FINE vs c2f on 3 patients, at the deployed per400.
# c2f already has val0/1/2 in data/post500k_val* ; this adds fine-only val1/val2 (val0 has TWO
# reps already: data/persweep_fineonly + data/fine_per400b).
# val0 evidence so far -- fine 1279s / rot 0.075 / x_t 39.44 dB SSIM 0.9851 (n=2)
#                         c2f   949s / rot 0.06  / x_t 39.03 dB SSIM 0.9853 (n=1)
# i.e. quality inside the 0.0018 SSIM rerun noise, c2f 26% cheaper. 3 patients decides it.
cd /home/mirlab/Desktop/Flow_matching_motion_3D
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
export CUDA_VISIBLE_DEVICES=0
CK=logs/fm3d_cq500_leap/ckpt_iter500000.pth
for r in 1 2; do
  echo "##### fine per400 val$r #####"
  $PY scripts/run_posterior3d.py --ckpt $CK --out data/fine400_val$r --run $r --est_coarse 1 \
      > logs/fine400_val$r.log 2>&1
  echo "--- fine400 val$r done ---"
done
echo "PHASE5 COMPLETE"
