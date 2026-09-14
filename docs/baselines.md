# Comparison methods under the paper protocol

All comparison cases use `(test, i, 1000+i)`, 360 views and nominal node-sampling widths
10 mm / 10°. The proposed prior checkpoint supplied to the autofocus/export scripts specifies
our acquisition and simulation setup; it is not used as their image prior.

## Learned autofocus

Install the optional `numba` / `numba-cuda` dependencies for the released backprojector.
The evaluated quality network used 20,000 training iterations, batch 16, learning rate 0.001,
seed 0, `--amp ours`, and the best validation-RPE checkpoint. Its 150-patient training split
matches the proposed method. Train it with:

```bash
python scripts/bench_thies_train_qm.py --root data/CQ500 --out logs/bench_thies_qm2 \
    --iters 20000 --batch 16 --lr .001 --amp ours --seed 0
```

Reconstruct the paired test cases using the authors' original kernels, as in the manuscript's
runtime comparison (run from the repository root):

```bash
for i in $(seq 0 29); do
    tag=$(printf 'p%02d' "$i")
    FM3D_THIES_VENDOR_BP=1 CUDA_VISIBLE_DEVICES=0 python scripts/bench_thies_original.py \
        --qm logs/bench_thies_qm2/qmnet_best.pth \
        --ckpt logs/fm3d_databridge/ckpt_iter500000.pth --root data/CQ500 \
        --split test --run "$i" --seed "$((1000+i))" --trans_mm 10 --rot_deg 10 \
        --iters 100 --s0 100 --decay .97 --est_nodes 30 \
        --out "data/bench_thies_test30/$tag" || break
done
```

`backend.json` records the selected implementation. The accelerated local backprojector is
available through `bench_thies_estimate.py`, but its runtime is a different implementation
comparison. Use a fresh output root for timing runs and record full-process elapsed time;
`est_seconds` measures only the estimation interval.

The native benchmark includes its own image masks and reference conventions. For the paper's
measured-FOV PSNR/SSIM, rescore its `out_vol` with the common evaluator:

```bash
python scripts/roi_body_metrics.py --ckpt logs/fm3d_databridge/ckpt_iter500000.pth \
    --root data/CQ500 --arm thies=data/bench_thies_test30 \
    --out data/autofocus_common_metrics.json
```

Read `thies.psnr_meas` / `thies.ssim_meas` in this JSON, not the separately reported body-ROI
scores. `python scripts/rpe_report.py data/bench_thies_test30/p*/result.pt` reports zero-centered
RPE. Upstream source provenance and licenses remain in `bench/thies/PROVENANCE.md` and `vendor/`.

## JRM-ADM

See [the baseline adapter guide](../baselines/jrm_adm/README.md). The manuscript table uses
our retrained 300k-iteration prior with gamma **33000**, 224³ reconstruction and prior z-flip.
This is distinct from released-checkpoint diagnostics and subsequent retraining experiments.
