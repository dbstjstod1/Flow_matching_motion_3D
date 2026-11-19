# CQ500 comparison methods

Use the same `(split=test, run=i, seed=1000+i)` cases, `i=0..29`, as the proposed
method: 360 views and nominal control-point sampling widths 10 mm / 10 degrees.
The flow-matching checkpoint passed to the export/autofocus scripts defines the
common simulation settings; it is not the comparison method's image prior.

## Learned autofocus

The additional Python packages in the inspected environment were `numba==0.66.0`,
`numba-cuda==0.30.4`, `cuda-bindings==12.9.7` and
`nvidia-cuda-nvcc-cu12==12.2.140`. Install these for the vendored CUDA backprojector;
the toolkit/driver combination must support Numba's PTX. Source attribution and
implementation differences are in [bench/thies/PROVENANCE.md](../bench/thies/PROVENANCE.md).

Train on the same 150 CQ500 training patients:

```bash
python scripts/bench_thies_train_qm.py --root data/CQ500 \
    --out logs/bench_thies_qm2 --iters 20000 --batch 16 --lr .001 \
    --amp ours --seed 0
```

`qmnet_best.pth` is selected by **mean validation L1** over all 50 validation
patients with fixed motion draws. The archived best checkpoint is iteration
19,500. The separate validation RPE probe does not select the checkpoint.

Run from the repository root using the accelerated backprojector used for the
submitted paper's 1.9-minute full-process comparison:

```bash
set -e
for i in $(seq 0 29); do
    tag=$(printf 'p%02d' "$i")
    FM3D_THIES_VENDOR_BP=0 CUDA_VISIBLE_DEVICES=0 python scripts/bench_thies_estimate.py \
        --qm logs/bench_thies_qm2/qmnet_best.pth \
        --ckpt logs/fm3d_databridge/ckpt_iter500000.pth --root data/CQ500 \
        --split test --run "$i" --seed "$((1000+i))" --trans_mm 10 --rot_deg 10 \
        --iters 100 --s0 100 --decay .97 --est_nodes 30 \
        --out "data/bench_thies_test30/$tag"
done
```

Use a fresh output directory for each run. For an original-kernel diagnostic,
set `FM3D_THIES_VENDOR_BP=1` and use `scripts/bench_thies_original.py`; that wrapper
records `backend.json`. Keep its timing separate from the paper's accelerated
implementation. `est_seconds` covers pose estimation only; the paper's runtime
covers the complete per-patient process, including simulation, evaluation and output.

Rescore the reconstructed `out_vol` with the paper's common measured-FOV image
metrics, since the native benchmark also reports its own reference conventions:

```bash
python scripts/roi_body_metrics.py --ckpt logs/fm3d_databridge/ckpt_iter500000.pth \
    --root data/CQ500 --arm thies=data/bench_thies_test30 \
    --out data/autofocus_common_metrics.json
python scripts/rpe_report.py data/bench_thies_test30/p*/result.pt
```

Read `thies.psnr_meas` / `thies.ssim_meas`, rather than the additional body-ROI
scores. The motion report includes the zero-centered RPE convention.

## JRM-ADM

Follow [the adapter guide](../baselines/jrm_adm/README.md) in a separate environment.
The manuscript comparison uses the prior retrained for 300,000 iterations on our
150 training patients, 224³ reconstruction, gamma 33000 and a z-flip around prior
calls. Released-checkpoint diagnostics and other prior-swap experiments are outside
this release.
