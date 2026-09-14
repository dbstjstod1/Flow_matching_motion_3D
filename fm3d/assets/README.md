# Cubic B-spline interpolation basis

`jrm_cubic_bspline_V360_C20.npy` is the 360-view × 20-control-point float32 interpolation
matrix used in the manuscript pose-fitting ablation. It was evaluated from
`torch-cubic-spline-grids==0.5.2`, the interpolation package used by released JRM-ADM.
The array contains interpolation weights, not patient data or model weights.

SHA256: `2a609f0fc10fac8fb1ee33a754458cfbd14eace2b653bc25e3dec81c94e16ce6`.
`bspline_provenance.json` records the original forward/gradient comparison errors.
The runtime checks the 360×20 shape; other view counts require a newly evaluated basis.
