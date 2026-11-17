# JRM-ADM adapter provenance

Upstream: [antoinedepaepe/jrm-adm](https://github.com/antoinedepaepe/jrm-adm), commit
`4552c7123aaa38848d75071eb0e0028b0206e052`, by De Paepe and colleagues,
*Adaptive Diffusion Models for Sparse-View Motion-Corrected Head Cone-Beam CT*.
The upstream source and its license are obtained separately; this directory
contains our adapters, not a copy of the solver or released model weights.

`train_w3dm.py` implements our training recipe because the upstream repository
provides inference code. The manuscript baseline uses a prior trained on our
150 CQ500 training patients for 300,000 iterations: AdamW, learning rate 1e-4,
batch 1, EMA .999, seed 0, fp32 and no augmentation. The noise schedule and
wavelet network come from the pinned upstream code. This recipe is not a claim
that every unpublished training choice of the original paper was recovered.

`run_on_ours.py` connects the released sampler, RMSprop optimizers, cubic-spline
motion model and wavelet prior to our common 360-view measurements. The adaptations
are detector pitch .64 mm; angles shifted by +90 degrees to match the projector
conventions; a 224³ reconstruction grid; angle batches of 36; gamma 33000;
and a z-flip before and after each prior call to match training orientation.
The supplied noiseless statistical weights are `yi = 5e5 * exp(-b)`.
The measurement trajectory is the shared 10-node Akima realization, while the
baseline estimates its own 30-control-point cubic-spline trajectory.

Training crops are 160×192×192, whereas inference uses 224³ to cover the measurement
support. The recovered affine sampling transforms are inverted into our axis-angle
motion convention by `scripts/jrm_theta_convert.py`; that conversion assumes the
cubic 1 mm reconstruction grid. Native output attenuation is scaled by
`0.02 / 0.0193` before common evaluation.

The archived inference adapter did not seed or record its initial Gaussian image.
The release adds explicit per-case seeds and saves them in `port_args` for future
runs. This improves repeatability without asserting numerical identity with the
archived comparison. Model weights and CQ500 images are not bundled.
