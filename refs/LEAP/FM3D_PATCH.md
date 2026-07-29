# THIS IS A PATCHED LEAP. Do not mistake it for stock.

Vendored from LLNL LEAP (MIT), commit `0c8846f`, plus ONE local change, applied 2026-07-30.
`git diff refs/LEAP` in the parent repo shows it in full (6 files, 42 lines, additive only).

## The change: `forceJosephModular`

Stock LEAP selects the modular-beam FORWARD projector per GEOMETRY SET, inside
`project_Joseph_modular` (`src/projectors_Joseph.cu`):

```cpp
if (params->modularbeamIsAxiallyAligned() == true && useSF == true)
    modularBeamProjectorKernel_SF   <<<...>>>   // separable footprint
else
    modularBeamJosephProjectorKernel<<<...>>>   // ray driven
```

`modularbeamIsAxiallyAligned()` is true only while EVERY view's unit rowVector keeps
`z >= 0.9961` (a 5.06 deg panel tilt) and the source z-span stays under half the panel height
(`parameters::set_sourcesAndModules`). The patch adds `params.forceJosephModular`, which sets
`useSF = false` before that test, plus `set_forceJosephModular` / `get_forceJosephModular`
(C entry points + `leapctype` methods).

## Why we need it

We drive LEAP with per-view rigid MOTION geometry (`P_nom @ T(theta)`), and reconstruction lives
in the OBJECT frame, so an object rotation IS a panel tilt. Measured on our akima motion:
training amplitude flips 60/60 draws to Joseph, eval amplitude 54/60, half amplitude 0/60. The
switch therefore fires **silently and mid-optimization**, and it cost us three concrete things:

1. **A cliff in the estimator's loss.** Crossing the bar moves the loss by ~11% of its value in
   one step -- 1.9x the genuine physical change over the same step -- because the model for ALL
   360 views flips at once.
2. **Mixed operators inside one training bridge.** The t=1 anchor is simulated at the NOMINAL
   orbit (SF) while the draw's y is simulated at the motion geometry (Joseph), so ~26% of the
   bridge's detrend `Delta` was operator difference rather than motion artefact.
3. **The SF kernel's lattice ripple.** It projects ROUNDED voxel centres, so its exact geometry
   gradient is the slope of a ~1e-3 rad ripple and is SIGN-FLIPPED against the loss trend --
   which forced a second, surrogate gradient path (`triton_sf.sf_grad_P`) to exist purely to
   cover the SF branch. Joseph is bilinear in continuous coordinates: no ripple, and its exact
   gradient IS the trend (FD parity 4e-4 vs SF's 1e-2).

Forcing Joseph makes ONE operator serve every geometry in the project.

## Rebuilding

```
cd refs/LEAP && rm -rf build && mkdir build && cd build && cmake .. && cmake --build . -j12
cp lib/libleapct.so <site-packages>/libleapct.so      # keep libleapct.so.stock next to it
```

`fm3d/leap_projector.py` calls `set_forceJosephModular(True)` on every model instance and
RAISES if the symbol is absent, so an unpatched .so fails loudly instead of silently retraining
the prior against a different operator.
