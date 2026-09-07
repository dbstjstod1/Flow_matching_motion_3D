# LEAP with the modular-beam forward pinned to Joseph

The projector of this repository is [LLNL LEAP](https://github.com/LLNL/LEAP) (MIT), commit
`0c8846f`, plus ONE local change: `leap_joseph_pin.patch` (6 files, 42 lines, additive only).

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
`z >= 0.9961` (a 5.06 deg panel tilt) and the source z-span stays under half the panel height.
The patch adds `params.forceJosephModular`, which sets `useSF = false` before that test, plus
`set_forceJosephModular` / `get_forceJosephModular` (C entry points + `leapctype` methods).

## Why we need it

We drive LEAP with per-view rigid MOTION geometry (`P_nom @ T(theta)`), and reconstruction lives
in the OBJECT frame, so an object rotation IS a panel tilt. At our motion amplitudes the switch
above fires silently and mid-optimization, which costs three concrete things:

1. **A cliff in the estimator's loss.** Crossing the bar moves the loss by ~11% of its value in
   one step, because the model for ALL views flips at once.
2. **Mixed operators inside one training bridge.** The t=1 image (nominal orbit -> SF) and the
   motion-corrupted measurements (tilted -> Joseph) would come from different kernels.
3. **The SF kernel's lattice ripple.** It projects ROUNDED voxel centres, so its exact geometry
   gradient is the slope of a ~1e-3 rad ripple and is sign-flipped against the loss trend.
   Joseph is bilinear in continuous coordinates: no ripple, and its exact gradient IS the trend.

Forcing Joseph makes ONE operator serve every geometry in the project; our own kernels in
`fm3d/triton_leap_grad.py` are exact derivatives of that one kernel.

## Building

```
git clone https://github.com/LLNL/LEAP.git && cd LEAP && git checkout 0c8846f
git apply /path/to/third_party/leap/leap_joseph_pin.patch
mkdir build && cd build && cmake .. && cmake --build . -j12
pip install ..                                   # installs leapctype.py
cp lib/libleapct.so <site-packages>/libleapct.so # replace the stock library
```

`fm3d/leap_projector.py` calls `set_forceJosephModular(True)` on every model instance and
RAISES if the symbol is absent, so an unpatched library fails loudly instead of silently
training the prior against a different operator.
