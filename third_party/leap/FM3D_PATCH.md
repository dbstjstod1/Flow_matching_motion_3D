# Patched LEAP projector

Use [LLNL LEAP](https://github.com/LLNL/LEAP) (MIT) at commit `0c8846f` with
[leap_joseph_pin.patch](leap_joseph_pin.patch). The patch adds a
`forceJosephModular` flag and its Python/C accessors in six source files.
It pins the modular-beam forward projector to the Joseph kernel for every pose.
Stock LEAP can switch between separable-footprint and Joseph kernels as the
geometry changes; that would change the operator during training and pose fitting.
`fm3d/triton_leap_grad.py` differentiates the pinned Joseph operator.

Build from the repository root with an NVIDIA CUDA toolkit, a compatible C++17
compiler, OpenMP and CMake 3.23 or newer. CMake 3.x avoids the upstream build's
legacy `FindCUDA` compatibility warnings. The toolkit must support your GPU.

```bash
git clone https://github.com/LLNL/LEAP.git refs/LEAP
git -C refs/LEAP checkout 0c8846f
git -C refs/LEAP apply --check ../../third_party/leap/leap_joseph_pin.patch
git -C refs/LEAP apply ../../third_party/leap/leap_joseph_pin.patch
cmake -S refs/LEAP -B refs/LEAP/build -DCMAKE_BUILD_TYPE=Release
cmake --build refs/LEAP/build --parallel 8
export PYTHONPATH="$PWD/refs/LEAP/src${PYTHONPATH:+:$PYTHONPATH}"
python - <<'PY'
from leapctype import tomographicModels
m = tomographicModels()
assert m.set_forceJosephModular(True)
assert m.get_forceJosephModular()
print('Patched LEAP loaded')
PY
```

Keep that `PYTHONPATH` setting in each training/inference shell. The upstream
`src/leapctype.py` loader finds `../build/lib/libleapct.so` directly, so this recipe
does not copy a library into an unrelated Python environment. If CUDA detection
fails, pass `-DCMAKE_CUDA_COMPILER=/path/to/nvcc` to the configure command.

The application also sets the flag on every new projector instance and rejects an
unpatched library. CUDA-dependent operator gates are listed in the main README.
The upstream license remains in the cloned `refs/LEAP/LICENSE`; no compiled library
or upstream source tree is distributed in this repository.
