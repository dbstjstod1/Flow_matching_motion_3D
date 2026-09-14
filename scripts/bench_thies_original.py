"""Run the autofocus benchmark with the authors' original released CUDA kernels."""
from pathlib import Path
import hashlib
import inspect
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if os.environ.get('FM3D_THIES_VENDOR_BP') != '1':
    raise RuntimeError('Original-kernel timing requires FM3D_THIES_VENDOR_BP=1 before import')
from bench.thies.fast_backprojector import FAST, backprojector
from bench.thies.vendor_import import cone_backprojector
from scripts.bench_thies_estimate import build_args, main


def verify_backend():
    selected = backprojector()
    source = Path(inspect.getfile(selected)).resolve()
    expected = ROOT / 'bench/thies/vendor/geometry_gradients_CT/backprojector_cone.py'
    assert not FAST and selected is cone_backprojector()
    assert source == expected.resolve(), source
    return dict(backend='authors_released_kernels', accelerated=False,
                class_name=selected.__module__+'.'+selected.__name__,
                source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
                FM3D_THIES_VENDOR_BP=os.environ['FM3D_THIES_VENDOR_BP'])


if __name__ == '__main__':
    backend = verify_backend()
    if sys.argv[1:] == ['--verify-only']:
        print(json.dumps(backend, indent=2))
    else:
        args = build_args()
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out/'backend.json').write_text(json.dumps(backend, indent=2)+'\n')
        print('[original-backend] '+json.dumps(backend), flush=True)
        main()
