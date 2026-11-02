"""Verify the H200 environment: CUDA 12.9 wheels, SM90 device, vLLM FlashAttention-3, CuTe DSL JIT, patch anchor.

    source scripts/h200/env.sh && python scripts/h200/check_env.py [--jit] [--tirx]
"""
import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def section(title):
    print(f'\n== {title}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jit', action='store_true', help='also JIT-compile and run a tiny CuTe DSL kernel on the GPU')
    ap.add_argument('--tirx', action='store_true', help='also import the TIRx-harness stack')
    args = ap.parse_args()
    ok = True

    section('python / CUDA env')
    print('python', sys.version.split()[0], sys.executable)
    print('CUDA_HOME', os.environ.get('CUDA_HOME'), '| CUDA_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES'))
    print('nvcc:', subprocess.run(['nvcc', '--version'], capture_output=True, text=True).stdout.strip().splitlines()[-1])

    section('torch')
    import torch
    print('torch', torch.__version__, '| cuda', torch.version.cuda, '| cudnn', torch.backends.cudnn.version())
    assert torch.cuda.is_available(), 'torch.cuda.is_available() is False'
    p = torch.cuda.get_device_properties(0)
    cc = (p.major, p.minor)
    print(f'device {p.name} | CC {p.major}.{p.minor} | SMs {p.multi_processor_count} | HBM {p.total_memory / 2**30:.0f} GiB'
          f' | smem/block optin {getattr(p, "shared_memory_per_block_optin", 0) // 1024} KiB')
    if cc != (9, 0):
        print('WARNING: expected an SM90 (H100/H200) device for this branch')
        ok = False

    section('vLLM')
    import vllm
    print('vllm', vllm.__version__, Path(vllm.__file__).parent)
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import FA3_AVAILABLE, FA2_AVAILABLE
        print('vllm_flash_attn: FA2', FA2_AVAILABLE, '| FA3', FA3_AVAILABLE)
    except Exception as e:  # the interface moves between releases
        print('vllm_flash_attn import note:', e)
    fa = Path(vllm.__file__).parent / 'v1/attention/backends/flash_attn.py'
    r = subprocess.run([sys.executable, str(REPO / 'tools/patch_vllm.py'), '--target', str(Path(vllm.__file__).parent), '--check'],
                       capture_output=True, text=True)
    print('patch_vllm.py --check:', (r.stdout + r.stderr).strip().replace('\n', ' | '))
    if 'NOT FOUND' in r.stdout:
        ok = False

    section('CuTe DSL / cuda-python')
    import cutlass, cuda.bindings
    print('nvidia-cutlass-dsl', getattr(cutlass, '__version__', '?'), '| cuda-bindings', cuda.bindings.__version__)
    import cuda.bindings.driver as cu
    err, ver = cu.cuDriverGetVersion()
    print('driver CUDA version', ver // 1000, '.', (ver % 1000) // 10, sep='')
    if args.jit:
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack

        @cute.kernel
        def k_add(a: cute.Tensor, b: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            b[tidx] = a[tidx] + 1.0

        @cute.jit
        def launch(a: cute.Tensor, b: cute.Tensor):
            k_add(a, b).launch(grid=(1, 1, 1), block=(256, 1, 1))

        a = torch.arange(256, dtype=torch.float32, device='cuda')
        b = torch.zeros_like(a)
        launch(from_dlpack(a), from_dlpack(b))
        torch.cuda.synchronize()
        assert torch.equal(b, a + 1), 'CuTe DSL kernel produced wrong output'
        print('CuTe DSL JIT kernel on', p.name, ': OK')

    section('evaluation deps')
    for m in ('transformers', 'httpx', 'yaml', 'sm120_attn'):
        try:
            mod = importlib.import_module(m)
            print(m, getattr(mod, '__version__', ''), Path(mod.__file__).parent if m == 'sm120_attn' else '')
        except Exception as e:
            print(m, 'MISSING:', e)
            ok = False

    if args.tirx:
        section('TIRx-harness')
        try:
            import tvm.tirx, tvm_ffi, tirx_kernels.tirx_lite, tirx_harness  # noqa: F401
            print('tirx-harness', tirx_harness.__version__, '| core imports OK')
            print('cargo:', subprocess.run(['cargo', '--version'], capture_output=True, text=True).stdout.strip() or 'MISSING')
        except Exception as e:
            print('TIRx import failed:', e)
            ok = False

    section('model')
    m = Path(os.environ.get('QWEN38_MODEL', ''))
    shards = sorted(m.glob('model-*.safetensors'))
    print(m, '| config' if (m / 'config.json').exists() else '| NO config.json', '|', len(shards), 'safetensors shards')
    ok &= (m / 'config.json').exists() and len(shards) == 18

    print('\nRESULT:', 'OK' if ok else 'PROBLEMS FOUND')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
