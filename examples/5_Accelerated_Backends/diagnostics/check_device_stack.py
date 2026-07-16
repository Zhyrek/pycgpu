"""GPU device stack-limit probe (mainly for ROCm/HIP troubleshooting).

    python check_device_stack.py

The solver kernels call generated energy functions through function
pointers; those frames come out of the DYNAMIC per-thread stack budget,
which defaults to 1024 bytes on CUDA and ROCm. Measured frames of the
generated code are 1.9-3.3 KB per function, so the launch path raises
the limit to 64 KB. Most CUDA drivers accept this; some ROCm stacks
reject it (hipErrorUnknown), in which case correctness depends on
whether the compiler's static scratch sizing already covers indirect
calls — run check_backend_correctness.py to find out empirically.

PYCGPU_DEVICE_STACK_BYTES=<n> requests a different size; =0 skips the
call entirely.
"""
import warnings

warnings.filterwarnings('ignore')

try:
    import cupy as cp
except ImportError:
    raise SystemExit('CuPy is not installed - the gpu backend is unavailable '
                     '(the c++ backend does not need it).')

is_hip = bool(getattr(cp.cuda.runtime, 'is_hip', False))
print(f'platform            : {"ROCm/HIP" if is_hip else "CUDA"}')
print(f'device              : {cp.cuda.runtime.getDeviceProperties(0)["name"].decode(errors="replace")}')
try:
    limit = cp.cuda.runtime.cudaLimitStackSize
    print(f'current stack limit : {cp.cuda.runtime.deviceGetLimit(limit)} bytes')
except Exception as e:
    print(f'current stack limit : unreadable ({e})')

from pycalphad.gpu.kernel_manager import ensure_device_stack_limit
ok = ensure_device_stack_limit(65536, verbose=True)
print(f'raise to 65536      : {"accepted" if ok else "REJECTED (warning above explains the risk)"}')
try:
    print(f'stack limit now     : {cp.cuda.runtime.deviceGetLimit(cp.cuda.runtime.cudaLimitStackSize)} bytes')
except Exception:
    pass
if ok:
    print('verdict             : stack limit OK for the gpu backend')
else:
    print('verdict             : run check_backend_correctness.py gpu — if it reports')
    print('                      HEALTHY, this platform statically covers indirect-call')
    print('                      frames and the rejected raise does not matter.')
