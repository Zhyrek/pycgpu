"""
CUDA/ROCm module compilation helpers for the CuPy backend.
"""

import os


def cuda_raw_module(code, options, verbose=False):
    """CuPy RawModule with compiler selection for the pip-only CUDA story.

    PYCGPU_CUDA_COMPILER: 'nvcc' (default when CuPy can find nvcc; the
    validated toolchain), 'nvrtc' (runtime compilation via the library the
    CuPy wheel ships — no CUDA toolkit install needed), or 'auto'. When
    CuPy finds no nvcc, NVRTC is used automatically instead of failing; the
    check asks CuPy, which can miss an nvcc that is on PATH (e.g. when it
    finds no CUDA root). NVRTC rejects nvcc-style optimization flags, so
    only -std/-D/--fmad options are forwarded to it; the kernel sources
    carry an RTC-compat block for the missing host C headers.
    """
    import cupy as _cp
    choice = os.environ.get('PYCGPU_CUDA_COMPILER', 'auto').lower()
    if choice not in ('nvcc', 'nvrtc', 'auto'):
        raise ValueError(f"PYCGPU_CUDA_COMPILER must be nvcc/nvrtc/auto, got {choice!r}")
    # On ROCm builds of CuPy, backend='nvcc' translates to hipcc and the RTC
    # backend to hipRTC; probe for the platform's actual offline compiler.
    _is_hip = bool(getattr(_cp.cuda.runtime, 'is_hip', False))
    _offline = 'hipcc' if _is_hip else 'nvcc'
    if choice == 'auto':
        choice = 'nvcc' if _cupy_offline_compiler(_cp, _is_hip) else 'nvrtc'
        if choice == 'nvrtc' and verbose:
            print(f"[GPU] {_offline} not found on PATH; compiling with "
                  f"{'hipRTC' if _is_hip else 'NVRTC'}")
    if choice == 'nvcc':
        return _cp.RawModule(code=code, options=tuple(options), backend='nvcc')
    rtc_options = tuple(o for o in options if o.startswith(('-std', '-D', '--fmad')))
    return _cp.RawModule(code=code, options=rtc_options, backend='nvrtc')


def _cupy_offline_compiler(cp, is_hip):
    """Whether CuPy itself finds the offline compiler (nvcc, or hipcc on ROCm)."""
    finder = getattr(cp.cuda, 'get_hipcc_path' if is_hip else 'get_nvcc_path', None)
    if finder is None:  # older CuPy
        import shutil
        return shutil.which('hipcc' if is_hip else 'nvcc') is not None
    try:
        return finder() is not None
    except Exception:
        return False


def ensure_device_stack_limit(nbytes=65536, verbose=False):
    """Raise the per-thread device stack limit if below nbytes.

    The solver kernels call the generated energy functions through function
    pointers, so the compiler cannot statically size the per-thread stack:
    frames reached through an indirect call come out of the DYNAMIC stack
    budget, which defaults to 1 KB on both CUDA and ROCm. Overflowing it
    silently corrupts other threads' local memory (this showed up as
    nondeterministic results at batch sizes over ~200 conditions on CUDA;
    the same budget applies to AMD's scratch-backed dynamic stack).

    The raise is ATTEMPTED on both platforms — newer ROCm supports
    hipDeviceSetLimit(hipLimitStackSize); older stacks reject it (the
    observed failure is hipErrorUnknown), in which case a loud warning
    explains the residual risk instead of the whole accelerated path dying.

    PYCGPU_DEVICE_STACK_BYTES overrides the requested size (some ROCm
    versions accept certain values only); set it to 0 to skip the call
    entirely.

    Returns True if the limit is known to satisfy the request.
    """
    import warnings
    import cupy as _cp
    override = os.environ.get('PYCGPU_DEVICE_STACK_BYTES')
    if override is not None:
        nbytes = int(override)
        if nbytes == 0:
            return False
    _is_hip = bool(getattr(_cp.cuda.runtime, 'is_hip', False))
    try:
        limit = _cp.cuda.runtime.cudaLimitStackSize
        if _cp.cuda.runtime.deviceGetLimit(limit) < nbytes:
            _cp.cuda.runtime.deviceSetLimit(limit, nbytes)
        return True
    except Exception as e:
        if _is_hip:
            # Most ROCm stacks reject hipDeviceSetLimit(hipLimitStackSize).
            # VALIDATED BENIGN on real AMD hardware: with the limit pinned at
            # the 1024-byte default, the full correctness A/B still agrees to
            # eps-class (1e-5 J GM) — ROCm's compiler sizes kernel scratch
            # statically (including for indirect calls), so this runtime
            # limit does not back these kernels' stack. If it did, the
            # measured 1.9-3.3 KB per-function frames would corrupt every
            # result, loudly.
            warnings.warn(
                f"ROCm rejected the device stack-limit raise ({e}); this is "
                f"expected and benign on AMD (kernel scratch is sized "
                f"statically at load). One-time confirmation for a new "
                f"machine: examples/5_Accelerated_Backends/diagnostics/"
                f"check_backend_correctness.py gpu")
            return False
        warnings.warn(
            f"could not raise the device stack limit to {nbytes} bytes on "
            f"CUDA ({e}); continuing with the driver default (typically "
            f"1024 bytes). On CUDA this limit DOES back the kernels' "
            f"function-pointer call frames and 1 KB is known to be "
            f"insufficient — overflow corrupts results SILENTLY at larger "
            f"batch sizes. Validate against the reference before trusting "
            f"gpu results, or use the c++ backend (identical numerics). "
            f"PYCGPU_DEVICE_STACK_BYTES=<n> requests a different size.")
        return False



try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False
