"""
Metal backend runtime: runs the generated pycalphad kernels on Apple GPUs.

The generated solver is written once, as CUDA-flavored C++ (see gpu_codegen
and the *.h kernel headers). This module turns that source into Metal
Shading Language and runs it through a tiny Objective-C++ bridge
(metal_bridge.mm) that is compiled with the system clang++ on first use, so
the backend needs nothing beyond the Xcode command line tools.

Two things make the translation work:

* Precision. Apple GPUs have no FP64 units. Every ``double`` becomes
  ``pyc_real``, a double-float pair with an extra exponent (see
  metal_compat.h) carrying ~48 significand bits over double's range.
  Floating literals are split into exact ``PYC_C(hi, lo, e)`` triples here,
  because Metal parses unsuffixed literals as single precision. GPU buffers
  keep the IEEE float64 layout and are converted on load/store, so host
  arrays are passed unchanged.

* Address spaces. Metal requires every pointer and reference to name its
  address space. The solver keeps all of its per-condition state in thread
  memory, so :func:`lower_to_msl` qualifies every pointer/reference that has
  no explicit space as ``thread``; the few pointers into shared GPU buffers
  are spelled ``PYCGPU_GLOBAL`` in the headers (empty on CUDA/C++,
  ``device`` here).
"""
import ctypes
import hashlib
import os
import platform
import re
import shutil
import subprocess
import threading

import numpy as np

__all__ = ['is_available', 'runtime', 'lower_to_msl', 'real_literal',
           'MetalUnavailableError']

_HERE = os.path.dirname(os.path.abspath(__file__))


class MetalUnavailableError(RuntimeError):
    """No usable Metal device or toolchain on this machine."""


class MetalResourceError(RuntimeError):
    """A kernel needs more per-thread resources (stack) than Metal allows."""


# --------------------------------------------------------------------------
# CUDA/C++ -> Metal Shading Language lowering
# --------------------------------------------------------------------------

_FLOAT_LITERAL = re.compile(
    r'(?<![\w.])'
    r'((?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)'
    r'([fFlL]?)'
    r'(?![\w.])')

# Type names that may precede a pointer or reference declarator. Struct names
# from the kernel headers are added dynamically (see _struct_names).
_BASE_TYPES = ('double', 'float', 'int', 'bool', 'char', 'void', 'long', 'short',
               'size_t', 'uint', 'uchar', 'ulong', 'unsigned', 'pyc_real', 'pyc_f64')

_ADDRESS_SPACES = ('thread', 'device', 'constant', 'threadgroup')


def _f32(v):
    v = float(v)
    if v == 0.0:
        return '0.0f'
    return repr(v) + 'f'


def real_literal(value):
    """Exact PYC_C(hi, lo, e) spelling of a float64 constant (see the
    canonical form in metal_compat.h)."""
    value = float(value)
    if np.isnan(value):
        return 'PYC_C(NAN, 0.0f, 0)'
    if np.isinf(value):
        return f"PYC_C({'' if value > 0 else '-'}INFINITY, 0.0f, 0)"
    if value == 0.0:
        return 'PYC_C(0.0f, 0.0f, 0)'
    exp2 = np.frexp(value)[1]               # value = mant * 2^exp2, |mant| in [0.5, 1)
    e = 0
    if not (2.0 ** -62 <= abs(value) < 2.0 ** 62):
        e = int(exp2) - 1
    scaled = float(np.ldexp(value, -e))
    hi = np.float32(scaled)
    lo = np.float32(scaled - float(hi))
    return f'PYC_C({_f32(hi)}, {_f32(lo)}, {e})'


def _real_literal_sub(match):
    text, suffix = match.group(1), match.group(2)
    if suffix in ('f', 'F'):
        return match.group(0)  # explicitly single precision: leave alone
    return real_literal(float(text))


def _struct_names(source):
    names = set(re.findall(r'\bstruct\s+([A-Za-z_]\w*)', source))
    names |= set(re.findall(r'\btypedef\s+struct\s+\w*\s*\{[^}]*\}\s*([A-Za-z_]\w*)\s*;', source, re.S))
    names |= set(re.findall(r'}\s*([A-Za-z_]\w*)\s*;', source))
    return {n for n in names if n[0].isupper() or n.startswith('pyc')}


def _qualify_pointers(source, type_names):
    """Insert `thread` before every pointer/reference declarator whose
    pointee has no explicit address space."""
    types = '|'.join(sorted(set(_BASE_TYPES) | set(type_names), key=len, reverse=True))
    # TYPE [const] (* | &) [const] (* | &) ...
    pattern = re.compile(
        r'(?P<prefix>\b(?:(?:' + '|'.join(_ADDRESS_SPACES) + r')\s+)?(?:const\s+)?)'
        r'(?P<type>\b(?:' + types + r')\b(?:\s+(?:' + types + r')\b)*)'
        r'(?P<decl>(?:\s*(?:const\s*)?(?:' + '|'.join(_ADDRESS_SPACES) + r')?\s*[*&](?![&=*]))+)')

    def fix(m):
        prefix, typ, decl = m.group('prefix'), m.group('type'), m.group('decl')
        if any(prefix.strip().startswith(a) for a in _ADDRESS_SPACES):
            return m.group(0)  # first level already qualified (e.g. device double*)
        out = []
        for part in re.finditer(r'\s*(const\s*)?((?:' + '|'.join(_ADDRESS_SPACES) + r')\s*)?([*&])', decl):
            const, space, sym = part.group(1) or '', part.group(2), part.group(3)
            out.append(f' {const}{space.strip() if space else "thread"}{sym}')
        return prefix + typ + ''.join(out)

    return pattern.sub(fix, source)


def lower_to_msl(source, extra_type_names=()):
    """Rewrite CUDA/C++ kernel source into Metal Shading Language.

    The result still needs the metal_compat.h prelude (prepended by
    :func:`build_msl_source`).
    """
    src = source
    # Strip C linkage and host includes; Metal has neither.
    src = re.sub(r'^\s*#\s*include\s*<[^>]+>.*$', '', src, flags=re.M)
    src = re.sub(r'extern\s+"C"\s*\{', '', src)
    src = re.sub(r'extern\s+"C"\s+', '', src)
    src = re.sub(r'#ifdef\s+__cplusplus\s*\n\s*\}\s*\n\s*#endif', '', src)
    src = re.sub(r'\bunsigned\s+long\s+long\b', 'ulong', src)
    src = re.sub(r'\blong\s+long\b', 'long', src)
    src = re.sub(r'\bunsigned\s+int\b', 'uint', src)
    src = re.sub(r'\bunsigned\s+char\b', 'uchar', src)
    src = _FLOAT_LITERAL.sub(_real_literal_sub, src)
    # Address-space markers (left alone on preprocessor lines, where the
    # headers define them empty for CUDA/C++). Doubles in GPU buffers keep
    # the IEEE float64 layout.
    def code_lines(fn):
        return '\n'.join(line if line.lstrip().startswith('#') else fn(line)
                         for line in src.split('\n'))
    src = code_lines(lambda l: re.sub(r'\bPYCGPU_GLOBAL\s+(const\s+)?double\s*\*', r'\1pyc_f64 device*', l))
    src = code_lines(lambda l: re.sub(r'\bPYCGPU_GLOBAL\b', 'device', l))
    src = code_lines(lambda l: re.sub(r'\bPYCGPU_LITERAL\b', 'constant', l))
    # Identifiers that are reserved words in Metal.
    src = re.sub(r'\b(half|vec)\b(?!\s*\()', r'pyc_\1', src)
    src = _qualify_pointers(src, _struct_names(src) | set(extra_type_names))
    return src


def read_prelude():
    with open(os.path.join(_HERE, 'metal_compat.h')) as f:
        return f.read()


def build_msl_source(body, defines=None):
    """Prelude + -D style defines + lowered body."""
    define_lines = ''.join(f'#define {k} {v}\n' for k, v in (defines or {}).items())
    return read_prelude() + '\n' + define_lines + '\n' + body


# --------------------------------------------------------------------------
# Runtime
# --------------------------------------------------------------------------

class _Bridge:
    def __init__(self, lib):
        self.lib = lib
        c = ctypes
        lib.pyc_mtl_init.argtypes = [c.POINTER(c.c_void_p), c.c_char_p, c.c_int, c.c_char_p, c.c_int]
        lib.pyc_mtl_recommended_working_set.argtypes = [c.c_void_p]
        lib.pyc_mtl_recommended_working_set.restype = c.c_uint64
        lib.pyc_mtl_compile.argtypes = [c.c_void_p, c.c_char_p, c.c_int, c.POINTER(c.c_void_p),
                                        c.c_char_p, c.c_int]
        lib.pyc_mtl_pipeline.argtypes = [c.c_void_p, c.c_void_p, c.c_char_p, c.POINTER(c.c_void_p),
                                         c.POINTER(c.c_int), c.POINTER(c.c_int), c.c_char_p, c.c_int]
        lib.pyc_mtl_buffer.argtypes = [c.c_void_p, c.c_uint64, c.POINTER(c.c_void_p)]
        lib.pyc_mtl_buffer.restype = c.c_void_p
        lib.pyc_mtl_release.argtypes = [c.c_void_p]
        lib.pyc_mtl_dispatch.argtypes = [c.c_void_p, c.c_void_p, c.c_int, c.POINTER(c.c_void_p),
                                         c.c_void_p, c.c_uint64, c.c_uint32, c.c_uint32,
                                         c.POINTER(c.c_double), c.c_char_p, c.c_int]


_ERRLEN = 65536


def _cache_dir():
    from pycalphad.gpu.gpu_equilibrium import _kernel_cache_dir
    return str(_kernel_cache_dir())


def _build_bridge():
    if platform.system() != 'Darwin':
        raise MetalUnavailableError("the Metal backend requires macOS")
    cxx = shutil.which('clang++')
    if cxx is None:
        raise MetalUnavailableError(
            "the Metal backend needs clang++ to build its Metal bridge; install the "
            "Xcode command line tools with `xcode-select --install`")
    src = os.path.join(_HERE, 'metal_bridge.mm')
    with open(src, 'rb') as f:
        tag = hashlib.md5(f.read() + platform.mac_ver()[0].encode()).hexdigest()[:16]
    lib_path = os.path.join(_cache_dir(), f'pycgpu_metal_bridge_{tag}.dylib')
    if not os.path.exists(lib_path):
        tmp = lib_path + f'.{os.getpid()}.tmp'
        cmd = [cxx, '-std=c++17', '-O2', '-fobjc-arc', '-shared', '-fPIC', src,
               '-framework', 'Metal', '-framework', 'Foundation', '-o', tmp]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise MetalUnavailableError(f"building the Metal bridge failed:\n{res.stderr[-3000:]}")
        os.replace(tmp, lib_path)
    return _Bridge(ctypes.CDLL(lib_path))


class MetalBuffer:
    """Shared-storage MTLBuffer with a zero-copy numpy view."""

    def __init__(self, rt, nbytes):
        self._rt = rt
        contents = ctypes.c_void_p()
        self.nbytes = int(nbytes)
        self.handle = rt._bridge.lib.pyc_mtl_buffer(rt._ctx, max(self.nbytes, 16), ctypes.byref(contents))
        if not self.handle:
            raise MemoryError(f"Metal could not allocate a {self.nbytes}-byte buffer")
        self._ptr = contents.value

    def view(self, dtype=np.uint8, shape=None):
        dtype = np.dtype(dtype)
        count = self.nbytes // dtype.itemsize
        buf = (ctypes.c_char * max(self.nbytes, 1)).from_address(self._ptr)
        arr = np.frombuffer(buf, dtype=dtype, count=count)
        return arr if shape is None else arr.reshape(shape)

    def __del__(self):
        try:
            if self.handle:
                self._rt._bridge.lib.pyc_mtl_release(self.handle)
                self.handle = None
        except Exception:
            pass


class MetalKernel:
    def __init__(self, rt, handle, name, max_threads, simd_width):
        self._rt = rt
        self.handle = handle
        self.name = name
        self.max_threads = max_threads
        self.simd_width = simd_width

    def __call__(self, buffers, params=b'', nthreads=1, threadgroup=None):
        """Dispatch over `nthreads` threads; `buffers` bind to indices
        0..n-1 and the packed `params` bytes to index n. Returns GPU seconds."""
        rt = self._rt
        tg = threadgroup or min(self.max_threads, 64)
        tg = max(1, min(tg, self.max_threads))
        handles = (ctypes.c_void_p * len(buffers))(*[b.handle for b in buffers])
        params = bytes(params)
        pbuf = ctypes.create_string_buffer(params, len(params)) if params else None
        gpu_s = ctypes.c_double(0.0)
        err = ctypes.create_string_buffer(_ERRLEN)
        rc = rt._bridge.lib.pyc_mtl_dispatch(
            rt._ctx, self.handle, len(buffers), handles,
            ctypes.cast(pbuf, ctypes.c_void_p) if pbuf is not None else None, len(params),
            int(nthreads), int(tg), ctypes.byref(gpu_s), err, _ERRLEN)
        if rc != 0:
            raise RuntimeError(f"Metal kernel '{self.name}' failed: {err.value.decode(errors='replace')}")
        return gpu_s.value


class MetalRuntime:
    def __init__(self):
        self._bridge = _build_bridge()
        ctx = ctypes.c_void_p()
        name = ctypes.create_string_buffer(256)
        err = ctypes.create_string_buffer(_ERRLEN)
        if self._bridge.lib.pyc_mtl_init(ctypes.byref(ctx), name, 256, err, _ERRLEN) != 0:
            raise MetalUnavailableError(err.value.decode(errors='replace'))
        self._ctx = ctx.value
        self.device_name = name.value.decode()
        self._libraries = {}
        self._lock = threading.Lock()

    @property
    def recommended_working_set(self):
        return int(self._bridge.lib.pyc_mtl_recommended_working_set(self._ctx))

    def compile(self, msl_source):
        """Compile MSL source (cached per source text for this process)."""
        key = hashlib.sha256(msl_source.encode()).hexdigest()
        with self._lock:
            if key in self._libraries:
                return self._libraries[key]
            lib = ctypes.c_void_p()
            err = ctypes.create_string_buffer(_ERRLEN)
            rc = self._bridge.lib.pyc_mtl_compile(self._ctx, msl_source.encode(), 0,
                                                  ctypes.byref(lib), err, _ERRLEN)
            if rc != 0:
                msg = err.value.decode(errors='replace')
                dump = os.path.join(_cache_dir(), f'failed_{key[:12]}.metal')
                try:
                    with open(dump, 'w') as f:
                        f.write(msl_source)
                except OSError:
                    dump = '<not written>'
                raise RuntimeError(f"Metal shader compilation failed (source: {dump}):\n{msg[:6000]}")
            entry = {'lib': lib.value, 'kernels': {}}
            self._libraries[key] = entry
            return entry

    def kernel(self, library, name):
        with self._lock:
            if name in library['kernels']:
                return library['kernels'][name]
            pso = ctypes.c_void_p()
            max_t = ctypes.c_int()
            simd = ctypes.c_int()
            err = ctypes.create_string_buffer(_ERRLEN)
            rc = self._bridge.lib.pyc_mtl_pipeline(self._ctx, library['lib'], name.encode(),
                                                   ctypes.byref(pso), ctypes.byref(max_t),
                                                   ctypes.byref(simd), err, _ERRLEN)
            if rc != 0:
                msg = err.value.decode(errors='replace')
                if 'stack space' in msg:
                    raise MetalResourceError(msg)
                raise RuntimeError(f"Metal pipeline '{name}' failed: {msg}")
            k = MetalKernel(self, pso.value, name, max_t.value, simd.value)
            library['kernels'][name] = k
            return k

    def buffer(self, nbytes):
        return MetalBuffer(self, nbytes)

    def buffer_from(self, array):
        """New shared buffer holding a copy of `array`'s bytes."""
        a = np.ascontiguousarray(array)
        buf = MetalBuffer(self, a.nbytes)
        if a.nbytes:
            buf.view(np.uint8)[:] = a.view(np.uint8).reshape(-1)
        return buf


_runtime = None
_runtime_error = None
_runtime_lock = threading.Lock()


def runtime():
    """The process-wide Metal runtime (built on first use)."""
    global _runtime, _runtime_error
    with _runtime_lock:
        if _runtime is None:
            if _runtime_error is not None:
                raise _runtime_error
            try:
                _runtime = MetalRuntime()
            except MetalUnavailableError as e:
                _runtime_error = e
                raise
    return _runtime


def is_available():
    """True when a Metal device and the bridge toolchain are usable."""
    try:
        runtime()
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# Equilibrium module source
# --------------------------------------------------------------------------

_INIT_CALL = re.compile(
    r'g_phase_records_array\[(\d+)\]\.init\(&(\w+), &\w+, &\w+, (nullptr|&\w+), &\w+, &\w+, '
    r'&\w+, &\w+, &\w+, (\d+), (\d+), (\d+), (\d+), (\d+)\);')

_SCALAR_FNS = (('PYCGPU_FN_OBJ', 'obj'), ('PYCGPU_FN_FORMULAOBJ', 'formulaobj'))
_ARRAY_FNS = (('PYCGPU_FN_FORMULAGRAD', 'formulagrad'), ('PYCGPU_FN_FORMULAHESS', 'formulahess'),
              ('PYCGPU_FN_INTERNAL_CONS_FUNC', 'internal_cons_func'),
              ('PYCGPU_FN_INTERNAL_CONS_JAC', 'internal_cons_jac'),
              ('PYCGPU_FN_MASS_OBJ', 'mass_obj'), ('PYCGPU_FN_FORMULAMOLE_OBJ', 'formulamole_obj'),
              ('PYCGPU_FN_FORMULAMOLE_GRAD', 'formulamole_grad'),
              ('PYCGPU_FN_FORMULAPARAMGRAD', 'formulaparamgrad'),
              ('PYCGPU_FN_FORMULAPARAMMIXED', 'formulaparammixed'))


def _phase_record_table(init_calls):
    """Recover each model's PhaseRecord metadata from the CUDA init calls."""
    records = []
    for call in init_calls:
        m = _INIT_CALL.search(call)
        if m is None:
            raise ValueError(f"unrecognized PhaseRecord init call: {call.strip()[:200]}")
        idx = int(m.group(1))
        prefix = m.group(2)[:-len('obj')]
        records.append(dict(
            idx=idx, prefix=prefix,
            has_hess=m.group(3) != 'nullptr',
            has_fused='.formulafused' in call,
            has_param_derivs='.formulaparamgrad' in call,
            ns=int(m.group(4)), pd=int(m.group(5)), ne=int(m.group(6)),
            nic=int(m.group(7)), nve=int(m.group(8))))
    records.sort(key=lambda r: r['idx'])
    return records


def _drop_functions(source, names):
    """Remove the top-level function definitions named in `names`."""
    out, pos = [], 0
    for m in re.finditer(r'^(?:__device__\s+)?(?:double|void)\s+(\w+)\(', source, re.M):
        if m.group(1) not in names or m.start() < pos:
            continue
        end = source.index('\n}\n', m.end()) + 3   # generated bodies close at column 0
        out.append(source[pos:m.start()])
        pos = end
    out.append(source[pos:])
    return ''.join(out)


def _dispatch_source(records):
    """Switch-based replacements for the CUDA function-pointer table."""
    scalar, array, fused, init = [], [], [], []
    for r in records:
        i, p = r['idx'], r['prefix']
        for const, name in _SCALAR_FNS:
            scalar.append(f"        case {i} * 16 + {const}: return {p}{name}(x);")
        for const, name in _ARRAY_FNS:
            if name == 'formulahess' and not r['has_hess']:
                continue
            if name in ('formulagrad', 'formulahess') and r['has_fused']:
                continue  # unused: the fused function supplies both
            if name.startswith('formulaparam') and not r['has_param_derivs']:
                continue
            array.append(f"        case {i} * 16 + {const}: {p}{name}(out, x); return;")
        if r['has_fused']:
            fused.append(f"        case {i}: {p}fused(out_eg, out_hess, x); return;")
        init.append(
            f"    records[{i}].init_model({i}, {str(r['has_hess']).lower()}, "
            f"{str(r['has_fused']).lower()}, {str(r['has_param_derivs']).lower()}, "
            f"{r['ns']}, {r['pd']}, {r['ne']}, {r['nic']}, {r['nve']});")
    nl = '\n'
    return f"""
PYC_NOINLINE double pycgpu_dispatch_scalar(int fn, int model, const double* x) {{
    switch (model * 16 + fn) {{
{nl.join(scalar)}
    }}
    return 0.0;
}}
PYC_NOINLINE void pycgpu_dispatch_array(int fn, int model, double* out, const double* x) {{
    switch (model * 16 + fn) {{
{nl.join(array)}
    }}
}}
PYC_NOINLINE void pycgpu_dispatch_fused(int model, double* out_eg, double* out_hess, const double* x) {{
    switch (model) {{
{nl.join(fused)}
    }}
}}
#define PYCGPU_NUM_RECORDS {max(len(records), 1)}
void pycgpu_init_phase_records(PhaseRecord* records) {{
{nl.join(init)}
}}
"""


def equilibrium_source(model_functions_c, init_calls, defines):
    """Full Metal source of the equilibrium module.

    `model_functions_c` and `init_calls` come from
    gpu_codegen._generate_c_code_for_phase_models; `defines` are the
    -D size defines (name -> value).
    """
    from pycalphad.gpu.gpu_codegen import solver_headers_source, _read_gpu_header
    records = _phase_record_table(init_calls)
    # With a fused energy+gradient+Hessian function the solver never calls a
    # model's separate gradient/Hessian functions; leaving them out roughly
    # halves the Metal compile time (the Hessian is the largest function).
    fused = {r['idx'] for r in records if r['has_fused']}
    model_functions_c = _drop_functions(
        model_functions_c,
        {f'pycgpu_model_{i}_{name}' for i in fused for name in ('formulagrad', 'formulahess')})
    # Each generated model function is compiled once, out of line: inlining
    # them into every PhaseRecord call site multiplies compile time.
    model_functions_c = re.sub(r'__device__\s+(double|void)\s+(pycgpu_model_\d+_\w+)\(',
                               r'PYC_NOINLINE \1 \2(', model_functions_c)
    body = (solver_headers_source()
            + "\n// --- generated model functions ---\n"
            + model_functions_c
            + _dispatch_source(records)
            + _read_gpu_header("metal_kernels.h"))
    return build_msl_source(lower_to_msl(body), defines)


# --------------------------------------------------------------------------
# Equilibrium solver launches
# --------------------------------------------------------------------------

# Metal's per-thread stack limit; the solver keeps each condition's state in
# thread memory.
METAL_THREAD_STACK_BYTES = 256 * 1024


def thread_stack_estimate(sizes, extra_doubles=0):
    """Approximate thread memory (bytes) the Metal solver needs per condition:
    the SystemState (SYSTEM_STATE_SIZE carries a 1.3x margin over the struct)
    plus the work arrays and `extra_doubles`, at 12 bytes per emulated
    double."""
    from pycalphad.gpu.work_arrays import work_array_strides
    doubles = sum(stride for name, stride in work_array_strides(sizes) if name != 'system_states')
    doubles += int(sizes['SYSTEM_STATE_SIZE']) / 1.3 + extra_doubles
    return int(12 * doubles)


def source_fingerprint():
    """Hash of the Metal-specific inputs to generated sources (the prelude,
    the kernel entry points and the lowering code), for cache keys."""
    import inspect
    h = hashlib.md5()
    for name in ('metal_compat.h', 'metal_kernels.h'):
        with open(os.path.join(_HERE, name), 'rb') as f:
            h.update(f.read())
    for fn in (real_literal, _qualify_pointers, _struct_names, lower_to_msl, build_msl_source,
               _phase_record_table, _drop_functions, _dispatch_source, equilibrium_source):
        h.update(inspect.getsource(fn).encode())
    h.update((_FLOAT_LITERAL.pattern + repr(_BASE_TYPES)).encode())
    return h.hexdigest()[:12]


def equilibrium_module_source(source_path, make_source, sizes, extra_thread_doubles=0):
    """Metal source of a system's equilibrium module (generated with
    `make_source()` and cached at `source_path`), or None when the system
    cannot run on a Metal GPU thread: its per-condition solver state exceeds
    Metal's per-thread stack. Such systems use the C++ solver; that outcome
    is remembered next to the cache so the check is not repeated."""
    import logging
    marker = str(source_path) + '.unsupported'
    if os.path.exists(marker):
        return None
    stack = thread_stack_estimate(sizes, extra_thread_doubles)
    if stack > METAL_THREAD_STACK_BYTES:
        logging.getLogger(__name__).info(
            "system too large for the Metal solver (~%d KB of per-condition state); "
            "using the C++ solver", stack // 1024)
        return None
    if os.path.exists(source_path):
        with open(source_path) as f:
            return f.read()
    src = make_source()
    with open(source_path, 'w') as f:
        f.write(src)
    return src


class EquilibriumModule:
    """A compiled Metal equilibrium module (one per system + model set)."""

    # Conditions per command buffer. A dispatch lasts about as long as its
    # slowest thread (the iteration cap bounds that), so launches should be
    # as wide as the GPU can hold; the cap also keeps each command buffer
    # well inside macOS's GPU watchdog.
    MAX_DISPATCH = 1 << 15

    def __init__(self, msl_source):
        self.runtime = runtime()
        self.library = self.runtime.compile(msl_source)
        self.solver = self.runtime.kernel(self.library, 'pycgpu_equilibrium_kernel')

    def solve(self, *, system_spec, condition_args, results, initial_phase_data,
              grid_data, grid_block_indices, grid_block_stride_bytes,
              num_conditions, condition_stride, python_max_statevars,
              initial_phase_data_stride, system_spec_stride, max_solver_iterations,
              jansson=None):
        """Solve every condition; `results` (float64, host) is filled in place,
        and so is `jansson` ((num_conditions, PYJAN_COND_STRIDE) float64) for
        modules built with the Jansson epilogue."""
        rt = self.runtime
        b_spec = rt.buffer_from(system_spec)
        b_cond = rt.buffer_from(condition_args)
        b_ipd = rt.buffer_from(initial_phase_data)
        b_res = rt.buffer(results.nbytes)
        has_grid = grid_data is not None
        b_grid = rt.buffer_from(grid_data) if has_grid else rt.buffer(16)
        b_gbi = (rt.buffer_from(np.ascontiguousarray(grid_block_indices, dtype=np.int32))
                 if has_grid and grid_block_indices is not None else rt.buffer(16))
        buffers = [b_spec, b_cond, b_res, b_ipd, b_grid, b_gbi]
        if jansson is not None:
            b_jansson = rt.buffer(jansson.nbytes)
            buffers.append(b_jansson)
        tg = min(self.solver.max_threads, 32)  # one SIMD group; larger groups measured no faster
        start = 0
        while start < num_conditions:
            n = min(self.MAX_DISPATCH, num_conditions - start)
            params = np.zeros(1, dtype=[
                ('num_conditions', 'i4'), ('condition_offset', 'i4'), ('condition_stride', 'i4'),
                ('python_max_statevars', 'i4'), ('initial_phase_data_stride', 'i4'),
                ('system_spec_stride', 'i4'), ('max_solver_iterations', 'i4'), ('has_grid', 'i4'),
                ('grid_block_stride_bytes', 'i8')])
            params[0] = (num_conditions, start, condition_stride, python_max_statevars,
                         initial_phase_data_stride, system_spec_stride, max_solver_iterations,
                         int(has_grid), int(grid_block_stride_bytes) if has_grid else 0)
            seconds = self.solver(buffers, params.tobytes(), n, tg)
            if os.environ.get('PYCGPU_TIME'):
                print(f"[METAL] dispatch {start}..{start + n}: {seconds:.3f} s GPU")
            start += n
        results[...] = b_res.view(np.float64)[:results.size].reshape(results.shape)
        if jansson is not None:
            jansson[...] = b_jansson.view(np.float64)[:jansson.size].reshape(jansson.shape)


class HybridSolver:
    """Solves a batch on the GPU and the CPU at once.

    The Metal kernel runs in emulated double precision and one GPU thread is
    much slower than a CPU core, so the GPU pays off only for wide batches
    and only for conditions that converge quickly. Each call therefore

    * splits the batch between the GPU and the C++ solver in proportion to
      their measured throughput (small batches stay on the CPU),
    * runs the GPU part with a small iteration cap while the CPU threads
      solve their share, and
    * reruns every condition the GPU pass leaves unconverged or capped on
      the C++ solver at the full iteration budget.

    The GPU module is compiled in a background thread when the first batch
    large enough to use it arrives (tens of seconds to a few minutes the first
    time a system is seen; macOS caches the result); batches run on the CPU
    until it is ready. Systems only ever solved in small batches never pay
    for a GPU compile.
    """

    # A GPU pass takes a roughly fixed minimum time (its slowest SIMD group),
    # so the GPU only pays off on batches the CPU needs clearly longer for;
    # smaller batches stay on the CPU. Measured crossover on an Apple M4:
    # ~30k Al-Ni conditions (PYCGPU_METAL_MIN_BATCH overrides).
    MIN_GPU_BATCH = 32768

    def __init__(self, cpu_lib, msl_source, unsupported_marker):
        self.cpu_lib = cpu_lib
        self.gpu_share = 0.3   # refined after every call from measured rates
        self._module = None
        self._source = msl_source
        self._marker = unsupported_marker
        self._ready = threading.Event()
        self._started = False
        self._lock = threading.Lock()

    def _start_compile(self):
        with self._lock:
            if self._started:
                return
            self._started = True
        threading.Thread(target=self._compile, args=(self._source,), daemon=True,
                         name='pycalphad-metal-compile').start()
        if os.environ.get('PYCGPU_METAL_SYNC_COMPILE'):
            self._ready.wait()

    def _compile(self, source):
        import logging
        import warnings
        try:
            self._module = EquilibriumModule(source)
        except MetalResourceError as e:
            logging.getLogger(__name__).info("system too large for the Metal solver (%s)", e)
            with open(self._marker, 'w') as f:
                f.write(str(e))
        except Exception as e:
            warnings.warn(f"Metal kernel compilation failed; this system runs on the C++ "
                          f"solver ({type(e).__name__}: {str(e)[:300]})")
        finally:
            self._ready.set()

    def gpu_module(self, wait=False):
        """The compiled GPU module, or None if it is unavailable or (with
        wait=False) still compiling."""
        if wait:
            self._start_compile()
            self._ready.wait()
        return self._module if self._ready.is_set() else None

    # Conditions with a prescribed mole fraction this close to 0 or 1 run on
    # the C++ solver: the dilute component's amount and chemical potential
    # need more than the GPU's ~48-bit precision.
    DILUTE_LIMIT = 1e-6

    def solve_batch(self, *, sizes, system_spec, condition_args, results, initial_phase_data,
                    grid_data, grid_block_indices, grid_block_stride_bytes, num_conditions,
                    condition_stride, python_max_statevars, initial_phase_data_stride,
                    system_spec_stride, results_per_condition, gpu_iterations, max_iterations,
                    jansson_stride=0, verbose=False):
        """Solve `num_conditions` conditions, writing `results` in place.

        With ``jansson_stride`` (modules built with the Jansson epilogue),
        `results` holds the result records followed by one delta block of
        that many doubles per condition."""
        import time
        from pycalphad.gpu.cpu_backend import solve_rows
        n = int(num_conditions)
        rpc = results_per_condition
        jansson = (results[n * rpc:n * (rpc + jansson_stride)].reshape(n, jansson_stride)
                   if jansson_stride else None)
        results = results[:n * rpc]
        worth_it = n >= int(os.environ.get('PYCGPU_METAL_MIN_BATCH', self.MIN_GPU_BATCH))
        if worth_it:
            self._start_compile()
        module = self.gpu_module()
        share = float(os.environ.get('PYCGPU_METAL_GPU_SHARE', self.gpu_share))
        layout = dict(grid_data=grid_data, grid_block_stride_bytes=grid_block_stride_bytes,
                      condition_stride=condition_stride, python_max_statevars=python_max_statevars,
                      initial_phase_data_stride=initial_phase_data_stride,
                      system_spec_stride=system_spec_stride)

        gpu_rows = np.empty(0, dtype=np.int64)
        if module is not None and worth_it:
            # Prescribed X(component) conditions are the identity rows of the
            # specification's mole-fraction constraint block:
            # [nsv, ncomp, N, MU[MC], coefficients[MFC][MC], rhs[MFC], count, ...]
            mc, mfc = int(sizes['MAX_COMPONENTS']), int(sizes['MAX_FIXED_MOLE_FRACTION_CONDITIONS'])
            spec = system_spec.reshape(n, system_spec_stride)
            coef = spec[:, 3 + mc:3 + mc + mfc * mc].reshape(n, mfc, mc)
            rhs = spec[:, 3 + mc + mfc * mc:3 + mc + mfc * mc + mfc]
            count = spec[:, 3 + mc + mfc * mc + mfc].astype(int)
            identity = ((np.abs(coef - 1.0) < 1e-12).sum(axis=2) == 1) & \
                       ((np.abs(coef) > 1e-12).sum(axis=2) == 1)
            active = np.arange(mfc)[None, :] < count[:, None]
            near_edge = (active & identity & ((rhs < self.DILUTE_LIMIT)
                                              | (rhs > 1.0 - self.DILUTE_LIMIT))).any(axis=1)
            # the balance component (one minus the prescribed fractions) too
            balance = 1.0 - np.where(active & identity, rhs, 0.0).sum(axis=1)
            near_edge |= ((active & identity).sum(axis=1) == count) & (balance < self.DILUTE_LIMIT)
            eligible = np.nonzero(~near_edge)[0]
            gpu_rows = eligible[:int(round(eligible.size * share))]
        cpu_rows = np.setdiff1d(np.arange(n), gpu_rows, assume_unique=True)

        def cpu_solve(rows):
            solve_rows(self.cpu_lib, sizes=sizes, rows=rows, system_spec=system_spec,
                       condition_args=condition_args, initial_phase_data=initial_phase_data,
                       results=results, grid_block_indices=grid_block_indices,
                       results_per_condition=rpc, max_solver_iterations=max_iterations,
                       jansson=jansson, **layout)

        # The CPU share runs in a worker thread (the C++ call releases the
        # GIL) while this thread drives the GPU.
        timing = {}
        error = []

        def cpu_part():
            t = time.perf_counter()
            try:
                cpu_solve(cpu_rows)
            except BaseException as e:  # re-raised on the calling thread
                error.append(e)
            timing['cpu'] = time.perf_counter() - t

        worker = threading.Thread(target=cpu_part, name='pycalphad-cpu-share')
        worker.start()
        n_gpu = gpu_rows.size
        flagged = np.empty(0, dtype=np.int64)
        if n_gpu:
            t = time.perf_counter()
            take = lambda flat, stride: np.ascontiguousarray(flat.reshape(n, stride)[gpu_rows]).reshape(-1)
            gpu_results = np.zeros(n_gpu * rpc)
            gpu_jansson = np.zeros((n_gpu, jansson_stride)) if jansson_stride else None
            module.solve(
                system_spec=take(system_spec, system_spec_stride),
                condition_args=take(condition_args, condition_stride),
                results=gpu_results,
                initial_phase_data=take(initial_phase_data, initial_phase_data_stride),
                grid_block_indices=(np.asarray(grid_block_indices)[gpu_rows]
                                    if grid_block_indices is not None else None),
                num_conditions=n_gpu, max_solver_iterations=gpu_iterations,
                jansson=gpu_jansson, **layout)
            timing['gpu'] = time.perf_counter() - t
            gv = gpu_results.reshape(n_gpu, rpc)
            mc, mp = int(sizes['MAX_COMPONENTS']), int(sizes['MAX_PHASES'])
            flagged = np.nonzero((gv[:, 1 + mc + mp] < 0.5) | (gv[:, 5 + mc + mp] > 0.5))[0]
        worker.join()
        if error:
            raise error[0]
        if n_gpu:
            results.reshape(n, rpc)[gpu_rows] = gv
            if jansson_stride:
                jansson[gpu_rows] = gpu_jansson
            if flagged.size:
                cpu_solve(gpu_rows[flagged])
            # Rebalance for the next call: per-condition rates of the two
            # sides (GPU conditions that needed the CPU tail count against it).
            if cpu_rows.size:
                gpu_rate = max(n_gpu - flagged.size, 1) / max(timing['gpu'], 1e-6)
                cpu_rate = cpu_rows.size / max(timing.get('cpu', 1e-6), 1e-6)
                self.gpu_share = float(np.clip(gpu_rate / (gpu_rate + cpu_rate), 0.05, 0.95))
        if verbose or os.environ.get('PYCGPU_TIME'):
            print(f"[METAL] {n_gpu} conditions on the GPU ({timing.get('gpu', 0):.2f} s, "
                  f"{flagged.size} finished on the CPU), {cpu_rows.size} on the CPU "
                  f"({timing.get('cpu', 0):.2f} s); next GPU share {self.gpu_share:.2f}")
