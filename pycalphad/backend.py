"""
Computation-backend selection for pycalphad.

Pick a backend once and every ``equilibrium()``, ``calculate()`` and
``Workspace`` call after that uses it::

    import pycalphad
    pycalphad.set_backend('auto')      # fastest backend available here
    eq = pycalphad.equilibrium(dbf, comps, phases, conds)

Backends
--------
``'default'`` (aliases ``'cpu'``, ``'reference'``)
    The reference implementation (Cython + LAPACK). Always available.
``'cpp'`` (aliases ``'c++'``, ``'openmp'``)
    The generated solver compiled for the host with the system C++
    compiler, using all available CPU cores by default.
``'cuda'``
    The generated solver on an NVIDIA (or ROCm) GPU through CuPy.
    ``'cuda-fast'`` adds a lockstep first pass that pays off on cards with
    full-rate FP64.
``'metal'``
    The generated solver on an Apple-silicon GPU. Apple GPUs have no FP64
    hardware, so the kernels run in emulated double precision; conditions
    the GPU pass does not settle quickly are finished by the C++ solver.
``'gpu'``
    The GPU backend for this machine: ``'cuda'`` when CuPy finds a device,
    otherwise ``'metal'`` on Apple silicon.
``'auto'``
    The fastest backend available: ``'cuda'`` if usable, else ``'cpp'``,
    else ``'default'``.

Problems a compiled backend cannot handle exactly (custom ``Model``
subclasses, unsupported condition types, vectorized parameter sweeps, ...)
run on the reference implementation automatically, so a backend can be set
globally without changing any other code.

Scoped and per-call selection::

    with pycalphad.backend('cpp', threads=4):
        eq = pycalphad.equilibrium(...)

    eq = pycalphad.equilibrium(..., backend='metal')

The initial default comes from the ``PYCALPHAD_BACKEND`` environment
variable when it is set.

Options
-------
``threads`` (int)
    Worker threads for the C++ solver and starting-point hull. Default: all
    cores available to the process. Use ``threads=1`` when you already run
    pycalphad in parallel processes (e.g. ESPEI MCMC walkers). Results are
    bit-identical for any thread count.
``max_iters`` (int)
    Newton-iteration budget per condition (reference value: 1000).
``chunk`` (int)
    Conditions per solver launch; bounds memory for very large batches.
    Chosen automatically from free memory by default.
``pass1_iters`` (int)
    Two-pass solving: run every condition with this iteration cap first and
    rerun the ones that hit it at the full budget (results are identical to
    a single full-budget pass). CUDA default 50; 0 disables.
``robust_phase_removal`` (bool)
    Count phase removals during composition-set consolidation toward the
    per-set removal budget, which terminates add/remove cycles between
    near-duplicate composition sets. Default True.
``hull_procs``, ``calc_procs`` (int)
    Worker processes for the reference starting-point hull and for grid
    sampling over temperatures (1 = serial, 0 = automatic).
"""
import contextvars
import os
import platform
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace

__all__ = ['set_backend', 'get_backend', 'backend', 'available_backends',
           'AcceleratedCapabilityError', 'BackendOptions']


class AcceleratedCapabilityError(RuntimeError):
    """The problem is outside a compiled backend's capabilities (e.g. a custom
    Model subclass). Raised inside the accelerated path; the dispatch then runs
    the reference implementation instead.

    Any OTHER exception from a compiled backend (a compiler failure, a kernel
    error) propagates to the caller rather than silently degrading to the
    reference solver. Set PYCGPU_FALLBACK=1 to fall back on any error."""


@dataclass(frozen=True)
class BackendOptions:
    """Tunables for the compiled backends (see the module docstring)."""
    threads: int = None
    max_iters: int = 1000
    chunk: int = None
    pass1_iters: int = None
    robust_phase_removal: bool = True
    hull_procs: int = 1
    calc_procs: int = 1

    @classmethod
    def from_kwargs(cls, options):
        known = {f.name for f in fields(cls)}
        unknown = set(options) - known
        if unknown:
            raise TypeError(f"Unknown backend option(s) {sorted(unknown)}; "
                            f"valid options are {sorted(known)}")
        return cls(**options)


# Canonical names: 'default', 'cpp', 'cuda', 'cuda-fast', 'metal', plus the
# selectors 'gpu' and 'auto' that resolve to one of them.
_ALIASES = {
    'default': 'default', 'cpu': 'default', 'reference': 'default',
    'cpp': 'cpp', 'c++': 'cpp', 'openmp': 'cpp',
    'cuda': 'cuda', 'cuda-fast': 'cuda-fast', 'gpu-fast': 'cuda-fast',
    'metal': 'metal', 'mps': 'metal',
    'gpu': 'gpu', 'auto': 'auto',
}
_DISPLAY = {'default': 'default', 'cpp': 'cpp', 'cuda': 'cuda', 'cuda-fast': 'cuda-fast',
            'metal': 'metal'}

_global = {'name': None, 'options': BackendOptions()}
_scoped = contextvars.ContextVar('pycalphad_backend', default=None)
_probe_cache = {}


def _normalize(name):
    try:
        return _ALIASES[str(name).strip().lower()]
    except KeyError:
        raise ValueError(f"Unknown backend {name!r}. Valid choices: "
                         "'auto', 'default', 'cpp', 'gpu', 'cuda', 'metal'.") from None


# ---------------------------------------------------------------------------
# availability probes (cached: they can be slow, e.g. building the Metal bridge)
# ---------------------------------------------------------------------------

def _probe(canonical):
    """None if the backend is usable here, else a reason string."""
    if canonical in _probe_cache:
        return _probe_cache[canonical]
    reason = None
    if canonical == 'default':
        pass
    elif canonical == 'cpp':
        if shutil.which('clang++' if platform.system() == 'Darwin' else 'g++') is None \
                and shutil.which('clang++') is None:
            reason = ("needs a C++17 compiler on PATH (Xcode command line tools on macOS, "
                      "g++ on Linux, MinGW-w64 or WSL on Windows)")
    elif canonical in ('cuda', 'cuda-fast'):
        try:
            import cupy as cp
            cp.cuda.Device().compute_capability
        except Exception as e:
            reason = f"needs CuPy and a CUDA/ROCm device ({type(e).__name__}: {e})"
    elif canonical == 'metal':
        if platform.system() != 'Darwin' or platform.machine() != 'arm64':
            reason = "needs macOS on Apple silicon"
        elif _probe('cpp') is not None:
            reason = "needs the Xcode command line tools (clang++)"
        else:
            try:
                from pycalphad.gpu.metal_backend import runtime
                runtime()
            except Exception as e:
                reason = f"Metal device unavailable ({e})"
    _probe_cache[canonical] = reason
    return reason


def _resolve(canonical):
    """Map the 'gpu'/'auto' selectors to a concrete, usable backend."""
    if canonical == 'gpu':
        for candidate in ('cuda', 'metal'):
            if _probe(candidate) is None:
                return candidate
        raise RuntimeError(
            "backend 'gpu' found no usable GPU: "
            f"cuda {_probe('cuda')}; metal {_probe('metal')}. "
            "Use set_backend('cpp') for the compiled CPU backend.")
    if canonical == 'auto':
        for candidate in ('cuda', 'cpp'):
            if _probe(candidate) is None:
                return candidate
        return 'default'
    reason = _probe(canonical)
    if reason is not None:
        raise RuntimeError(f"backend {_DISPLAY[canonical]!r} {reason}. "
                           "set_backend('auto') picks the best available backend.")
    return canonical


def available_backends():
    """{name: None if usable, else the reason it is not} for every backend."""
    return {_DISPLAY[c]: _probe(c) for c in ('default', 'cpp', 'cuda', 'metal')}


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

def set_backend(name='auto', **options):
    """Set the computation backend for all subsequent calls.

    Parameters
    ----------
    name : str
        'auto' (default), 'default', 'cpp', 'gpu', 'cuda', 'cuda-fast' or
        'metal' ('cpu' is an alias of 'default', the reference solver). The 'gpu' and 'auto' selectors are resolved immediately, so
        a missing toolchain or device is reported here rather than in the
        middle of a calculation.
    **options
        Backend tunables (see the module docstring). They replace any
        previously set options.
    """
    opts = BackendOptions.from_kwargs(options)
    _global['name'] = _resolve(_normalize(name))
    _global['options'] = opts
    return _DISPLAY[_global['name']]


def _current():
    """(canonical backend, BackendOptions) in effect for the calling context."""
    scoped = _scoped.get()
    if scoped is not None:
        return scoped
    if _global['name'] is None:
        env = os.environ.get('PYCALPHAD_BACKEND')
        if env:
            # Resolved lazily: importing pycalphad must not fail because the
            # environment names a backend this machine lacks.
            try:
                _global['name'] = _resolve(_normalize(env))
            except RuntimeError as e:
                import warnings
                warnings.warn(f"PYCALPHAD_BACKEND={env!r} is not usable here ({e}); "
                              "using the reference implementation.")
                _global['name'] = 'default'
        else:
            _global['name'] = 'default'
    return _global['name'], _global['options']


def get_backend():
    """Return ``(name, options)`` for the backend in effect.

    ``name`` is the canonical backend ('default', 'cpp', 'cuda', 'cuda-fast'
    or 'metal'); ``options`` is a dict of the non-default options.
    """
    name, opts = _current()
    defaults = BackendOptions()
    return name, {f.name: getattr(opts, f.name) for f in fields(opts)
                  if getattr(opts, f.name) != getattr(defaults, f.name)}


@contextmanager
def backend(name, **options):
    """Use a backend for the duration of a ``with`` block.

    The selection is context-local (safe with threads and asyncio)::

        with pycalphad.backend('cpp', threads=4):
            eq = pycalphad.equilibrium(...)
    """
    token = _scoped.set((_resolve(_normalize(name)), BackendOptions.from_kwargs(options)))
    try:
        yield
    finally:
        _scoped.reset(token)


@contextmanager
def _use(name, options):
    """Internal: run a block with an already-resolved backend and options."""
    token = _scoped.set((name, options))
    try:
        yield
    finally:
        _scoped.reset(token)


def _resolve_per_call(name, **overrides):
    """Backend and options for one call: a per-call `backend=` keyword wins
    over the scoped/global selection; `overrides` (non-None values) replace
    individual options."""
    if name is None:
        canonical, opts = _current()
    else:
        canonical = _resolve(_normalize(name))
        opts = _current()[1]
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if overrides:
        opts = replace(opts, **overrides)
    return canonical, opts


def available_cpus():
    """CPU cores this process may use (respects affinity masks, e.g. the
    cores a cluster job was allocated)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def solver_threads():
    """Threads the C++ solver should use in the current context."""
    threads = _current()[1].threads
    if threads is None:
        env = os.environ.get('PYCGPU_CPU_THREADS')
        threads = int(env) if env else available_cpus()
    return int(threads)
