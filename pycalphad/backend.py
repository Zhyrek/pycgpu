"""
Global computation-backend selection for pycalphad.

pycalphad ships three execution backends for supported routines (currently
``equilibrium``; ``calculate`` energy evaluation where available):

* ``"default"`` — the reference CPU implementation (Cython + LAPACK).
* ``"c++"``    — the C++/OpenMP backend: the generated solver compiled with
  the host compiler. Same algorithm, typically 40-80x faster single-core.
  Thread count follows OMP_NUM_THREADS.
* ``"gpu"``    — the CUDA backend (CuPy RawModule), one condition per thread.

Usage::

    import pycalphad
    pycalphad.set_backend("c++")                 # global default from here on
    eq = pycalphad.equilibrium(dbf, comps, phases, conds)

    with pycalphad.backend("gpu", robust_phase_removal=True):
        eq = pycalphad.equilibrium(...)          # scoped override

The per-call ``backend=`` keyword of ``equilibrium`` always wins over the
global setting. The initial default may be set with the ``PYCALPHAD_BACKEND``
environment variable. Backend availability is validated eagerly at
``set_backend`` time so misconfiguration fails immediately with a clear
message instead of minutes into a calculation.

Accepted option keywords (apply to accelerated backends):

* ``robust_phase_removal`` (bool): count consolidation phase-removals toward
  the per-compset removal budget (terminates add/collapse cycles).
* ``chunk`` (int): conditions per solver launch (bounds work-array memory;
  required for batches beyond VRAM/RAM).
* ``pass1_iters`` (int): enable the two-pass driver with this pass-1
  iteration cap (cap-touching conditions are rerun at the full budget;
  results are bit-identical to a single full-budget run).
* ``max_iters`` (int): full Newton-iteration budget (CPU parity value 1000).
* ``hull_procs`` (int): worker processes for the starting-point hull
  (1 = serial).
"""
import os
import shutil
from contextlib import contextmanager

__all__ = ['set_backend', 'get_backend', 'backend']

# Canonical internal names are 'default', 'cpp', 'cuda'.
_ALIASES = {
    'default': 'default', 'cpu': 'default', 'reference': 'default',
    'c++': 'cpp', 'cpp': 'cpp', 'openmp': 'cpp',
    'gpu': 'cuda', 'cuda': 'cuda',
}

# Mapping of option keywords to the environment tunables the accelerated
# pipeline reads. Values are stringified; None/absent options leave the
# corresponding environment variable untouched.
_OPTION_ENV = {
    'robust_phase_removal': ('PYCGPU_ROBUST', lambda v: '1' if v else ''),
    'chunk': ('PYCGPU_CHUNK', str),
    'pass1_iters': ('PYCGPU_PASS1_ITERS', str),
    'max_iters': ('PYCGPU_MAXITER', str),
    'hull_procs': ('PYCGPU_HULL_PROCS', str),
}

_state = {'backend': None, 'options': {}}


def _normalize(name):
    try:
        return _ALIASES[str(name).strip().lower()]
    except KeyError:
        raise ValueError(
            f"Unknown backend {name!r}. Valid choices: 'default', 'c++', 'gpu'.")


def _validate(canonical):
    """Fail fast at set time with an actionable message."""
    if canonical == 'cpp':
        if shutil.which('g++') is None and shutil.which('clang++') is None:
            raise RuntimeError(
                "backend 'c++' needs a C++17/OpenMP compiler (g++ or clang++) on PATH. "
                "Install one (e.g. `apt install g++`, `brew install gcc`, or MinGW-w64/WSL "
                "on Windows), or use set_backend('default').")
    elif canonical == 'cuda':
        try:
            import cupy as cp
            cp.cuda.Device().compute_capability
        except Exception as e:
            raise RuntimeError(
                "backend 'gpu' needs CuPy and a working CUDA device "
                f"(import/probe failed: {e!r}). Install cupy matching your CUDA "
                "toolkit, or use set_backend('c++') for the host-only accelerated "
                "backend.") from e


def _validate_options(options):
    unknown = set(options) - set(_OPTION_ENV)
    if unknown:
        raise TypeError(f"Unknown backend option(s): {sorted(unknown)}. "
                        f"Valid options: {sorted(_OPTION_ENV)}")


def set_backend(name, **options):
    """Set the global computation backend (and optional tunables).

    Parameters
    ----------
    name : str
        'default' (reference CPU), 'c++' (host OpenMP), or 'gpu' (CUDA).
    **options
        Backend tunables; see the module docstring. Options replace any
        previously set options entirely.
    """
    canonical = _normalize(name)
    _validate_options(options)
    if canonical != 'default':
        _validate(canonical)
    _state['backend'] = canonical
    _state['options'] = dict(options)


def get_backend():
    """Return (canonical_backend_name, options_dict) currently in effect.

    Resolution order: last ``set_backend`` call, else the
    ``PYCALPHAD_BACKEND`` environment variable, else 'default'.
    """
    if _state['backend'] is not None:
        return _state['backend'], dict(_state['options'])
    env = os.environ.get('PYCALPHAD_BACKEND')
    if env:
        # Environment-selected backends are validated lazily (at first use):
        # import of pycalphad must not fail on a machine where the env var is
        # set but the backend is unavailable.
        return _normalize(env), {}
    return 'default', {}


@contextmanager
def backend(name, **options):
    """Context manager: temporarily switch the global backend.

    ::

        with pycalphad.backend("gpu", chunk=8192):
            eq = pycalphad.equilibrium(...)
    """
    prev = (_state['backend'], _state['options'])
    set_backend(name, **options)
    try:
        yield
    finally:
        _state['backend'], _state['options'] = prev


@contextmanager
def _option_env(options):
    """Apply backend options as environment tunables for the duration of a call.

    Explicit per-call kwargs and pre-existing environment variables set by the
    user are NOT overridden: options only fill in unset variables, so the
    priority is call kwarg > user env > set_backend option.
    """
    applied = {}
    try:
        for key, value in options.items():
            env_name, conv = _OPTION_ENV[key]
            if env_name in os.environ:
                continue  # user's explicit environment wins
            sval = conv(value)
            if sval:
                os.environ[env_name] = sval
                applied[env_name] = None
        yield
    finally:
        for env_name in applied:
            os.environ.pop(env_name, None)
