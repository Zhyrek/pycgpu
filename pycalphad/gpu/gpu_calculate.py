"""
Accelerated grid-energy evaluation for pycalphad.calculate().

The heavy part of calculate() is evaluating the phase energy over the sampled
points; the sampling and dataset assembly stay on the CPU. This module builds
(or loads from cache) the same generated kernel module the equilibrium solver
uses and exposes a per-phase evaluator that computes GM over a dof matrix
with the generated obj functions:

* backend 'cpp':  pycgpu_cpu_grid_eval, single-threaded loop over points,
  zero copies.
* backend 'cuda': grid_eval_kernel, one point per thread.

Energies computed by the generated functions agree with the reference
compiled callables to floating-point-noise level (~1e-12 relative), the same
class as pycalphad's own cross-process nondeterminism; the 'default' backend
is unchanged.
"""
import ctypes
import hashlib
import os
from types import SimpleNamespace

import numpy as np

try:
    import cupy as cp
except Exception:
    cp = None

_evaluator_cache = {}


def _build_module(backend_name, shim, verbose=False):
    from pycalphad.gpu.gpu_codegen import (
        compute_dynamic_kernel_sizes, _generate_c_code_for_phase_models,
        _generate_full_gpu_source, _unique_models_for_gpu)
    from pycalphad.gpu.gpu_equilibrium import _kernel_cache_dir

    dynamic_sizes = compute_dynamic_kernel_sizes(shim)
    define_flags = [f'-D{k}={v}' for k, v in dynamic_sizes.items()]

    gpu_dir = os.path.dirname(os.path.abspath(__file__))
    hasher = hashlib.md5()
    for hdr in ("svd.c", "phase_rec.h", "comp_set.h", "lu_solver.h", "hyperplane.h",
                "minimizer.h", "eqsolver.h", "gpu_codegen.py"):
        with open(os.path.join(gpu_dir, hdr), "rb") as f:
            hasher.update(f.read())
    # Fingerprint the model energy expressions: phase/component names alone
    # collide between different assessments of the same system.
    model_hasher = hashlib.md5()
    for ph in sorted(shim.phases):
        model_hasher.update(ph.encode())
        model_hasher.update(str(shim.models[ph].GM).encode())
    key_input = "|".join([
        "calc", backend_name,
        ",".join(sorted(shim.phases)),
        ",".join(sorted(c.name for c in shim.components)),
        # The generated functions bake in the statevar->dof-column mapping
        "statevars:" + ",".join(str(sv) for sv in shim.phase_record_factory.state_variables),
        "models:" + model_hasher.hexdigest(),
        str(sorted(dynamic_sizes.items())),
        hasher.hexdigest(),
    ])
    cache_key = hashlib.md5(key_input.encode()).hexdigest()

    cache_dir = _kernel_cache_dir()
    cache_file = cache_dir / f"{cache_key}.cu"
    if cache_file.exists():
        full_source = cache_file.read_text()
    else:
        model_funcs_c, pr_init_calls_c, unique_models, _ = \
            _generate_c_code_for_phase_models(shim, include_hess=True, validate=False)
        full_source = _generate_full_gpu_source(shim, model_funcs_c, pr_init_calls_c,
                                                len(unique_models))
        cache_file.write_text(full_source)

    if backend_name == 'cpp':
        from pycalphad.gpu.cpu_backend import build_cpu_library
        lib = build_cpu_library(full_source, define_flags, cache_dir=str(cache_dir),
                                verbose=verbose)
        fn = lib.pycgpu_cpu_grid_eval
        fn.restype = None
        fn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_longlong, ctypes.c_int]
        return ('cpp', fn)
    else:
        if cp is None:
            raise RuntimeError("backend 'gpu' requires CuPy")
        module = cp.RawModule(code=full_source,
                              options=tuple(['-std=c++11', '-O2'] + define_flags),
                              backend='nvcc')
        init_k = module.get_function('init_all_gpu_phase_records')
        init_k((1,), (1,), ())
        cp.cuda.runtime.deviceSynchronize()
        return ('cuda', module.get_function('grid_eval_kernel'))


def get_grid_evaluator(backend_name, components, phases, models,
                       phase_record_factory, verbose=False):
    """Return evaluate(phase_name, dof_2d, out_1d) for the given system.

    Modules are cached per (backend, phases, components, sizes); the first
    call for a system compiles the kernel (one-time, disk-cached thereafter).
    Returns None if the system cannot be built (caller falls back to CPU).
    """
    from pycalphad.model import Model as _PlainModel
    for _ph in phases:
        if type(models[_ph]) is not _PlainModel:
            raise RuntimeError(
                f"accelerated calculate() supports plain Model instances only; "
                f"phase {_ph} uses {type(models[_ph]).__name__}")
    shim = SimpleNamespace(
        components=list(components),
        phases=list(phases),
        models=models,
        phase_record_factory=phase_record_factory,
        conditions={},
        verbose=verbose,
    )
    # In-process cache must also fingerprint the model expressions (same
    # collision as the disk key: two assessments sharing phase names).
    _mh = hashlib.md5()
    for _ph in sorted(shim.phases):
        _mh.update(_ph.encode())
        _mh.update(str(models[_ph].GM).encode())
    cache_id = (backend_name, tuple(sorted(shim.phases)),
                tuple(sorted(c.name for c in shim.components)),
                tuple(str(sv) for sv in shim.phase_record_factory.state_variables),
                _mh.hexdigest())
    if cache_id in _evaluator_cache:
        entry = _evaluator_cache[cache_id]
    else:
        entry = _build_module(backend_name, shim, verbose=verbose)
        _evaluator_cache[cache_id] = entry

    from pycalphad.gpu.gpu_codegen import _unique_models_for_gpu
    _, name_to_idx = _unique_models_for_gpu(shim, validate=False)
    kind, fn = entry

    # The single-threaded C++ path has near-zero per-call overhead and matches
    # or beats the reference LLVM callables at every measured size, so it is
    # always used. The CUDA path pays H2D+D2H transfer per call and only wins
    # on very large point sets. Override with PYCGPU_CALC_MIN_POINTS.
    default_min = 0 if kind == 'cpp' else 1_000_000
    min_points = int(os.environ.get('PYCGPU_CALC_MIN_POINTS', default_min))

    # Fit parameters (factory.param_values) ride in trailing dof slots; read
    # LIVE at call time so per-step updates (e.g. ESPEI MCMC poking
    # param_values in place) take effect without rebuilding anything.
    n_params = len(getattr(phase_record_factory, 'param_symbols', []) or [])

    def evaluate(phase_name, dof, out):
        model_idx = name_to_idx[phase_name]
        n_points = dof.shape[0]
        if n_params:
            pv = np.asarray(phase_record_factory.param_values, dtype=np.float64).reshape(-1)[:n_params]
            dof = np.concatenate([dof, np.broadcast_to(pv, (n_points, n_params))], axis=1)
        dof_stride = dof.shape[1]
        dof_c = np.ascontiguousarray(dof, dtype=np.float64)
        if kind == 'cpp':
            fn(int(model_idx), dof_c.ctypes.data, out.ctypes.data,
               n_points, dof_stride)
        else:
            d_dof = cp.asarray(dof_c)
            d_out = cp.empty(n_points, dtype=cp.float64)
            tpb = 128
            blocks = (n_points + tpb - 1) // tpb
            fn((blocks,), (tpb,),
               (np.int32(model_idx), d_dof, d_out,
                np.int64(n_points), np.int32(dof_stride)))
            cp.cuda.runtime.deviceSynchronize()
            out[:] = cp.asnumpy(d_out)

    evaluate.min_points = min_points
    return evaluate
