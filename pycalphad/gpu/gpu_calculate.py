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
import re
from types import SimpleNamespace

import numpy as np

try:
    import cupy as cp
except Exception:
    cp = None

_evaluator_cache = {}

# Strength-reduced pow, same as the equilibrium module's CUDA path (see
# _generate_full_gpu_source). Unlike the solver — which keeps libm pow on the
# C++ backend for bit-parity with the reference minimizer's iteration path —
# property evaluation is a single expression with no iteration to diverge, so
# both backends take the fast integer-exponent chain (<=1-2 ulp from libm,
# the same eps class as symengine's own LLVM lowering of integer powers).
_POW_SHIM = r"""
__device__ inline double pycgpu_pow(double b, double e) {
    int ei = (int)e;
    if ((double)ei == e && ei > -64 && ei < 64) {
        unsigned int n = ei < 0 ? (unsigned int)(-ei) : (unsigned int)ei;
        double r = 1.0, p = b;
        while (n) { if (n & 1u) r *= p; p *= p; n >>= 1u; }
        return ei < 0 ? 1.0 / r : r;
    }
    return (pow)(b, e);
}
#define pow(b, e) pycgpu_pow((b), (e))
"""

_PROP_CPU_DRIVER = r"""
extern "C" void pycgpu_cpu_grid_eval(int model_idx, const double* dof, double* out,
                                     long long n_points, int dof_stride)
{
    for (long long i = 0; i < n_points; ++i) {
        out[i] = pycgpu_eval_prop(model_idx, &dof[i * (long long)dof_stride]);
    }
}
extern "C" void pycgpu_cpu_grid_eval_grad(int model_idx, const double* dof, double* out,
                                          long long n_points, int dof_stride, int grad_len)
{
    for (long long i = 0; i < n_points; ++i) {
        pycgpu_eval_prop_grad(model_idx, &dof[i * (long long)dof_stride],
                              &out[i * (long long)grad_len]);
    }
}
"""


def _property_expr(model, output, param_symbols):
    """The model expression for `output` with reference-path semantics.

    Mirrors PhaseRecordFactory.get_phase_property: undefined non-state-variable
    symbols (other than fit parameters) are forced to zero before compilation.
    """
    import symengine
    from pycalphad import variables as v
    expr = getattr(model, output, None)
    if expr is None:
        raise RuntimeError(f"Model property {output} is not defined")
    expr = symengine.sympify(expr)
    undefs = {x for x in expr.free_symbols if not isinstance(x, v.StateVariable)} - set(param_symbols)
    if undefs:
        expr = expr.xreplace({x: 0. for x in undefs})
    return expr


def _generate_property_source(shim, output):
    """Standalone module source evaluating `output` over dof rows.

    Unlike the GM path (which reuses the full equilibrium module and its
    PhaseRecord obj functions), non-GM outputs get a lightweight module with
    just the generated property functions and the grid kernel — no solver,
    so it compiles in seconds and its cache entry is independent.
    """
    from pycalphad.gpu.gpu_codegen import (
        _unique_models_for_gpu, notebook_source_from_expr,
        notebook_model_c_func_name_prefix)

    unique_models, _ = _unique_models_for_gpu(shim, validate=False)
    param_symbols = list(getattr(shim.phase_record_factory, 'param_symbols', []) or [])
    funcs = []
    for idx, model in enumerate(unique_models):
        expr = _property_expr(model, output, param_symbols)
        func_c = notebook_source_from_expr(
            expr, "prop", model, idx, shim,
            expr_type="func", c_output_type="double", validate=False,
            verbose=shim.verbose)
        # Property expressions can contain literal NaN/inf (e.g. _MIX on
        # partitioned order/disorder models, where the reference model is
        # undefined and the reference callables evaluate to NaN); symengine
        # prints them as bare `nan`/`inf`, which is not valid C.
        func_c = re.sub(r'\bnan(\.0)?\b', '(0.0/0.0)', func_c)
        func_c = re.sub(r'\binf(\.0)?\b', '(1.0/0.0)', func_c)
        funcs.append(func_c)
        # Gradient over the full dof vector (statevars + site fractions
        # [+ params]) — the numerator side of Jansson derivatives.
        grad_c = notebook_source_from_expr(
            expr, "propgrad", model, idx, shim,
            expr_type="grad", c_output_type="void", validate=False,
            verbose=shim.verbose)
        grad_c = re.sub(r'\bnan(\.0)?\b', '(0.0/0.0)', grad_c)
        grad_c = re.sub(r'\binf(\.0)?\b', '(1.0/0.0)', grad_c)
        funcs.append(grad_c)
    cases = "\n".join(
        f"        case {idx}: return {notebook_model_c_func_name_prefix(idx)}prop(x);"
        for idx in range(len(unique_models)))
    grad_cases = "\n".join(
        f"        case {idx}: {notebook_model_c_func_name_prefix(idx)}propgrad(out, x); return;"
        for idx in range(len(unique_models)))
    return f"""
#if defined(__CUDACC_RTC__) || defined(__HIPCC_RTC__)
#define DBL_MAX 1.7976931348623157e+308
#define DBL_EPSILON 2.2204460492503131e-16
#ifndef INFINITY
#define INFINITY (1.0/0.0)
#endif
extern "C" __device__ int printf(const char*, ...);
#else
#include <float.h>
#include <math.h>
#include <stdio.h>
#endif
{_POW_SHIM}
{''.join(funcs)}
__device__ double pycgpu_eval_prop(int model_idx, const double* x) {{
    switch (model_idx) {{
{cases}
    }}
    return 0.0 / 0.0;
}}

__device__ void pycgpu_eval_prop_grad(int model_idx, const double* x, double* out) {{
    switch (model_idx) {{
{grad_cases}
    }}
}}

extern "C" {{
__global__ void grid_eval_kernel(int model_idx, const double* dof, double* out,
                                 long long n_points, int dof_stride) {{
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n_points) return;
    out[i] = pycgpu_eval_prop(model_idx, &dof[i * (long long)dof_stride]);
}}
__global__ void grid_eval_grad_kernel(int model_idx, const double* dof, double* out,
                                      long long n_points, int dof_stride, int grad_len) {{
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n_points) return;
    pycgpu_eval_prop_grad(model_idx, &dof[i * (long long)dof_stride],
                          &out[i * (long long)grad_len]);
}}
}}
"""


def _build_module(backend_name, shim, verbose=False, output='GM',
                  force_property_module=False):
    from pycalphad.gpu.gpu_codegen import (
        compute_dynamic_kernel_sizes, _generate_c_code_for_phase_models,
        _generate_full_gpu_source, _unique_models_for_gpu)
    from pycalphad.gpu.gpu_equilibrium import _kernel_cache_dir

    dynamic_sizes = compute_dynamic_kernel_sizes(shim)
    define_flags = [f'-D{k}={v}' for k, v in dynamic_sizes.items()]

    gpu_dir = os.path.dirname(os.path.abspath(__file__))
    hasher = hashlib.md5()
    for hdr in ("phase_rec.h", "comp_set.h", "hyperplane.h",
                "minimizer.h", "eqsolver.h", "gpu_codegen.py", "gpu_calculate.py"):
        with open(os.path.join(gpu_dir, hdr), "rb") as f:
            hasher.update(f.read())
    # Fingerprint the model expressions being compiled: phase/component names
    # alone collide between different assessments of the same system. For
    # non-GM outputs the property expression itself is the fingerprint.
    param_symbols = list(getattr(shim.phase_record_factory, 'param_symbols', []) or [])
    model_hasher = hashlib.md5()
    for ph in sorted(shim.phases):
        model_hasher.update(ph.encode())
        if output == 'GM' and not force_property_module:
            model_hasher.update(str(shim.models[ph].GM).encode())
        else:
            model_hasher.update(str(_property_expr(shim.models[ph], output, param_symbols)).encode())
    key_input = "|".join([
        "calc", backend_name, output,
        "propmod" if force_property_module else "solver-or-prop",
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
    elif output == 'GM' and not force_property_module:
        model_funcs_c, pr_init_calls_c, unique_models, _ = \
            _generate_c_code_for_phase_models(shim, include_hess=True, validate=False)
        full_source = _generate_full_gpu_source(shim, model_funcs_c, pr_init_calls_c,
                                                len(unique_models))
        cache_file.write_text(full_source)
    else:
        full_source = _generate_property_source(shim, output)
        cache_file.write_text(full_source)

    if backend_name == 'cpp':
        from pycalphad.gpu.cpu_backend import build_cpu_library
        lib = build_cpu_library(full_source, define_flags, cache_dir=str(cache_dir),
                                verbose=verbose,
                                driver_src=None if (output == 'GM' and not force_property_module) else _PROP_CPU_DRIVER)
        fn = lib.pycgpu_cpu_grid_eval
        fn.restype = None
        fn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_longlong, ctypes.c_int]
        gfn = None
        if output != 'GM' or force_property_module:
            gfn = lib.pycgpu_cpu_grid_eval_grad
            gfn.restype = None
            gfn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                            ctypes.c_longlong, ctypes.c_int, ctypes.c_int]
        return ('cpp', fn, gfn)
    else:
        if cp is None:
            raise RuntimeError("backend 'gpu' requires CuPy")
        from pycalphad.gpu.kernel_manager import cuda_raw_module
        module = cuda_raw_module(full_source, ['-std=c++11', '-O2'] + define_flags,
                                 verbose=verbose)
        if output == 'GM' and not force_property_module:
            init_k = module.get_function('init_all_gpu_phase_records')
            init_k((1,), (1,), ())
            cp.cuda.runtime.deviceSynchronize()
        gk = None if (output == 'GM' and not force_property_module) else module.get_function('grid_eval_grad_kernel')
        return ('cuda', module.get_function('grid_eval_kernel'), gk)


def get_grid_evaluator(backend_name, components, phases, models,
                       phase_record_factory, verbose=False, output='GM',
                       force_property_module=False):
    """Return evaluate(phase_name, dof_2d, out_1d) for the given system.

    `output` names any Model property that is a symengine expression (GM, HM,
    SM, CPM, the _MIX/_FORM variants, ...); GM shares the full equilibrium
    module, other outputs build a lightweight property module. Modules are
    cached per (backend, output, phases, components, sizes); the first call
    for a system compiles the kernel (one-time, disk-cached thereafter).
    Raises if the system/output cannot be built (caller falls back to CPU).
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
    _param_symbols = list(getattr(phase_record_factory, 'param_symbols', []) or [])
    _mh = hashlib.md5()
    for _ph in sorted(shim.phases):
        _mh.update(_ph.encode())
        if output == 'GM' and not force_property_module:
            _mh.update(str(models[_ph].GM).encode())
        else:
            _mh.update(str(_property_expr(models[_ph], output, _param_symbols)).encode())
    cache_id = (backend_name, output, bool(force_property_module),
                tuple(sorted(shim.phases)),
                tuple(sorted(c.name for c in shim.components)),
                tuple(str(sv) for sv in shim.phase_record_factory.state_variables),
                _mh.hexdigest())
    if cache_id in _evaluator_cache:
        entry = _evaluator_cache[cache_id]
    else:
        entry = _build_module(backend_name, shim, verbose=verbose, output=output,
                              force_property_module=force_property_module)
        _evaluator_cache[cache_id] = entry

    from pycalphad.gpu.gpu_codegen import _unique_models_for_gpu
    _, name_to_idx = _unique_models_for_gpu(shim, validate=False)
    kind, fn, grad_fn = entry if len(entry) == 3 else (entry[0], entry[1], None)

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

    def evaluate(phase_name, dof, out, param_rows=None):
        """Evaluate over dof rows; `param_rows` (L, n_params) evaluates every
        (point, parameter-sample) pair — `out` must then be (n_points * L,),
        sample-minor (matching prop_parameters_2d's (n_points, L) layout)."""
        model_idx = name_to_idx[phase_name]
        n_points = dof.shape[0]
        if param_rows is not None and n_params:
            pr = np.asarray(param_rows, dtype=np.float64).reshape(-1, n_params)
            L = pr.shape[0]
            dof = np.concatenate([np.repeat(dof, L, axis=0),
                                  np.tile(pr, (n_points, 1))], axis=1)
        elif n_params:
            pv = np.asarray(phase_record_factory.param_values, dtype=np.float64).reshape(-1)[:n_params]
            dof = np.concatenate([dof, np.broadcast_to(pv, (n_points, n_params))], axis=1)
        n_points = dof.shape[0]
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

    def evaluate_grad(phase_name, dof, out):
        """Property gradient over dof rows: out is (n_points, dof_stride)
        where dof_stride includes trailing fit-parameter slots (appended
        automatically from factory.param_values, as in evaluate())."""
        if grad_fn is None:
            raise RuntimeError('gradients unavailable for this output module')
        model_idx = name_to_idx[phase_name]
        if n_params:
            pv = np.asarray(phase_record_factory.param_values, dtype=np.float64).reshape(-1)[:n_params]
            dof = np.concatenate([dof, np.broadcast_to(pv, (dof.shape[0], n_params))], axis=1)
        n_points, dof_stride = dof.shape
        dof_c = np.ascontiguousarray(dof, dtype=np.float64)
        assert out.shape == (n_points, dof_stride), (out.shape, dof.shape)
        if kind == 'cpp':
            grad_fn(int(model_idx), dof_c.ctypes.data, out.ctypes.data,
                    n_points, dof_stride, dof_stride)
        else:
            d_dof = cp.asarray(dof_c)
            d_out = cp.empty((n_points, dof_stride), dtype=cp.float64)
            tpb = 128
            blocks = (n_points + tpb - 1) // tpb
            grad_fn((blocks,), (tpb,),
                    (np.int32(model_idx), d_dof, d_out,
                     np.int64(n_points), np.int32(dof_stride), np.int32(dof_stride)))
            cp.cuda.runtime.deviceSynchronize()
            out[:] = cp.asnumpy(d_out)

    evaluate.min_points = min_points
    evaluate.grad = evaluate_grad
    return evaluate
