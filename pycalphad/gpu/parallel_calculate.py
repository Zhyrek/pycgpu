"""
Parallel grid-energy sampling: calculate() forked over the temperature axis.

calculate() cost scales with the number of temperature values (a full point
sample is evaluated per T); the per-T work is independent, so the T array is
split across forked workers running unmodified calculate() and the
LightDataset slices are concatenated. Forked children inherit the parent's
hash seed and compiled callables, so the merged result is bit-identical to a
serial call.

OPT-IN (one pycalphad call uses one core by default):
  PYCGPU_CALC_PROCS   worker count; unset/1 = serial; 0 = auto (cpu_count
                      when there are at least 2 T values per worker)
"""
import multiprocessing
import os

import numpy as np

from pycalphad.core.light_dataset import LightDataset

_WORKER_PAYLOAD = None


def _calc_worker(args):
    lo, hi = args
    func, call_args, call_kwargs, t_key = _WORKER_PAYLOAD
    kwargs = dict(call_kwargs)
    kwargs[t_key] = np.asarray(kwargs[t_key])[lo:hi]
    res = func(*call_args, **kwargs)
    return res.data_vars, res.coords, res.attrs


def parallel_calculate(calc_func, call_args, call_kwargs, t_key='T', procs=None,
                       verbose=False):
    """Run `calc_func(*call_args, **call_kwargs)` forked over the T axis.

    Falls back to a plain serial call when parallelism is off/inapplicable or
    on any worker failure. `calc_func` must return a LightDataset
    (`to_xarray=False`).
    """
    t_values = np.atleast_1d(np.asarray(call_kwargs.get(t_key, [])))
    if procs is None:
        env = os.environ.get('PYCGPU_CALC_PROCS')
        if env is None:
            procs = 1
        elif int(env) == 0:
            procs = max(1, min(os.cpu_count() or 1, t_values.size // 2))
        else:
            procs = int(env)
    procs = max(1, min(int(procs), max(1, t_values.size)))
    if procs <= 1 or t_values.size < 2:
        return calc_func(*call_args, **call_kwargs)

    bounds = np.linspace(0, t_values.size, procs + 1).astype(int)
    tasks = [(int(lo), int(hi)) for lo, hi in zip(bounds[:-1], bounds[1:]) if hi > lo]

    global _WORKER_PAYLOAD
    _WORKER_PAYLOAD = (calc_func, call_args, call_kwargs, t_key)
    try:
        ctx = multiprocessing.get_context('fork')
        with ctx.Pool(processes=len(tasks)) as pool:
            parts = pool.map(_calc_worker, tasks)
    except Exception as e:
        if verbose:
            print(f"[GPU] parallel calculate failed ({e!r}); falling back to serial")
        return calc_func(*call_args, **call_kwargs)
    finally:
        _WORKER_PAYLOAD = None

    first_vars, first_coords, first_attrs = parts[0]
    merged_vars = {}
    for name, (dims, _) in first_vars.items():
        if t_key in dims:
            axis = list(dims).index(t_key)
            merged = np.concatenate([p[0][name][1] for p in parts], axis=axis)
        else:
            merged = first_vars[name][1]
        merged_vars[name] = (dims, merged)
    merged_coords = dict(first_coords)
    merged_coords[t_key] = t_values
    if verbose:
        print(f"[GPU] calculate parallelized: {len(tasks)} workers over "
              f"{t_values.size} T values")
    return LightDataset(merged_vars, coords=merged_coords, attrs=first_attrs)
