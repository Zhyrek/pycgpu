"""
Parallel starting-point computation for the accelerated backends.

lower_convex_hull cost is linear in the number of conditions and entirely
independent per condition, so the condition grid is split along a composition
axis and starting_point() (unmodified CPU pycalphad code) runs in forked
worker processes on each slice. Forked children inherit the parent's hash
seed and compiled callables, so the merged result is bit-identical to a
serial call in the same process.

Controls (env):
  PYCGPU_HULL_PROCS   number of workers; unset/1 = serial (the default: one
                      pycalphad call uses one core); 0 = auto (cpu_count
                      above PYCGPU_HULL_MIN conditions)
  PYCGPU_HULL_MIN     auto-mode threshold in conditions (default 512)
"""
import multiprocessing
import os

import numpy as np

from pycalphad import variables as v
from pycalphad.core.light_dataset import LightDataset
from pycalphad.core.starting_point import starting_point

# Payload handed to forked workers by inheritance (never pickled): the
# phase_record_factory holds compiled callables that may not pickle.
_WORKER_PAYLOAD = None


def _hull_worker(args):
    split_key_str, lo, hi = args
    conditions, state_variables, phase_records, grid = _WORKER_PAYLOAD
    sub_conds = type(conditions)()
    for key, value in conditions.items():
        if str(key) == split_key_str:
            sub_conds[key] = np.asarray(value)[lo:hi]
        else:
            sub_conds[key] = value
    res = starting_point(sub_conds, state_variables, phase_records, grid)
    # LightDataset contents are plain numpy => picklable back to the parent
    return res.data_vars, res.coords, res.attrs


def _pick_split_axis(conditions):
    """Choose a composition condition to split along (grid is T/P-dependent
    but composition-independent, so X-splits can share the full grid)."""
    best = None
    for key, value in conditions.items():
        if isinstance(key, v.MoleFraction) and getattr(key, 'phase_name', None) is None:
            n = np.asarray(value).size
            if n > 1 and (best is None or n > best[1]):
                best = (key, n)
    return best


def parallel_starting_point(conditions, state_variables, phase_records, grid,
                            verbose=False):
    """starting_point() over the condition grid, parallelized across processes.

    Falls back to the plain serial call when parallelism is disabled, not
    applicable (no multi-valued composition axis), or on any worker failure.
    """
    num_conditions = 1
    for value in conditions.values():
        num_conditions *= np.asarray(value).size

    # OPT-IN parallelism: like the C++ backend, pycalphad defaults to one
    # core per call so users can parallelize by running their own pycalphad
    # calls concurrently. Set PYCGPU_HULL_PROCS (or the hull_procs backend
    # option) to a worker count, or to 0 for auto (cpu_count above the
    # PYCGPU_HULL_MIN condition threshold, default 512).
    procs_env = os.environ.get('PYCGPU_HULL_PROCS')
    if procs_env is None:
        procs = 1
    elif int(procs_env) == 0:
        min_conds = int(os.environ.get('PYCGPU_HULL_MIN', 512))
        procs = (os.cpu_count() or 1) if num_conditions >= min_conds else 1
    else:
        procs = int(procs_env)

    split = _pick_split_axis(conditions)
    if procs <= 1 or split is None:
        return starting_point(conditions, state_variables, phase_records, grid)

    split_key, n_split = split
    procs = min(procs, n_split)
    bounds = np.linspace(0, n_split, procs + 1).astype(int)
    tasks = [(str(split_key), int(lo), int(hi))
             for lo, hi in zip(bounds[:-1], bounds[1:]) if hi > lo]

    global _WORKER_PAYLOAD
    _WORKER_PAYLOAD = (conditions, state_variables, phase_records, grid)
    try:
        ctx = multiprocessing.get_context('fork')
        with ctx.Pool(processes=len(tasks)) as pool:
            parts = pool.map(_hull_worker, tasks)
    except Exception as e:
        if verbose:
            print(f"[GPU] parallel hull failed ({e!r}); falling back to serial")
        return starting_point(conditions, state_variables, phase_records, grid)
    finally:
        _WORKER_PAYLOAD = None

    # Merge: concatenate every data var along the split axis
    first_vars, first_coords, first_attrs = parts[0]
    merged_vars = {}
    for name, (dims, _) in first_vars.items():
        axis = list(dims).index(str(split_key))
        merged = np.concatenate([p[0][name][1] for p in parts], axis=axis)
        merged_vars[name] = (dims, merged)
    merged_coords = dict(first_coords)
    merged_coords[str(split_key)] = np.asarray(conditions[split_key])
    if verbose:
        print(f"[GPU] starting_point parallelized: {len(tasks)} workers over "
              f"{str(split_key)} ({n_split} values, {num_conditions} conditions)")
    return LightDataset(merged_vars, coords=merged_coords, attrs=first_attrs)
