"""
Global-minimum candidate search for ZPF mapping on the compiled backends.

pycalphad.mapping.zpf_equilibrium._find_global_min_cs samples every phase
with calculate(), computes the driving force X.mu - GM of every sample and
picks the samples with the largest driving force. On a compiled backend
this module produces the same GM, X and site-fraction arrays without
calculate()'s dataset assembly (most of its cost at mapping's call rate):
the same cached sample points, energies from the same compiled evaluator
calculate() uses, and compositions from the same phase-record calls. The
driving forces are then computed exactly as before, so they are
bit-identical, and the top candidates are selected with a fast path that
returns exactly what np.argsort(dGs)[::-1][:k] would.
"""
import weakref
from collections import OrderedDict

import numpy as np

__all__ = ['sampled_energies', 'top_candidates']

# Mapping repeats the global-minimum check at the same state variables (an
# isothermal section: every check), where samples, energies and compositions
# are identical and only the chemical potentials change. A few recent results
# are kept, keyed by the phase records (held weakly), phases, state variables
# and sampling density.
_CACHE_SIZE = 4
_cache = OrderedDict()


def sampled_energies(dbf, comps, phases, models, phase_records, state_conds, pdens):
    """(gm, x, y, phase_ids) as _find_global_min_cs derives them from
    calculate(dbf, comps, phases, model=models, phase_records=phase_records,
    output='GM', pdens=pdens, **state_conds), or None when the reference
    path should be used (no compiled backend selected, or an unsupported
    case)."""
    from pycalphad.backend import _current
    if _current()[0] == 'default':
        return None
    from pycalphad import variables as v
    from pycalphad.core.calculate import _sample_phase_constitution
    from pycalphad.core.utils import filter_phases, get_pure_elements, point_sample, unpack_species
    from pycalphad.gpu.gpu_calculate import accelerated_evaluator

    comps = sorted(unpack_species(dbf, comps))
    active_phases = sorted(filter_phases(dbf, comps, phases))
    if not active_phases:
        return None
    statevars = {str(k): float(np.asarray(val).reshape(-1)[0]) for k, val in state_conds.items()}
    statevars.setdefault('N', 1.0)
    if statevars['N'] != 1.0:
        return None
    required = [str(sv) for sv in phase_records.state_variables]
    if set(required) != set(statevars):
        return None
    key = (id(phase_records), tuple(active_phases), tuple(sorted(statevars.items())), pdens)
    entry = _cache.get(key)
    if entry is not None and entry[0]() is phase_records and entry[1] is models:
        _cache.move_to_end(key)
        return entry[2]
    evaluator = accelerated_evaluator(comps, active_phases, models, phase_records, 'GM', {})
    if evaluator is None:
        return None
    pure_elements = get_pure_elements(dbf, comps)
    sv_row = np.array([statevars[k] for k in sorted(statevars)], dtype=np.float64)
    maximum_internal_dof = max(len(models[ph].site_fractions) for ph in active_phases)

    gms, xs, ys, names = [], [], [], []
    for phase_name in active_phases:
        phase_record = phase_records[phase_name]
        points = np.atleast_2d(_sample_phase_constitution(models[phase_name], point_sample, True, pdens, {}))
        if points.size == 0:
            continue
        dof = np.ascontiguousarray(np.concatenate((np.broadcast_to(sv_row, (points.shape[0], sv_row.size)), points), axis=1))
        gm = np.zeros(dof.shape[0])
        evaluator(phase_name, dof, gm)
        x = np.zeros((dof.shape[0], len(pure_elements)), order='F')
        for el_idx in range(len(pure_elements)):
            phase_record.mass_obj_2d(x[:, el_idx], dof, el_idx)
        y = np.full((points.shape[0], maximum_internal_dof), np.nan)
        y[:, :points.shape[1]] = points
        gms.append(gm)
        xs.append(np.asarray(x, dtype=np.float64))
        ys.append(y)
        names.append(np.full(points.shape[0], phase_name, dtype='U' + str(len(phase_name))))
    if not gms:
        return None
    # x is C-contiguous in the reference too (np.dot's summation can depend on layout)
    result = (np.concatenate(gms), np.ascontiguousarray(np.concatenate(xs)), np.concatenate(ys),
              np.concatenate(names))
    for array in result:
        array.setflags(write=False)
    _cache[key] = (weakref.ref(phase_records), models, result)
    while len(_cache) > _CACHE_SIZE:
        _cache.popitem(last=False)
    return result


def top_candidates(dGs, k, phase_ids=None, y=None):
    """Indices equivalent to np.argsort(dGs)[::-1][:k] for _find_global_min_cs,
    without sorting everything:

    - when the top k values are distinct and strictly above the rest, they
      are exactly argsort's;
    - for k == 1 with a tie at the maximum, argsort's choice among the tied
      samples is not reproducible without sorting, but when they are all the
      same phase with identical site fractions every choice yields the same
      candidate composition set and driving force.

    Otherwise (e.g. NaNs) the full argsort decides."""
    n = dGs.shape[0]
    k = min(k, n)
    if k == 1 and n > 0 and phase_ids is not None:
        largest = dGs.max()  # NaN if any value is NaN
        tied = np.flatnonzero(dGs == largest) if not np.isnan(largest) else np.empty(0, dtype=np.intp)
        if tied.size == 0:
            return np.argsort(dGs)[::-1][:k]
        if tied.size == 1:
            return tied
        if np.all(phase_ids[tied] == phase_ids[tied[0]]) and \
                np.array_equal(y[tied], np.broadcast_to(y[tied[0]], y[tied].shape), equal_nan=True):
            return tied[:1]
    if 0 < k < n and not np.isnan(dGs).any():
        top = np.argpartition(dGs, n - k)[n - k:]
        values = dGs[top]
        rest_max = np.delete(dGs, top).max()
        if np.unique(values).size == k and values.min() > rest_max:
            return top[np.argsort(values)[::-1]]
    return np.argsort(dGs)[::-1][:k]
