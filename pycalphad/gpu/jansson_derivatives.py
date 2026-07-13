"""Batched Jansson-derivative deltas on the accelerated backends.

``jansson_deltas`` solves the whole condition grid once on the selected
backend and, at every converged state, additionally solves the perturbed
equilibrium system of Sundman et al. (2015) Eq. 74/78 for the derivative
deltas with respect to one state-variable condition — the same quantities
the reference computes per condition in
``pycalphad.core.minimizer.state_variable_differential`` /
``site_fraction_differential``, but for all conditions in one launch.

Phase 1 scope: state-variable denominators (v.T, v.P).  Composition and
fit-parameter denominators are phase 2; the property-side chain rule
(``JanssonDerivative`` numerators) composes these deltas with generated
property gradients on the host.
"""
import os

import numpy as np

from pycalphad import Workspace

__all__ = ['jansson_deltas', 'jansson_derivative']


def jansson_deltas(dbf, comps, phases, conditions, denominator, backend=None,
                   **wks_kwargs):
    """Per-condition Jansson deltas over a condition grid.

    Parameters
    ----------
    denominator : pycalphad StateVariable condition to differentiate against
        (must be one of the fixed state variables, e.g. ``v.T``).

    Returns
    -------
    dict with arrays over the flattened condition grid:
      ``delta_MU``            (n_conds, n_components)
      ``delta_statevars``     (n_conds, n_statevars)  [d(target)/d(target)=1]
      ``delta_phase_amounts`` (n_conds, max_phases)   formula-unit convention
      ``delta_sitefracs``     (n_conds, max_phases, max_dof)
      ``ok``                  (n_conds,) bool — converged and solved
      plus ``grid_dims``/``grid_coords`` describing the condition grid and
      ``statevar_index`` of the denominator.
    """
    import pycalphad
    from pycalphad.gpu.gpu_equilibrium import calculate_equilibrium_gpu

    if backend is None:
        from pycalphad.backend import get_backend
        backend, _ = get_backend()
        if backend in (None, 'default'):
            backend = 'c++'
    backend = {'cpp': 'c++', 'cuda': 'gpu'}.get(backend, backend)

    wks = Workspace(dbf, comps, phases, conditions, **wks_kwargs)
    state_variables = sorted(wks.phase_record_factory.state_variables, key=str)
    try:
        sv_idx = state_variables.index(denominator)
    except ValueError:
        raise ValueError(
            f'{denominator} is not a state variable of this system '
            f'({state_variables}); phase-1 jansson_deltas supports state-'
            f'variable denominators only')

    # The c++ driver is selected via the PYCGPU_CPU env (the same switch
    # run_accelerated_workspace uses); force_cpu means "no accelerated path".
    _env_saves = {}
    for key, val in (('PYCGPU_JANSSON_TARGET', str(sv_idx)),
                     ('PYCGPU_CPU', '1' if backend == 'c++' else None)):
        _env_saves[key] = os.environ.get(key)
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val
    try:
        with pycalphad.backend(backend):
            eq = calculate_equilibrium_gpu(wks)
    finally:
        for key, old_val in _env_saves.items():
            if old_val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_val

    stash = getattr(wks, '_jansson_deltas', None)
    if stash is None:
        raise RuntimeError('accelerated pipeline did not produce Jansson '
                           'deltas (dispatch fell back?)')
    raw = stash['raw']
    L = stash['layout']
    nc, nsv = L['MAX_COMPONENTS'], L['MAX_STATEVARS']
    nph, ndof = L['MAX_PHASES'], L['MAX_DOF_PER_PHASE']
    o = 0
    delta_mu = raw[:, o:o + nc]; o += nc
    delta_sv = raw[:, o:o + nsv]; o += nsv
    delta_amt = raw[:, o:o + nph]; o += nph
    delta_y = raw[:, o:o + nph * ndof].reshape(-1, nph, ndof); o += nph * ndof
    ok = raw[:, o] > 0.5

    n_active = len(wks.components) - (1 if any(str(c) == 'VA' for c in wks.components) else 0)
    dims = list(eq.GM.dims)
    return {
        'delta_MU': delta_mu[:, :n_active],
        'delta_statevars': delta_sv[:, :len(state_variables)],
        'delta_phase_amounts': delta_amt,
        'delta_sitefracs': delta_y,
        'ok': ok,
        'statevar_index': sv_idx,
        'grid_dims': dims,
        'grid_coords': {d: np.asarray(eq.coords[d]) for d in dims},
        'eq': eq,
    }


def jansson_derivative(dbf, comps, phases, conditions, numerator, denominator,
                       backend=None, **wks_kwargs):
    """Finished Jansson derivatives d(numerator)/d(denominator) per condition.

    ``numerator`` is a Model property name ('GM', 'HM', 'SM', ...);
    ``denominator`` is a fixed state-variable condition (v.T, v.P).  Returns
    a dict with ``values`` shaped like the flattened condition grid, plus the
    grid layout and the underlying deltas.

    The chain rule follows the reference implementation literally
    (property_framework.computed_property.jansson_derivative, Sundman 2015
    Eq. 73): compsets with zero amount are skipped, system properties weight
    the gradient terms by the compset's NP (moles of atoms) and add the
    delta-phase-amount term.
    """
    import numpy as np
    from pycalphad.gpu.gpu_calculate import get_grid_evaluator
    from pycalphad import Model

    res = jansson_deltas(dbf, comps, phases, conditions, denominator,
                         backend=backend, **wks_kwargs)
    eq = res['eq']
    if backend is None:
        from pycalphad.backend import get_backend
        backend, _ = get_backend()
        if backend in (None, 'default'):
            backend = 'c++'
    backend_name = {'c++': 'cpp', 'gpu': 'gpu'}.get(backend, backend)

    wks = Workspace(dbf, comps, phases, conditions, **wks_kwargs)
    models = wks.models
    prf = wks.phase_record_factory
    state_variables = sorted(prf.state_variables, key=str)
    nsv = len(state_variables)
    evaluator = get_grid_evaluator(backend_name, wks.components, list(wks.phases),
                                   models, prf, output=numerator,
                                   force_property_module=True)

    # Flatten the converged grid: (conditions, vertex) phase names, NP
    # (moles of atoms), Y site fractions, and per-condition statevar values.
    n_vert = eq.Phase.shape[-1]
    names = np.asarray(eq.Phase.values).reshape(-1, n_vert)
    np_amt = np.asarray(eq.NP.values, dtype=np.float64).reshape(-1, n_vert)
    y = np.asarray(eq.Y.values, dtype=np.float64).reshape(-1, n_vert, eq.Y.shape[-1])
    n_conds = names.shape[0]

    # Per-condition state variable values in sorted(statevar) order.
    dims = res['grid_dims']
    coords = res['grid_coords']
    mesh = np.meshgrid(*[coords[d] for d in dims], indexing='ij')
    sv_cols = {}
    for i, svar in enumerate(state_variables):
        name = str(svar)
        if name in coords:
            sv_cols[i] = np.asarray(mesh[dims.index(name)], dtype=np.float64).reshape(-1)
        else:
            sv_cols[i] = np.full(n_conds, float(np.asarray(wks.conditions[svar]).reshape(-1)[0]))

    d_mu = res['delta_MU']; d_sv_all = res['delta_statevars']
    d_amt = res['delta_phase_amounts']; d_y_all = res['delta_sitefracs']
    ok = res['ok']

    values = np.full(n_conds, np.nan)
    # Group evaluations by phase for batching.
    for phase_name in sorted(set(names.reshape(-1)) - {''}):
        pr = prf[str(phase_name)]
        pdof = pr.phase_dof
        rows = []
        locs = []   # (condition index, vertex index)
        for ci in range(n_conds):
            if not ok[ci]:
                continue
            for vi in range(n_vert):
                if names[ci, vi] != phase_name:
                    continue
                amt = np_amt[ci, vi]
                if not np.isfinite(amt) or amt == 0.0:
                    continue
                dof_row = np.empty(nsv + pdof)
                for i in range(nsv):
                    dof_row[i] = sv_cols[i][ci]
                dof_row[nsv:] = y[ci, vi, :pdof]
                rows.append(dof_row)
                locs.append((ci, vi))
        if not rows:
            continue
        dof2d = np.asarray(rows)
        func_vals = np.zeros(len(rows))
        evaluator(str(phase_name), dof2d, func_vals)
        # The generated property gradient is ordered [T, site fractions...]
        # (get_ordered_symbols_for_diff), NOT the [N, P, T, y...] dof layout:
        # no N/P columns are emitted.  d(anything)/dN and /dP contributions
        # are zero for T/P denominators anyway (dN=0 always; dP=0 for a T
        # denominator; a P denominator will need a codegen ordering
        # extension before dX/dP works — currently T only).
        t_pos = state_variables.index([sv for sv in state_variables
                                       if str(sv) == 'T'][0])
        grads = np.zeros((len(rows), dof2d.shape[1] + (
            len(getattr(prf, 'param_symbols', []) or []))))
        evaluator.grad(str(phase_name), dof2d, grads)
        for r, (ci, vi) in enumerate(locs):
            if np.isnan(func_vals[r]):
                continue
            if np.isnan(values[ci]):
                values[ci] = 0.0
            contrib = d_amt[ci, vi] * func_vals[r]
            contrib += np_amt[ci, vi] * d_sv_all[ci][t_pos] * grads[r, 0]
            contrib += np_amt[ci, vi] * float(
                np.dot(d_y_all[ci, vi, :pdof], grads[r, 1:1 + pdof]))
            values[ci] += contrib
    return {'values': values, 'grid_dims': dims, 'grid_coords': coords,
            'deltas': res}
