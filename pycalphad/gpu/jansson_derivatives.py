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

__all__ = ['jansson_deltas']


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
