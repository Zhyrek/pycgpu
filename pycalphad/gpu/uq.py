"""
Batched equilibrium over parameter samples (uncertainty propagation).

``equilibrium_samples`` solves the SAME condition grid for MANY parameter
sets (e.g. an ESPEI MCMC posterior sample) and stacks the results along a
leading ``sample`` dimension. The reference ``equilibrium()`` does not
support parameter arrays (it raises), so this is an accelerated-only
capability.

Implementation note: each sample is one accelerated ``equilibrium()``
call with that sample's parameters — measured against point-solver-based
batching, the accelerated equilibrium pipeline is already the fastest
per-sample path for condition grids (~0.1 s/sample warm on a 551-condition
Cu-Mg grid, one CPU core), since kernels, models and spec fast paths are
all cached across samples.
"""
import logging

import numpy as np

from pycalphad import equilibrium
from pycalphad.core.utils import extract_parameters

_log = logging.getLogger(__name__)


def equilibrium_samples(dbf, comps, phases, conditions, parameters,
                        backend=None, verbose=False, **eq_kwargs):
    """Equilibrium over a condition grid for many parameter sets.

    Parameters
    ----------
    conditions : dict
        Any conditions the accelerated ``equilibrium()`` supports.
    parameters : dict
        {symbol name: (L,) array}; arrays must share length L.
    backend : str, optional
        'c++' or 'gpu'; defaults to the globally selected backend.

    Returns
    -------
    dict of numpy arrays with a leading ``sample`` dimension: ``GM``,
    ``MU``, ``NP``, ``Phase``, ``X``, ``Y``; plus ``sample_parameters``,
    ``parameter_names``, ``grid_dims`` and ``grid_coords``.
    """
    import pycalphad

    param_syms, param_arr = extract_parameters(parameters)
    param_names = [str(s) for s in param_syms]
    param_arr = np.atleast_2d(np.asarray(param_arr, dtype=np.float64))
    L = param_arr.shape[0]

    if backend is None:
        from pycalphad.backend import get_backend
        backend, _ = get_backend()
        if backend in (None, 'default'):
            backend = 'c++'
    backend = {'cpp': 'c++', 'cuda': 'gpu'}.get(backend, backend)

    out = None
    fields = ('GM', 'MU', 'NP', 'Phase', 'X', 'Y')
    with pycalphad.backend(backend):
        for si in range(L):
            pdict = {nm: float(val) for nm, val in zip(param_names, param_arr[si])}
            eq = equilibrium(dbf, comps, phases, conditions, parameters=pdict,
                             **eq_kwargs)
            if out is None:
                dims = [d for d in eq.GM.dims]
                out = {'grid_dims': dims,
                       'grid_coords': {d: np.asarray(eq.coords[d]) for d in dims}}
                for f in fields:
                    arr = np.asarray(eq[f].values)
                    out[f] = np.empty((L,) + arr.shape, dtype=arr.dtype)
            for f in fields:
                out[f][si] = np.asarray(eq[f].values)
            if verbose:
                gm = np.asarray(eq.GM.values, dtype=np.float64)
                _log.info("sample %d/%d: %d/%d converged", si + 1, L,
                          int(np.isfinite(gm).sum()), gm.size)
    out['sample_parameters'] = param_arr
    out['parameter_names'] = param_names
    return out
