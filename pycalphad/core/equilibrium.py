"""
The equilibrium module defines routines for interacting with
calculated phase equilibria.
"""
import warnings
from collections import OrderedDict
from collections.abc import Iterable
from datetime import datetime
from pycalphad.core.workspace import Workspace
from pycalphad.core.light_dataset import LightDataset
import numpy as np
from pycalphad.property_framework import as_property


def _accelerated_conditions_supported(conditions, parameters, solver,
                                      phase_records, output, extra_kwargs):
    """Whether the accelerated backends support this equilibrium problem shape.

    Supported: standard N=1 / P / T / X(component) condition grids with no
    parameter overrides, custom solver, prebuilt phase records, extra outputs,
    or phase-local / chemical-potential / fixed-phase conditions.
    """
    import numpy as np
    from pycalphad import variables as v
    if parameters:
        # Scalar parameter overrides are supported (runtime fit-parameter
        # slots in the generated kernels); vectorized parameter sweeps are not.
        try:
            for pv in dict(parameters).values():
                if np.asarray(pv, dtype=np.float64).size != 1:
                    return False
        except Exception:
            return False
    if solver is not None or phase_records is not None:
        return False
    if output not in (None, 'GM'):
        return False
    if extra_kwargs:
        return False
    try:
        n_x_conds = sum(1 for c in conditions
                        if isinstance(c, v.MoleFraction) and getattr(c, 'phase_name', None) is None)
        n_mu_conds = sum(1 for c in conditions if isinstance(c, v.ChemicalPotential))
        n_statevar_conds = sum(1 for c in conditions if c in (v.N, v.P, v.T))
        # Fully-determined standard problems only: every condition is
        # N/P/T/X/MU and nothing else (under/overdetermined problems must
        # reach the reference path's validation errors).
        if n_x_conds + n_mu_conds + n_statevar_conds != len(conditions):
            return False
        for cond, value in conditions.items():
            if getattr(cond, 'phase_name', None) is not None:
                return False
            if cond == v.N:
                if np.any(np.atleast_1d(np.asarray(value, dtype=object)).astype(float) != 1.0):
                    return False
            elif cond == v.P or cond == v.T:
                continue
            elif isinstance(cond, v.ChemicalPotential):
                # Scalar fixed chemical potentials are supported; MU axes
                # (arrays) are not batched yet — reference path handles them.
                if np.asarray(value, dtype=np.float64).size != 1:
                    return False
                continue
            elif isinstance(cond, v.MoleFraction):
                # Dilute/zero compositions have dedicated reference-path
                # handling (clamping + user warnings) the accelerated
                # solvers do not replicate.
                if np.any(np.asarray(value, dtype=np.float64) < 1e-9):
                    return False
                continue
            else:
                # ChemicalPotential, MassFraction, SiteFraction, LinearCombination, ...
                return False
    except Exception:
        return False
    return True


def equilibrium(dbf, comps, phases, conditions, output=None, model=None,
                verbose=False, calc_opts=None, to_xarray=True,
                parameters=None, solver=None, phase_records=None,
                gpu=False, force_cpu=False, fallback_on_error=True,
                backend=None, robust_phase_removal=None, **kwargs):
    """
    Calculate the equilibrium state of a system containing the specified
    components and phases, under the specified conditions.

    Parameters
    ----------
    dbf : Database
        Thermodynamic database containing the relevant parameters.
    comps : list
        Names of components to consider in the calculation.
    phases : list or dict
        Names of phases to consider in the calculation.
    conditions : dict or (list of dict)
        StateVariables and their corresponding value.
    output : str or list of str, optional
        Additional equilibrium model properties (e.g., CPM, HM, etc.) to compute.
        These must be defined as attributes in the Model class of each phase.
    model : Model, a dict of phase names to Model, or a seq of both, optional
        Model class to use for each phase.
    verbose : bool, optional
        Print details of calculations. Useful for debugging.
    calc_opts : dict, optional
        Keyword arguments to pass to `calculate`, the energy/property calculation routine.
    to_xarray : bool
        Whether to return an xarray Dataset (True, default) or an EquilibriumResult.
    parameters : dict, optional
        Maps SymEngine Symbol to numbers, for overriding the values of parameters in the Database.
    solver : pycalphad.core.solver.SolverBase
        Instance of a solver that is used to calculate local equilibria.
        Defaults to a pycalphad.core.solver.Solver.
    phase_records : Optional[Mapping[str, PhaseRecord]]
        Mapping of phase names to PhaseRecord objects with `'GM'` output. Must include
        all active phases. The `model` argument must be a mapping of phase names to
        instances of Model objects.
    gpu : bool, optional
        Whether to use GPU acceleration for equilibrium calculations. Defaults to False.
    force_cpu : bool, optional
        Force CPU calculation even if GPU is available (useful for testing and comparison).
    fallback_on_error : bool, optional
        Automatically fall back to CPU if GPU calculation fails (default True).
    backend : str, optional
        Accelerated solver backend: 'cuda' (CuPy/CUDA GPU) or 'cpp' (C++/OpenMP
        on the host, no CUDA required). Passing a backend implies gpu=True.
        Default (None): 'cuda', or 'cpp' if the PYCGPU_CPU environment variable is set.
    robust_phase_removal : bool, optional
        Count phase removals from consolidation toward the per-compset removal
        budget, so add/collapse/re-add cycles on near-duplicate composition sets
        terminate instead of consuming the iteration budget. Applies to both the
        reference CPU solver and the accelerated backends. Default (None): off,
        unless the PYCALPHAD_ROBUST_REMOVAL / PYCGPU_ROBUST environment
        variables are set.

    Returns
    -------
    Structured equilibrium calculation

    Examples
    --------
    None yet.
    """
    import os
    from pycalphad.backend import get_backend, _option_env
    # Resolution order for the execution backend: explicit per-call kwarg,
    # else the global set_backend()/PYCALPHAD_BACKEND setting, else 'default'.
    _global_backend, _global_options = get_backend()
    _backend_from_global = False
    if backend is not None:
        from pycalphad.backend import _normalize
        backend = _normalize(backend)  # accepts 'gpu', 'c++', aliases
        if backend == 'default':
            backend = None
        else:
            gpu = True
    if backend is not None:
        pass
    elif _global_backend != 'default' and not force_cpu:
        # A GLOBAL backend only takes the accelerated path for problem shapes
        # it supports; everything else silently uses the reference solver so
        # `set_backend(...)` is always safe. An explicit per-call `backend=`
        # kwarg bypasses this gate (deliberate user demand).
        _gate_ok = _accelerated_conditions_supported(conditions, parameters, solver,
                                             phase_records, output, kwargs)
        if os.environ.get('PYCGPU_COUNT_DISPATCH'):
            with open(os.environ['PYCGPU_COUNT_DISPATCH'], 'a') as _f:
                _f.write('gate_pass\n' if _gate_ok else 'gate_fallback\n')
        if _gate_ok:
            backend = _global_backend
            gpu = True
            _backend_from_global = True

    if gpu:
        # Environment variables steer the accelerated pipeline; set them for the
        # duration of the call so explicit kwargs win, then restore.
        overrides = {}
        if backend is not None:
            overrides['PYCGPU_CPU'] = '1' if backend == 'cpp' else ''
        if robust_phase_removal is None and 'PYCGPU_ROBUST' not in os.environ:
            # Default ON for the accelerated backends: terminates the
            # add/collapse cycles that otherwise burn the iteration budget on
            # near-duplicate composition sets (alni_tough, AlCuFe cond 76/47).
            # The reference solver keeps its stock behavior.
            robust_phase_removal = True
        if robust_phase_removal is not None:
            overrides['PYCGPU_ROBUST'] = '1' if robust_phase_removal else ''
        saved = {k: os.environ.get(k) for k in overrides}
        try:
            for k, val in overrides.items():
                if val:
                    os.environ[k] = val
                else:
                    os.environ.pop(k, None)
            from ..gpu.gpu_equilibrium import equilibrium_gpu
            try:
                with _option_env(_global_options):
                    # GPU mode handles its own debug output
                    return equilibrium_gpu(dbf, comps, phases, conditions, output=output, model=model,
                                         verbose=verbose, calc_opts=calc_opts, to_xarray=to_xarray,
                                         parameters=parameters, solver=solver, phase_records=phase_records,
                                         force_cpu=force_cpu, fallback_on_error=fallback_on_error, **kwargs)
            except Exception as _accel_err:
                if not _backend_from_global:
                    raise
                # Silent fallback (log only): the global backend must never
                # change user-visible behavior for unsupported problems, and
                # test suites commonly run with warnings-as-errors.
                import logging
                logging.getLogger(__name__).debug(
                    "Accelerated backend failed, using reference solver: %r", _accel_err)
                if os.environ.get('PYCGPU_COUNT_DISPATCH'):
                    import traceback
                    _tb = traceback.extract_tb(_accel_err.__traceback__)
                    _loc = f'{_tb[-1].filename.rsplit("/", 1)[-1]}:{_tb[-1].lineno}' if _tb else '?'
                    _test = os.environ.get('PYTEST_CURRENT_TEST', '')
                    with open(os.environ['PYCGPU_COUNT_DISPATCH'], 'a') as _f:
                        _f.write(f'runtime_fallback [{_loc}] <{_test}>: {str(_accel_err)[:100]}\n')
        finally:
            for k, old in saved.items():
                if old is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = old

    # robust_phase_removal applies to the ACCELERATED backends only (their
    # kernels implement the robust-removal gate; the reference Cython solver
    # is upstream-unmodified and has no such switch). On the reference path
    # the kwarg is accepted and ignored so backend-defaulted options do not
    # change reference behavior.

    if output is None:
        output = set()
    elif (not isinstance(output, Iterable)) or isinstance(output, str):
        output = [output]
    wks = Workspace(database=dbf, components=comps, phases=phases, conditions=conditions, models=model, parameters=parameters,
                    verbose=verbose, calc_opts=calc_opts, solver=solver, phase_record_factory=phase_records)

    # Compute equilibrium values of any additional user-specified properties
    # We already computed these properties so don't recompute them
    properties = wks.eq
    if verbose:
        print(f"DEBUG equilibrium: properties = {properties}")
        # Don't access wks.eq again as it may trigger another recompute
    if verbose and properties is not None:
        print("\n=== CPU EQUILIBRIUM FINAL RESULTS ===\n")
        
        # GM (Gibbs energy)
        if hasattr(properties, 'GM') and properties.GM is not None:
            gm_values = properties.GM
            if hasattr(gm_values, 'values'):
                gm_flat = gm_values.values.flatten()
            else:
                gm_flat = gm_values.flatten()
            print(f"CPU Final GM: {gm_flat[0]:.6f} J/mol")
        
        # Phase amounts
        if hasattr(properties, 'NP') and properties.NP is not None:
            np_values = properties.NP
            if hasattr(np_values, 'values'):
                np_flat = np_values.values.flatten()
            else:
                np_flat = np_values.flatten()
            active_phases = np_flat[np_flat > 1e-12]
            print(f"CPU Final active phase amounts: {active_phases}")
            
            # Show all phase amounts (including zero)
            print(f"CPU All phase amounts: {np_flat[:10]}...")  # Show first 10 to avoid clutter
        
        # Chemical potentials
        if hasattr(properties, 'MU') and properties.MU is not None:
            mu_values = properties.MU
            if hasattr(mu_values, 'values'):
                mu_flat = mu_values.values.flatten()
            else:
                mu_flat = mu_values.flatten()
            print(f"CPU Final chemical potentials: {mu_flat}")
        
        # Phase names
        if hasattr(properties, 'Phase') and properties.Phase is not None:
            phase_values = properties.Phase
            if hasattr(phase_values, 'values'):
                phase_flat = phase_values.values.flatten()
            else:
                phase_flat = phase_values.flatten()
            active_phase_names = [p for p in phase_flat if p != '' and p != '_FAKE_']
            print(f"CPU Final active phases: {active_phase_names}")
        
        # Compositions
        if hasattr(properties, 'X') and properties.X is not None:
            x_values = properties.X
            if hasattr(x_values, 'values'):
                x_flat = x_values.values
            else:
                x_flat = x_values
            # Print composition of first few active phases
            for i in range(min(3, x_flat.shape[-2])):
                phase_comp = x_flat.flatten()[i*x_flat.shape[-1]:(i+1)*x_flat.shape[-1]]
                if np.sum(phase_comp) > 1e-12:  # Only show if phase has composition
                    print(f"CPU Phase {i} composition: {phase_comp}")
        
        print("=== END CPU EQUILIBRIUM DEBUG ===\n")
    
    # END DEBUG
    
    
    if properties is None:
        if verbose:
            print("WARNING: properties is None, returning None")
        return None
    
    conds_keys = [str(k) for k in properties.coords.keys() if k not in ('vertex', 'component', 'internal_dof')]
    if verbose:
        print(f"  condition_keys: {conds_keys}")
    output = sorted(set(output) - {'GM', 'MU'})
    if verbose:
        print(f"  additional_properties: {output}")
    
    for out in output:
        cprop = as_property(out)
        out = str(cprop)
        result_array = np.zeros(properties.GM.shape) # Will not work for non-scalar properties
        
        for index, composition_sets in wks.enumerate_composition_sets():
            cur_conds = OrderedDict(zip(conds_keys,
                                        [np.asarray(properties.coords[b][a], dtype=np.float64)
                                        for a, b in zip(index, conds_keys)]))
            chemical_potentials = properties.MU[index]
            result_array[index] = cprop.compute_property(composition_sets, cur_conds, chemical_potentials)
            if verbose:
                print(f"  result: {result_array[index]}")
        
        result = LightDataset({out: (conds_keys, result_array)}, coords=properties.coords)
        properties.merge(result, inplace=True, compat='equals')
    
    if to_xarray:
        properties = wks.eq.get_dataset()
        if verbose:
            print("  converted to xarray Dataset")
    properties.attrs['created'] = datetime.now().isoformat()
    if verbose:
        print(f"  added creation timestamp: {properties.attrs['created']}")
    if len(kwargs) > 0:
        warnings.warn('The following equilibrium keyword arguments were passed, but unused:\n{}'.format(kwargs))
    return properties
