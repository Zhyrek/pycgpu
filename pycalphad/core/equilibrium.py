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
from pycalphad.core.debug_output import init_debug_output, close_debug_output, debug_log


def equilibrium(dbf, comps, phases, conditions, output=None, model=None,
                verbose=False, calc_opts=None, to_xarray=True,
                parameters=None, solver=None, phase_records=None, 
                gpu=False, force_cpu=False, fallback_on_error=True, **kwargs):
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

    Returns
    -------
    Structured equilibrium calculation

    Examples
    --------
    None yet.
    """
    if gpu:
        from ..gpu.gpu_equilibrium import equilibrium_gpu
        # GPU mode handles its own debug output
        return equilibrium_gpu(dbf, comps, phases, conditions, output=output, model=model,
                             verbose=verbose, calc_opts=calc_opts, to_xarray=to_xarray,
                             parameters=parameters, solver=solver, phase_records=phase_records,
                             force_cpu=force_cpu, fallback_on_error=fallback_on_error, **kwargs)
    
    # Initialize debug output for CPU mode only
    init_debug_output(enabled=verbose, mode="CPU")
    
    # SEGMENT 1: ENTRY POINT AND PARAMETER VALIDATION
    debug_log(1, "Entry point and parameter validation", {
        "gpu_mode": gpu,
        "components": comps,
        "phases": phases,
        "conditions": conditions,
        "output_requested": output,
        "verbose": verbose,
        "force_cpu": False  # Add to match GPU output
    })
    
    if output is None:
        output = set()
    elif (not isinstance(output, Iterable)) or isinstance(output, str):
        output = [output]
    # DEBUG: Log equilibrium function start
    if verbose:
        print(f"\n=== CPU EQUILIBRIUM FUNCTION START ===\nComponents: {comps}\nPhases: {phases}\nConditions: {conditions}\n")
    
    # SEGMENT 2: WORKSPACE INITIALIZATION
    debug_log(2, "Workspace initialization", {
        "database": str(dbf),
        "models": str(model),
        "parameters": parameters,
        "calc_opts": calc_opts,
        "solver": str(solver),
        "phase_records": str(phase_records)
    })
    
    wks = Workspace(database=dbf, components=comps, phases=phases, conditions=conditions, models=model, parameters=parameters,
                    verbose=verbose, calc_opts=calc_opts, solver=solver, phase_record_factory=phase_records)

    # Compute equilibrium values of any additional user-specified properties
    # We already computed these properties so don't recompute them
    properties = wks.eq
    
    # DEBUG: Check properties value
    if verbose:
        print(f"DEBUG equilibrium: properties = {properties}")
        # Don't access wks.eq again as it may trigger another recompute
    
    # DEBUG: Add CPU-GPU comparison logging
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
    
    # SEGMENT 28: Post-equilibrium property calculation
    debug_log(28, "Post-equilibrium property calculation")
    
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
        # SEGMENT 29-30: Composition set enumeration and property computation
        debug_log(29, f"Property '{out}' calculation")
        cprop = as_property(out)
        out = str(cprop)
        result_array = np.zeros(properties.GM.shape) # Will not work for non-scalar properties
        
        for index, composition_sets in wks.enumerate_composition_sets():
            debug_log(30, f"Computing property at index {index}")
            cur_conds = OrderedDict(zip(conds_keys,
                                        [np.asarray(properties.coords[b][a], dtype=np.float64)
                                        for a, b in zip(index, conds_keys)]))
            chemical_potentials = properties.MU[index]
            result_array[index] = cprop.compute_property(composition_sets, cur_conds, chemical_potentials)
            if verbose:
                print(f"  result: {result_array[index]}")
        
        # SEGMENT 31: Property merge
        debug_log(31, f"Merging property '{out}' into dataset")
        result = LightDataset({out: (conds_keys, result_array)}, coords=properties.coords)
        properties.merge(result, inplace=True, compat='equals')
    
    # SEGMENT 32: Final result formatting
    debug_log(32, "Final result formatting")
    if to_xarray:
        properties = wks.eq.get_dataset()
        if verbose:
            print("  converted to xarray Dataset")
    properties.attrs['created'] = datetime.now().isoformat()
    if verbose:
        print(f"  added creation timestamp: {properties.attrs['created']}")
    if len(kwargs) > 0:
        warnings.warn('The following equilibrium keyword arguments were passed, but unused:\n{}'.format(kwargs))
    
    # SEGMENT 40: DEBUG OUTPUT AND CLEANUP
    debug_log(40, "Equilibrium calculation complete", {
        "converged": True,  # If we got here, calculation completed
        "properties_shape": properties.GM.shape if hasattr(properties, 'GM') else "unknown",
        "final_GM": float(properties.GM.values.flat[0]) if hasattr(properties, 'GM') and hasattr(properties.GM, 'values') else "unknown"
    })
    
    # DEBUG: Check CPU buffer contents before closing
    if verbose:
        from pycalphad.core.debug_output import _cpu_buffer
        print(f"[CPU] DEBUG: CPU buffer has {len(_cpu_buffer)} entries before close")
        if len(_cpu_buffer) > 0:
            print(f"[CPU] DEBUG: First entry: {_cpu_buffer[0]}")
    
    close_debug_output(mode="CPU")
    
    return properties
