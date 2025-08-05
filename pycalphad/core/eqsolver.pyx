# distutils: language = c++
from collections import OrderedDict
import numpy as np
cimport numpy as np
cimport cython
from pycalphad.core.solver import Solver
from pycalphad.core.composition_set cimport CompositionSet
from pycalphad.core.phase_rec cimport PhaseRecord
from pycalphad.core.constants import *

cpdef bint add_new_phases(object composition_sets, object removed_compsets, object phase_records,
                          object grid, object current_idx, np.ndarray[ndim=1, dtype=np.float64_t] chemical_potentials,
                          double[::1] state_variables, double minimum_df, bint verbose) except *:
    """
    Attempt to add a new phase with the largest driving force (based on chemical potentials). Candidate phases
    are taken from current_grid and modify the composition_sets object. The function returns a boolean indicating
    whether it modified composition_sets.
    """
    cdef double[:] driving_forces
    cdef int comp_idx
    cdef int df_idx = 0
    cdef double largest_df = -np.inf
    cdef double[:] df_comp
    cdef double[:,::1] current_grid_Y = grid.Y[*current_idx, ...]
    cdef double[:,::1] current_grid_X = grid.X[*current_idx, ...]
    cdef np.ndarray current_grid_Phase = grid.Phase[*current_idx, ...]
    cdef unicode df_phase_name
    cdef CompositionSet compset = composition_sets[0]
    cdef int num_statevars = len(compset.phase_record.state_variables)
    cdef bint distinct = False
    
    from pycalphad.core.debug_output import debug_log
    
    # SEGMENT 41: PHASE ADDITION - DRIVING FORCE CALCULATION
    debug_log(41, "Phase addition - driving force calculation")
    
    driving_forces = np.dot(current_grid_X, chemical_potentials) - grid.GM[*current_idx, ...]
    
    debug_log(f"  chemical_potentials: {np.array(chemical_potentials)}", verbose)
    debug_log(f"  max_driving_force: {np.max(driving_forces):.15e}", verbose)
    debug_log(f"  min_driving_force: {np.min(driving_forces):.15e}", verbose)
    
    for i in range(driving_forces.shape[0]):
        if driving_forces[i] > largest_df:
            df_comp = current_grid_Y[i]
            df_phase_name = <unicode>current_grid_Phase[i]
            distinct = True
            for compset in removed_compsets:
                if df_phase_name != compset.phase_record.phase_name:
                    continue
                distinct = False
                for comp_idx in range(compset.phase_record.phase_dof):
                    if abs(df_comp[comp_idx] - compset.dof[num_statevars+comp_idx]) > 10*COMP_DIFFERENCE_TOL:
                        distinct = True
                        break
                if not distinct:
                    break
            if not distinct:
                continue
            largest_df = driving_forces[i]
            df_idx = i
    
    # SEGMENT 42: PHASE ADDITION - DECISION
    debug_log(42, "Phase addition - decision")
    debug_log(f"  largest_df: {largest_df:.15e}", verbose)
    debug_log(f"  minimum_df: {minimum_df:.15e}", verbose)
    debug_log(f"  will_add_phase: {largest_df > minimum_df}", verbose)
    
    if largest_df > minimum_df:
        # To add a phase, must not be within COMP_DIFFERENCE_TOL of composition of the same phase of its type
        df_comp = current_grid_X[df_idx]
        df_phase_name = <unicode>current_grid_Phase[df_idx]
        
        debug_log(f"  candidate_phase: {df_phase_name}", verbose)
        debug_log(f"  candidate_composition: {np.array(df_comp)}", verbose)
        
        if df_phase_name == '_FAKE_':
            return False
        for compset in composition_sets:
            if compset.phase_record.phase_name != df_phase_name:
                continue
            distinct = False
            for comp_idx in range(df_comp.shape[0]):
                if abs(df_comp[comp_idx] - compset.X[comp_idx]) > COMP_DIFFERENCE_TOL:
                    distinct = True
            if not distinct:
                return False
        compset = CompositionSet(phase_records[df_phase_name])
        compset.update(current_grid_Y[df_idx, :compset.phase_record.phase_dof], 1e-6,
                       state_variables)
        composition_sets.append(compset)
        debug_log(f"  adding_phase: {df_phase_name} with driving force {largest_df:.15e}", verbose)
        return True
    
    return False

@cython.boundscheck(False)
cdef int argmax(double* a, int a_shape) nogil:
    cdef int i
    cdef int result = 0
    cdef double highest = -1e30
    for i in range(a_shape):
        if a[i] > highest:
            highest = a[i]
            result = i
    return result

def add_nearly_stable(object composition_sets, object phase_records,
                      object grid, object current_idx, np.ndarray[ndim=1, dtype=np.float64_t] chemical_potentials,
                      double[::1] state_variables, double minimum_df, bint verbose):
    cdef double[::1] driving_forces, driving_forces_for_phase
    cdef double[:,::1] current_grid_Y = grid.Y[*current_idx, ...]
    cdef double[:,::1] current_grid_X = grid.X[*current_idx, ...]
    cdef double[::1] current_grid_GM = grid.GM[*current_idx, ...]
    cdef unicode phase_name
    cdef CompositionSet compset = composition_sets[0]
    cdef set entered_phases = {compset.phase_record.phase_name for compset in composition_sets}
    cdef PhaseRecord phase_record
    cdef int num_statevars = len(compset.phase_record.state_variables)
    cdef int df_idx, minimum_df_idx
    cdef bint phases_added = False
    
    from pycalphad.core.debug_output import debug_log
    
    driving_forces = np.dot(current_grid_X, chemical_potentials) - current_grid_GM
    # Add unrepresented phases as metastable composition sets
    # This should help catch phases around the limit of stability
    for phase_name in sorted(phase_records.keys()):
        if phase_name in entered_phases:
            continue
        phase_record = phase_records[phase_name]
        phase_indices = grid.attrs['phase_indices'].get(phase_name, slice(0,0))
        if phase_indices.start == phase_indices.stop:
            # Phase has zero feasible grid points to consider
            continue
        driving_forces_for_phase = driving_forces[phase_indices.start:phase_indices.stop]
        minimum_df_idx = argmax(&driving_forces_for_phase[0], driving_forces_for_phase.shape[0])
        if driving_forces_for_phase[minimum_df_idx] >= minimum_df:
            phases_added = True
            df_idx = phase_indices.start + minimum_df_idx
            compset = CompositionSet(phase_record)
            compset.update(current_grid_Y[df_idx, :phase_record.phase_dof], 0.0, state_variables)
            debug_log(f"  adding_metastable: {phase_name} with driving force {driving_forces_for_phase[minimum_df_idx]:.15e}", verbose)
            composition_sets.append(compset)
    return phases_added

def _solve_eq_at_conditions(properties, phase_records, grid, conds_keys, state_variables, verbose, solver=None):
    """
    Compute equilibrium for the given conditions.
    This private function is meant to be called from a worker subprocess.
    For that case, usually only a small slice of the master 'properties' is provided.
    Since that slice will be copied, we also return the modified 'properties'.

    Parameters
    ----------
    properties : Dataset
        Will be modified! Thermodynamic properties and conditions.
    phase_records : dict of PhaseRecord
        Details on phase callables.
    grid : Dataset
        Sample of energy landscape of the system.
    conds_keys : List[v.StateVariable]
        List of conditions sorted in dimension order.
    state_variables : List[v.StateVariable]
        List of state variables sorted in dimension order.
    verbose : bool
        Print details.
    solver : pycalphad.core.solver.SolverBase
        Instance of a SolverBase subclass. If None is supplied, defaults to a Solver.

    Returns
    -------
    properties : Dataset
        Modified with equilibrium values.
    """
    cdef double indep_sum
    cdef int num_phases, num_vars, cur_iter, old_phase_length, new_phase_length, var_idx, dof_idx, comp_idx, phase_idx, sfidx, pfidx, m, n
    cdef bint converged, changed_phases
    cdef double vmax, minimum_df
    cdef PhaseRecord phase_record
    cdef CompositionSet compset
    cdef double[:,::1] l_hessian
    cdef double[:,:] inv_hess
    cdef double[::1] gradient_term, mass_buf
    cdef np.ndarray[ndim=1, dtype=np.float64_t] p_y, l_constraints, step, chemical_potentials
    cdef np.ndarray[ndim=1, dtype=np.float64_t] site_fracs, l_multipliers, phase_fracs
    cdef np.ndarray[ndim=2, dtype=np.float64_t] constraint_jac
    from pycalphad.core.debug_output import debug_log
    
    iter_solver = solver if solver is not None else Solver(verbose=verbose, remove_metastable=True)

    # DEBUG: Initialize debug output counters
    debug_condition_counter = 0 if verbose else -1

    # Factored out via profiling
    prop_MU_values = properties.MU
    prop_NP_values = properties.NP
    prop_Phase_values = properties.Phase
    prop_X_values = properties.X
    prop_Y_values = properties.Y
    prop_GM_values = properties.GM
    str_state_variables = [str(k) for k in state_variables if str(k) in grid.coords.keys()]
    it = np.nditer(prop_GM_values, flags=['multi_index'])

    while not it.finished:

        
        # A lot of this code relies on cur_conds being ordered!
        converged = False
        changed_phases = False
        cur_conds = OrderedDict(zip(conds_keys,
                                    [np.asarray(properties.coords[str(b)][a], dtype=np.float64)
                                     for a, b in zip(it.multi_index, conds_keys)]))

        
        # DEBUG: Start condition debug logging
        if verbose:
            debug_condition_counter += 1
        # assume 'points' and other dimensions (internal dof, etc.) always follow
        local_idx = [it.multi_index[i] for i, key in enumerate(conds_keys)
                     if getattr(key, 'phase_name', None) is not None]
        sv_idx = [it.multi_index[i] for i, key in enumerate(conds_keys)
                  if (str(key) in str_state_variables)]
        curr_idx = local_idx + sv_idx
        state_variable_values = [cur_conds[state_variables[str_state_variables.index(key)]] for key in str_state_variables]
        state_variable_values = np.array(state_variable_values)
        
        
        # sum of independently specified components
        indep_sum = np.sum([float(val) for i, val in cur_conds.items() if str(i).startswith('X_')])
        
        
        if indep_sum > 1:
            # Sum of independent component mole fractions greater than one
            # Skip this condition set
            # We silently allow this to make 2-D composition mapping easier
            prop_MU_values[it.multi_index] = np.nan
            prop_NP_values[it.multi_index + np.index_exp[:]] = np.nan
            prop_Phase_values[it.multi_index + np.index_exp[:]] = ''
            prop_X_values[it.multi_index + np.index_exp[:]] = np.nan
            prop_Y_values[it.multi_index] = np.nan
            prop_GM_values[it.multi_index] = np.nan
            it.iternext()
            continue


        
        composition_sets = []
        removed_compsets = []
        
        # DEBUG: Print what phases we're starting with
        if verbose:
            print(f"[CPU] Creating composition sets from starting_point result:")
            print(f"[CPU]   prop_Phase_values at index: {prop_Phase_values[it.multi_index]}")
            print(f"[CPU]   prop_NP_values at index: {prop_NP_values[it.multi_index]}")
        
        for phase_idx, phase_name in enumerate(prop_Phase_values[it.multi_index]):
            if phase_name == '' or phase_name == '_FAKE_':
                continue
            phase_record = phase_records[phase_name]
            sfx = prop_Y_values[it.multi_index + np.index_exp[phase_idx, :phase_record.phase_dof]]
            phase_amt = prop_NP_values[it.multi_index + np.index_exp[phase_idx]]
            
            # DEBUG: Log initial phase data for first few conditions
            if verbose and debug_condition_counter <= 3:
                debug_log(f"  cpu_initial_phase_{phase_idx}_amount: {phase_amt:.15e}", verbose)
                debug_log(f"  cpu_initial_phase_{phase_idx}_site_fractions: {np.array(sfx)}", verbose)
            phase_amt = max(phase_amt, MIN_PHASE_FRACTION)
            compset = CompositionSet(phase_record)
            compset.update(sfx, phase_amt, state_variable_values)
            composition_sets.append(compset)
            
            # DEBUG: Log initial phase setup - numerical only for first 3 conditions
            if verbose and debug_condition_counter <= 3:
                debug_log(f"  cpu_phase_{phase_idx}_energy: {compset.energy:.15e}", verbose)
                debug_log(f"  cpu_phase_{phase_idx}_amount: {phase_amt:.15e}", verbose)
                debug_log(f"  cpu_phase_{phase_idx}_X_after_update: {compset.X}", verbose)
        
        chemical_potentials = prop_MU_values[it.multi_index]
        energy = prop_GM_values[it.multi_index]
        
        # DEBUG: Log initial chemical potentials and energy - first 3 conditions only
        if verbose and debug_condition_counter <= 3:
            debug_log(f"  cpu_initial_chemical_potentials: {chemical_potentials}", verbose)
            debug_log(f"  cpu_initial_total_energy: {energy:.15e}", verbose)
        
        add_nearly_stable(composition_sets, phase_records, grid, curr_idx, chemical_potentials,
                          state_variable_values, -1000, verbose)
        
        #print('Composition Sets', composition_sets)
        phase_amt_sum = 0.0
        for compset in composition_sets:
            phase_amt_sum += compset.NP
        
        for compset in composition_sets:
            compset.NP /= phase_amt_sum
            
        # DEBUG: Log normalized phase amounts - first 3 conditions only
        if verbose and debug_condition_counter <= 3:
            debug_log(f"  cpu_phase_amount_normalization_sum: {phase_amt_sum:.15e}", verbose)
            for i, compset in enumerate(composition_sets):
                debug_log(f"  cpu_normalized_phase_{i}_amount: {compset.NP:.15e}", verbose)
        iterations = 0
        history = []
        

        
        while (iterations < 10) and (not iter_solver.ignore_convergence):

            
            if len(composition_sets) == 0:
                changed_phases = False
                break
            
            # DEBUG: Log solver iteration start - first 3 conditions only
            if verbose and debug_condition_counter <= 3:
                debug_log(f"  cpu_solver_iteration_{iterations + 1}_start", verbose)
                for i, compset in enumerate(composition_sets):
                    debug_log(f"  cpu_pre_solve_phase_{i}_amount: {compset.NP:.15e}", verbose)
                    debug_log(f"  cpu_pre_solve_phase_{i}_energy: {compset.energy:.15e}", verbose)
                    
                # Enable verbose mode on solver
                if hasattr(iter_solver, 'verbose'):
                    iter_solver.verbose = True
            


            
            result = iter_solver.solve(composition_sets, cur_conds)
            
            # DEBUG: Check result - first 3 conditions only
            if verbose and debug_condition_counter <= 3:
                if result is not None:
                    debug_log(f"  cpu_solver_converged: {result.converged}", verbose)
                    debug_log(f"  cpu_solver_chemical_potentials: {result.chemical_potentials}", verbose)
            
            if result is None:
                break
            

            chemical_potentials[:] = result.chemical_potentials
            
            # DEBUG: Log solver results - first 3 conditions only
            if verbose and debug_condition_counter <= 3:
                debug_log(f"  cpu_updated_chemical_potentials: {chemical_potentials}", verbose)
                for i, compset in enumerate(composition_sets):
                    debug_log(f"  cpu_post_solve_phase_{i}_amount: {compset.NP:.15e}", verbose)
                    debug_log(f"  cpu_post_solve_phase_{i}_energy: {compset.energy:.15e}", verbose)
            

            
            changed_phases = add_new_phases(composition_sets, removed_compsets, phase_records,
                                            grid, curr_idx, chemical_potentials, state_variable_values,
                                            1e-4, verbose)
            
            
            # DEBUG: Log phase changes - first 3 conditions only
            if verbose and changed_phases and debug_condition_counter <= 3:
                debug_log(f"  cpu_phases_changed_new_count: {len(composition_sets)}", verbose)
            
            iterations += 1
            if not changed_phases:
                break
        if changed_phases:

            
            # DEBUG: Final solve after phase changes - first 3 conditions only
            if verbose and debug_condition_counter <= 3:
                debug_log("  cpu_final_solve_after_phase_changes", verbose)
            
            result = iter_solver.solve(composition_sets, cur_conds)
            chemical_potentials[:] = result.chemical_potentials

            
            # DEBUG: Log final solve results - first 3 conditions only
            if verbose and debug_condition_counter <= 3:
                debug_log(f"  cpu_final_solve_converged: {result.converged}", verbose)
                debug_log(f"  cpu_final_chemical_potentials: {chemical_potentials}", verbose)

        if not iter_solver.ignore_convergence:
            converged = result.converged
        else:
            converged = True

        if converged:

            
            # DEBUG: Log final equilibrium results - first 3 conditions only with numerical data
            if verbose and debug_condition_counter <= 3:
                debug_log(f"  cpu_final_converged: {converged}", verbose)
                total_gm = 0
                for i, compset in enumerate(composition_sets):
                    debug_log(f"  cpu_final_phase_{i}_amount: {compset.NP:.15e}", verbose)
                    debug_log(f"  cpu_final_phase_{i}_energy: {compset.energy:.15e}", verbose)
                    debug_log(f"  cpu_final_phase_{i}_X: {np.array(compset.X)}", verbose)
                    total_gm += compset.NP * compset.energy
                debug_log(f"  cpu_final_total_gm: {total_gm:.15e}", verbose)
                debug_log(f"  cpu_final_chemical_potentials: {chemical_potentials}", verbose)
            
            prop_MU_values[it.multi_index] = chemical_potentials
            prop_Phase_values[it.multi_index] = ''
            prop_NP_values[it.multi_index + np.index_exp[:len(composition_sets)]] = [compset.NP for compset in composition_sets]
            prop_NP_values[it.multi_index + np.index_exp[len(composition_sets):]] = np.nan
            prop_Y_values[it.multi_index] = np.nan
            prop_X_values[it.multi_index + np.index_exp[:]] = 0
            prop_GM_values[it.multi_index] = 0

            # Copy out any free state variables (P, T, etc.)
            # All CompositionSets should have equal state variable values, so we copy from the first one
            for sv_idx, ssv in enumerate(str_state_variables):
                # If the state variable is listed as a free variable in our results
                # The LightDataset interface is not clear here
                if properties.data_vars.get(ssv, None) is not None:
                    properties.data_vars[ssv][1][it.multi_index] = composition_sets[0].dof[sv_idx]
            for phase_idx in range(len(composition_sets)):
                prop_Phase_values[it.multi_index + np.index_exp[phase_idx]] = composition_sets[phase_idx].phase_record.phase_name
            for phase_idx in range(len(composition_sets), prop_Phase_values.shape[-1]):
                prop_Phase_values[it.multi_index + np.index_exp[phase_idx]] = ''
                prop_X_values[it.multi_index + np.index_exp[phase_idx, :]] = np.nan

            
            # SEGMENT 40: FINAL GIBBS ENERGY CALCULATION
            # Only print for first 3 conditions to avoid clutter
            if debug_condition_counter <= 3:
                debug_log(40, "Final Gibbs energy calculation", condition_idx=it.multi_index[0])
            
            var_offset = 0
            total_comp = np.zeros(prop_X_values.shape[-1])
            for phase_idx in range(len(composition_sets)):
                compset = composition_sets[phase_idx]
                prop_Y_values[it.multi_index + np.index_exp[phase_idx, :compset.phase_record.phase_dof]] = \
                    compset.dof[len(compset.phase_record.state_variables):]
                prop_X_values[it.multi_index + np.index_exp[phase_idx, :]] = compset.X
                prop_GM_values[it.multi_index] += compset.NP * compset.energy
                var_offset += compset.phase_record.phase_dof
                
                debug_log(f"  phase_{phase_idx}_{compset.phase_record.phase_name}_Y: {compset.dof[len(compset.phase_record.state_variables):]}", verbose)
                debug_log(f"  phase_{phase_idx}_{compset.phase_record.phase_name}_X: {compset.X}", verbose)
                debug_log(f"  phase_{phase_idx}_{compset.phase_record.phase_name}_contribution: NP={compset.NP:.15e} * energy={compset.energy:.15e} = {compset.NP * compset.energy:.15e}", verbose)
                
                
                # DEBUG: Log per-phase energy contribution - first 3 conditions only
                if verbose and debug_condition_counter <= 3:
                    debug_log(f"  cpu_phase_{phase_idx}_energy_contribution: {compset.NP * compset.energy:.15e}", verbose)
            
            debug_log(f"  final_GM: {prop_GM_values[it.multi_index]:.15e}", verbose)

        else:

            
            # DEBUG: Log equilibrium failure - first 3 conditions only
            if verbose and debug_condition_counter <= 3:
                debug_log(f"  cpu_equilibrium_failed_converged: {converged}", verbose)
            
            prop_MU_values[it.multi_index] = np.nan
            prop_NP_values[it.multi_index] = np.nan
            prop_X_values[it.multi_index] = np.nan
            prop_Y_values[it.multi_index] = np.nan
            prop_GM_values[it.multi_index] = np.nan
            prop_Phase_values[it.multi_index] = ''
            
            
        # Remove end condition debug logging - no longer needed
        

        it.iternext()

    # Remove return logging - no numerical value to compare
        
    return properties
