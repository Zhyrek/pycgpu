import numpy as np
from collections import namedtuple
from pycalphad.core.minimizer import SystemSpecification

SolverResult = namedtuple('SolverResult', ['converged', 'x', 'chemical_potentials'])

class SolverBase(object):
    """"Base class for solvers."""
    ignore_convergence = False
    def solve(self, composition_sets, conditions):
        """
        *Implement this method.*
        Minimize the energy under the specified conditions using the given candidate composition sets.

        Parameters
        ----------
        composition_sets : List[pycalphad.core.composition_set.CompositionSet]
            List of CompositionSet objects in the starting point. Modified in place.
        conditions : OrderedDict[str, float]
            Conditions to satisfy.

        Returns
        -------
        pycalphad.core.solver.SolverResult
        """
        raise NotImplementedError("A subclass of Solver must be implemented.")


class Solver(SolverBase):
    def __init__(self, verbose=False, remove_metastable=True, **options):
        self.verbose = verbose
        self.remove_metastable = remove_metastable


    def get_system_spec(self, composition_sets, conditions):
        """
        Create a SystemSpecification object for the specified conditions.

        Parameters
        ----------
        composition_sets : List[pycalphad.core.composition_set.CompositionSet]
            List of CompositionSet objects in the starting point. Modified in place.
        conditions : OrderedDict[StateVariable, float]
            Conditions to satisfy.

        Returns
        -------
        SystemSpecification

        """
        # Prevent circular import
        from pycalphad.variables import ChemicalPotential, MassFraction, MoleFraction, \
            SiteFraction
        from pycalphad.core.debug_output import debug_log, debug_log_array_comparison
        
        # SEGMENT 16: GET SYSTEM SPECIFICATION - SETUP
        debug_log(16, "Get system specification - setup")
        
        compsets = composition_sets
        state_variables = compsets[0].phase_record.state_variables
        nonvacant_elements = compsets[0].phase_record.nonvacant_elements
        num_statevars = len(state_variables)
        num_components = len(nonvacant_elements)
        
        debug_log(f"  state_variables: {[str(sv) for sv in state_variables]}", self.verbose)
        debug_log(f"  nonvacant_elements: {list(nonvacant_elements)}", self.verbose)
        debug_log(f"  num_statevars: {num_statevars}", self.verbose)
        debug_log(f"  num_components: {num_components}", self.verbose)
        
        chemical_potentials = np.zeros(num_components)
        prescribed_mole_fraction_coefficients = []
        prescribed_mole_fraction_rhs = []
        local_conditions = {key: value for key, value in conditions.items()
                            if getattr(key, 'phase_name', None) is not None}
        
        debug_log(f"  local_conditions: {local_conditions}", self.verbose)
        
        for compset in compsets:
            phase_local_conditions = {key: value for key, value in local_conditions.items()
                                      if compset.phase_record.phase_name == key.phase_name}
            if len(phase_local_conditions) > 0:
                compset.set_local_conditions(phase_local_conditions)
        # SEGMENT 17: GET SYSTEM SPEC - PROCESS CONDITIONS
        debug_log(17, "Get system spec - process conditions")
        
        for cond, value in conditions.items():
            if isinstance(cond, MoleFraction) and cond.phase_name is None:
                el = str(cond)[2:]
                el_idx = list(nonvacant_elements).index(el)
                prescribed_mole_fraction_rhs.append(np.asarray(value).flat[0])
                coefs = np.zeros(num_components)
                coefs[el_idx] = 1.0
                prescribed_mole_fraction_coefficients.append(coefs)
                debug_log(f"  MoleFraction_{el}: value={value}, el_idx={el_idx}", self.verbose)
            elif isinstance(cond, MoleFraction) and cond.phase_name is not None:
                # phase-local condition; already handled
                continue
            elif isinstance(cond, SiteFraction):
                # phase-local condition; already handled
                continue
            elif isinstance(cond, MassFraction):
                # wA = k -> (1-k)*MWA*xA - k*MWB*xB - k*MWC*xC = 0
                el = str(cond)[2:]
                el_idx = list(nonvacant_elements).index(el)
                coef_vector = np.zeros(num_components)
                coef_vector -= value
                coef_vector[el_idx] += 1
                # multiply coef_vector times a vector of molecular weights
                coef_vector = np.multiply(coef_vector, compsets[0].phase_record.molar_masses)
                prescribed_mole_fraction_rhs.append(0.)
                prescribed_mole_fraction_coefficients.append(coef_vector)
                debug_log(f"  MassFraction_{el}: value={value}", self.verbose)
            elif str(cond).startswith('LinComb_'):
                coefs = np.zeros(num_components)
                constant = 0.0
                for symbol, coef in zip(cond.symbols, cond.coefs):
                    if symbol == 1:
                        constant = coef
                        continue
                    el = str(symbol)[2:]
                    el_idx = list(nonvacant_elements).index(el)
                    coefs[el_idx] = coef
                if cond.denominator == 1:
                    prescribed_mole_fraction_rhs.append(float(value) - float(constant))
                else:
                    # Adjust coefficients to account for molar ratio
                    prescribed_mole_fraction_rhs.append(-float(constant))
                    denominator_idx = cond.symbols.index(cond.denominator)
                    coefs[denominator_idx] -= float(value)
                prescribed_mole_fraction_coefficients.append(coefs)
                debug_log(f"  LinComb: {str(cond)}", self.verbose)
        
        debug_log(f"  num_constraints: {len(prescribed_mole_fraction_coefficients)}", self.verbose)
        # SEGMENT 18: GET SYSTEM SPEC - IDENTIFY INDICES
        debug_log(18, "Get system spec - identify indices")
        
        prescribed_mole_fraction_coefficients = np.atleast_2d(prescribed_mole_fraction_coefficients)
        prescribed_mole_fraction_rhs = np.array(prescribed_mole_fraction_rhs)
        prescribed_system_amount = conditions.get('N', 1.0)
        
        debug_log(f"  prescribed_system_amount: {prescribed_system_amount}", self.verbose)
        
        fixed_chemical_potential_indices = np.array([nonvacant_elements.index(str(key)[3:]) for key in conditions.keys() if str(key).startswith('MU_')], dtype=np.int32)
        free_chemical_potential_indices = np.array(sorted(set(range(num_components)) - set(fixed_chemical_potential_indices)), dtype=np.int32)
        
        debug_log(f"  fixed_chemical_potential_indices: {fixed_chemical_potential_indices}", self.verbose)
        debug_log(f"  free_chemical_potential_indices: {free_chemical_potential_indices}", self.verbose)
        
        for fixed_chempot_index in fixed_chemical_potential_indices:
            el = nonvacant_elements[fixed_chempot_index]
            chemical_potentials[fixed_chempot_index] = conditions.get(ChemicalPotential(el))
        
        fixed_statevar_indices = []
        for statevar_idx, statevar in enumerate(state_variables):
            if str(statevar) in [str(k) for k in conditions.keys()]:
                fixed_statevar_indices.append(statevar_idx)
        free_statevar_indices = np.array(sorted(set(range(num_statevars)) - set(fixed_statevar_indices)), dtype=np.int32)
        fixed_statevar_indices = np.array(fixed_statevar_indices, dtype=np.int32)
        
        debug_log(f"  fixed_statevar_indices: {fixed_statevar_indices}", self.verbose)
        debug_log(f"  free_statevar_indices: {free_statevar_indices}", self.verbose)
        
        fixed_stable_compset_indices = np.array([i for i, compset in enumerate(compsets) if compset.fixed], dtype=np.int32)
        
        debug_log(f"  fixed_stable_compset_indices: {fixed_stable_compset_indices}", self.verbose)
        
        # SEGMENT 19: CREATE SYSTEM SPECIFICATION OBJECT
        debug_log(19, "Create system specification object")
        
        spec = SystemSpecification(num_statevars, num_components, prescribed_system_amount,
                                   chemical_potentials, prescribed_mole_fraction_coefficients,
                                   prescribed_mole_fraction_rhs,
                                   free_chemical_potential_indices, free_statevar_indices,
                                   fixed_chemical_potential_indices, fixed_statevar_indices,
                                   fixed_stable_compset_indices)
        
        debug_log(f"  max_num_free_stable_phases: {spec.max_num_free_stable_phases}", self.verbose)
        debug_log(f"  ALLOWED_MASS_RESIDUAL: {spec.ALLOWED_MASS_RESIDUAL}", self.verbose)
        
        return spec

    @staticmethod
    def _fix_state_variables_in_compsets(composition_sets, conditions):
        "Ensure state variables in each CompositionSet are set to the fixed value."
        str_state_variables = [str(k) for k in composition_sets[0].phase_record.state_variables]
        for compset in composition_sets:
            for k,v in conditions.items():
                if str(k) in str_state_variables:
                    statevar_idx = str_state_variables.index(str(k))
                    compset.dof[statevar_idx] = v

    def solve(self, composition_sets, conditions):
        """
        Minimize the energy under the specified conditions using the given candidate composition sets.

        Parameters
        ----------
        composition_sets : List[pycalphad.core.composition_set.CompositionSet]
            List of CompositionSet objects in the starting point. Modified in place.
        conditions : OrderedDict[str, float]
            Conditions to satisfy.

        Returns
        -------
        SolverResult

        """
        from pycalphad.core.debug_output import debug_log, debug_log_array_comparison
        
        # SEGMENT 13: SOLVER INPUT VALIDATION
        debug_log(13, "Solver input validation")
        debug_log(f"  num_composition_sets: {len(composition_sets)}", self.verbose)
        
        # Extract numerical arrays for comparison
        phase_amounts = np.array([cs.NP for cs in composition_sets])
        phase_energies = np.array([cs.energy for cs in composition_sets])
        total_energy = np.sum(phase_amounts * phase_energies)
        
        # Add numerical comparisons for CPU side - force to verbose mode
        if self.verbose:
            print(f"[CPU] SEGMENT 13: Numerical validation")
            print(f"  cpu_initial_np_mean: {np.nanmean(phase_amounts):.15e}")
            print(f"  cpu_initial_np_min: {np.nanmin(phase_amounts):.15e}")
            print(f"  cpu_initial_np_max: {np.nanmax(phase_amounts):.15e}")
            print(f"  cpu_initial_energy_mean: {np.nanmean(phase_energies):.15e}")
            print(f"  cpu_initial_energy_min: {np.nanmin(phase_energies):.15e}")
            print(f"  cpu_initial_energy_max: {np.nanmax(phase_energies):.15e}")
            print(f"  cpu_total_energy: {total_energy:.15e}")
        
        # Also add numerical data to debug log for segment 13
        debug_log(13, "CPU Numerical validation", {
            "cpu_initial_np_mean": f"{np.nanmean(phase_amounts):.15e}",
            "cpu_initial_np_min": f"{np.nanmin(phase_amounts):.15e}",
            "cpu_initial_np_max": f"{np.nanmax(phase_amounts):.15e}",
            "cpu_initial_energy_mean": f"{np.nanmean(phase_energies):.15e}",
            "cpu_initial_energy_min": f"{np.nanmin(phase_energies):.15e}",
            "cpu_initial_energy_max": f"{np.nanmax(phase_energies):.15e}",
            "cpu_total_energy": f"{total_energy:.15e}"
        })
        
        # Add CPU equivalents for GPU array comparisons
        debug_log_array_comparison(13, "[CPU] Initial NP comparison", phase_amounts)
        debug_log_array_comparison(13, "[CPU] Initial Energy comparison", phase_energies)
        
        # Extract compositions if available
        if len(composition_sets) > 0:
            phase_compositions = []
            for cs in composition_sets:
                if hasattr(cs, 'X') and cs.X is not None:
                    phase_compositions.extend(cs.X)
                elif hasattr(cs, 'dof') and cs.dof is not None:
                    # Extract composition from DOF (skip state variables)
                    num_statevars = len(cs.phase_record.state_variables) if hasattr(cs.phase_record, 'state_variables') else 0
                    phase_compositions.extend(cs.dof[num_statevars:])
            
            if phase_compositions:
                comp_array = np.array(phase_compositions)
                if self.verbose:
                    print(f"  cpu_composition_mean: {np.nanmean(comp_array):.15e}")
                    print(f"  cpu_composition_min: {np.nanmin(comp_array):.15e}")
                    print(f"  cpu_composition_max: {np.nanmax(comp_array):.15e}")
                # Add composition data to segment 13
                debug_log(13, "CPU Composition validation", {
                    "cpu_composition_mean": f"{np.nanmean(comp_array):.15e}",
                    "cpu_composition_min": f"{np.nanmin(comp_array):.15e}",
                    "cpu_composition_max": f"{np.nanmax(comp_array):.15e}"
                })
                # Add CPU equivalent for GPU X comparison
                debug_log_array_comparison(13, "[CPU] Initial X comparison", comp_array)
        
        spec = self.get_system_spec(composition_sets, conditions)
        self._fix_state_variables_in_compsets(composition_sets, conditions)
        
        # SEGMENT 20: CREATE SYSTEM STATE
        debug_log(20, "Create system state")
        state = spec.get_new_state(composition_sets)
        debug_log(f"  state_created", self.verbose)
        debug_log(f"  num_phases: {len(state.compsets)}", self.verbose)
        debug_log(f"  phase_amounts: {state.phase_amt}", self.verbose)
        
        # Enable debug logging if verbose
        if self.verbose:
            try:
                state._debug_enabled = True
            except:
                pass  # Ignore if state doesn't support debug
        
        converged = spec.run_loop(state, 1000)

        # SEGMENT 38: SOLVER REMOVE METASTABLE
        from pycalphad.core.debug_output import debug_log, debug_log_array_comparison
        debug_log(38, "Solver remove metastable")
        debug_log(f"  remove_metastable_flag: {self.remove_metastable}", self.verbose)
        debug_log(f"  initial_phase_count: {len(composition_sets)}", self.verbose)
        
        if self.remove_metastable:
            phase_idx = 0
            compsets_to_remove = []
            for compset in composition_sets:
                # Mark unstable phases for removal
                if compset.NP <= 0.0 and not compset.fixed:
                    compsets_to_remove.append(int(phase_idx))
                    debug_log(f"  marking_phase_{phase_idx}_{compset.phase_record.phase_name}: NP={compset.NP:.15e}", self.verbose)
                    # Remove verbose print - numerical data already in debug_log
                phase_idx += 1
            # Watch removal order here, as the indices of composition_sets are changing!
            debug_log(f"  phases_to_remove: {len(compsets_to_remove)}", self.verbose)
            # Remove verbose print - numerical data already in debug_log
            for idx in reversed(compsets_to_remove):
                del composition_sets[idx]
        
        debug_log(f"  final_phase_count: {len(composition_sets)}", self.verbose)

        # SEGMENT 39: SOLVER RESULT ASSEMBLY
        debug_log(39, "Solver result assembly")
        
        phase_amt = [compset.NP for compset in composition_sets]
        debug_log(f"  phase_amounts: {phase_amt}", self.verbose)
        
        # Remove verbose print - numerical data already in debug_log

        x = composition_sets[0].dof
        state_variables = composition_sets[0].phase_record.state_variables
        num_statevars = len(state_variables)
        
        debug_log(f"  num_statevars: {num_statevars}", self.verbose)
        debug_log(f"  initial_dof_length: {len(x)}", self.verbose)
        
        for compset in composition_sets[1:]:
            x = np.r_[x, compset.dof[num_statevars:]]
        x = np.r_[x, phase_amt]
        chemical_potentials = np.array(state.chemical_potentials)
        
        debug_log(f"  solution_vector_length: {len(x)}", self.verbose)
        debug_log(f"  chemical_potentials: {chemical_potentials}", self.verbose)
        
        # Remove verbose prints - numerical data already in debug_log
        result = SolverResult(converged=converged, x=x, chemical_potentials=chemical_potentials)
        
        # DEBUG: Log solver result
        # Remove verbose print - no comparable GPU equivalent
        
        return result
