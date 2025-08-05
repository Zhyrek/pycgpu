#!/usr/bin/env python
"""Capture equilibrium matrices at iterations 0, 1, 2 for CPU and GPU."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

# Storage for captured matrices
captured_iterations = {
    'cpu': {},
    'gpu': {}
}

# Monkey patch the equilibrium solver to capture iteration info
import pycalphad.core.eqsolver as eq_module

original_solve = eq_module._solve_eq_at_conditions

def patched_solve_cpu(properties, phase_records, grid, conds_keys, state_variables, verbose, solver=None):
    """Patched solver to capture CPU iteration info."""
    # Hook into the solver to capture iterations
    import pycalphad.core.minimizer as min_module
    
    original_construct = min_module.construct_equilibrium_system
    iteration_count = [0]  # Use list to make it mutable in nested function
    
    def patched_construct(state, eq_matrix, eq_rhs, cond_dict, hess_data, constr_jac_data, 
                         constr_data, chem_pot, parameters, callables, num_statevars, 
                         num_components, num_constraints, num_phases, cur_iter, newton, spec):
        # Call original
        result = original_construct(state, eq_matrix, eq_rhs, cond_dict, hess_data, constr_jac_data,
                                  constr_data, chem_pot, parameters, callables, num_statevars,
                                  num_components, num_constraints, num_phases, cur_iter, newton, spec)
        
        # Capture matrix for iterations 0, 1, 2
        if cur_iter in [0, 1, 2] and iteration_count[0] < 10:  # Limit captures
            print(f"\n[CPU] Iteration {cur_iter} - Capturing equilibrium matrix")
            
            # Get active phases
            active_phases = []
            phase_amounts = []
            for i in range(num_phases):
                if state.cs_states[i].is_stable:
                    phase_name = state.cs_states[i].phase_name
                    if isinstance(phase_name, bytes):
                        phase_name = phase_name.decode('utf-8')
                    phase_amt = state.cs_states[i].NP
                    active_phases.append(phase_name)
                    phase_amounts.append(phase_amt)
                    print(f"  Active phase {i}: {phase_name} (NP={phase_amt:.6f})")
            
            # Store matrix info
            matrix_size = eq_matrix.shape[0]
            key = f"iter_{cur_iter}_{iteration_count[0]}"
            captured_iterations['cpu'][key] = {
                'active_phases': active_phases.copy(),
                'phase_amounts': phase_amounts.copy(),
                'matrix_size': matrix_size,
                'matrix_sample': eq_matrix[:min(6, matrix_size), :min(6, matrix_size)].copy(),
                'rhs_sample': eq_rhs[:min(6, matrix_size)].copy(),
                'full_shape': eq_matrix.shape
            }
            
            # Print matrix sample
            print(f"  Matrix size: {matrix_size}x{matrix_size}")
            print(f"  Matrix sample (first 6x6):")
            for i in range(min(6, matrix_size)):
                row_str = " ".join(f"{eq_matrix[i,j]:+.6e}" for j in range(min(6, matrix_size)))
                print(f"    Row {i}: {row_str} | RHS: {eq_rhs[i]:+.6e}")
            
            iteration_count[0] += 1
        
        return result
    
    # Apply patch
    min_module.construct_equilibrium_system = patched_construct
    
    try:
        # Call original solver
        result = original_solve(properties, phase_records, grid, conds_keys, state_variables, verbose, solver)
    finally:
        # Restore original
        min_module.construct_equilibrium_system = original_construct
    
    return result

def test_iteration_matrices():
    """Compare equilibrium matrices at early iterations."""
    
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    phases = ['AU2BI_C15', 'BCC_A2', 'FCC_A1', 'HCP_A3', 'LIQUID', 'RHOMBOHEDRAL_A7']
    
    conditions = {
        v.X('BI'): 0.1,
        v.T: 400,
        v.P: 101325
    }
    
    print("="*80)
    print("EQUILIBRIUM MATRIX COMPARISON - ITERATIONS 0, 1, 2")
    print("="*80)
    
    # Clear captures
    captured_iterations['cpu'] = {}
    captured_iterations['gpu'] = {}
    
    # CPU calculation with patched solver
    print("\n--- CPU CALCULATION ---")
    eq_module._solve_eq_at_conditions = patched_solve_cpu
    result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
    eq_module._solve_eq_at_conditions = original_solve  # Restore
    
    cpu_gm = result_cpu.GM.values[0,0,0,0]
    cpu_phases = result_cpu.Phase.values[0,0,0,0]
    cpu_np = result_cpu.NP.values[0,0,0,0]
    
    print(f"\nCPU Final result:")
    print(f"  GM: {cpu_gm:.6f}")
    print("  Active phases:")
    for phase, amount in zip(cpu_phases, cpu_np):
        if phase != '' and amount > 1e-8:
            print(f"    {phase}: {amount:.6f}")
    
    # GPU calculation - it prints its own iteration info
    print("\n\n--- GPU CALCULATION ---")
    # Set environment variable to enable GPU iteration debug
    import os
    os.environ['GPU_DEBUG_ITERATIONS'] = '1'
    
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)
    
    gpu_gm = result_gpu.GM.values[0,0,0,0]
    gpu_phases = result_gpu.Phase.values[0,0,0,0]
    gpu_np = result_gpu.NP.values[0,0,0,0]
    
    print(f"\nGPU Final result:")
    print(f"  GM: {gpu_gm:.6f}")
    print("  Active phases:")
    for phase, amount in zip(gpu_phases, gpu_np):
        if phase != '' and amount > 1e-8:
            print(f"    {phase}: {amount:.6f}")
    
    # Analysis
    print("\n" + "="*80)
    print("ITERATION COMPARISON")
    print("="*80)
    
    # Compare captured iterations
    cpu_keys = sorted([k for k in captured_iterations['cpu'].keys() if k.startswith('iter_')])
    
    print("\nCPU captured iterations:")
    for key in cpu_keys:
        data = captured_iterations['cpu'][key]
        print(f"\n{key}:")
        print(f"  Active phases: {data['active_phases']}")
        print(f"  Phase amounts: {[f'{x:.3f}' for x in data['phase_amounts']]}")
        print(f"  Matrix size: {data['matrix_size']}")
    
    print("\n" + "="*80)
    print("KEY OBSERVATIONS")
    print("="*80)
    
    # Check if HCP_A3 appears in early iterations
    for key in cpu_keys:
        if 'HCP_A3' in captured_iterations['cpu'][key]['active_phases']:
            print(f"⚠️  CPU has HCP_A3 active at {key}")
    
    print("\nThe GPU debug output above should show its iteration matrices.")
    print("Look for when HCP_A3 first appears in the GPU iterations.")

if __name__ == "__main__":
    test_iteration_matrices()