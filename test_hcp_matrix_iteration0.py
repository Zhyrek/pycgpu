#!/usr/bin/env python
"""Compare equilibrium matrices at iteration 0 between CPU and GPU for 4-phase vs 6-phase cases."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

# Monkey patch to capture equilibrium matrix at iteration 0
import pycalphad.core.minimizer as min_module
import pycalphad.gpu.minimizer as gpu_min_module

# Storage for captured matrices
captured_matrices = {'cpu': None, 'gpu': None}

# Original CPU function
original_cpu_fill_equilibrium_system = min_module.fill_equilibrium_system

def patched_cpu_fill_equilibrium_system(state, eq_matrix, eq_rhs, cond_dict, hess_data, constr_jac_data, constr_data, chem_pot, parameters, callables, num_statevars, num_components, num_constraints, num_phases, cur_iter, newton, spec):
    result = original_cpu_fill_equilibrium_system(state, eq_matrix, eq_rhs, cond_dict, hess_data, constr_jac_data, constr_data, chem_pot, parameters, callables, num_statevars, num_components, num_constraints, num_phases, cur_iter, newton, spec)
    
    # Capture matrix at iteration 0
    if cur_iter == 0 and captured_matrices['cpu'] is None:
        import numpy as np
        matrix_size = eq_matrix.shape[0]
        captured_matrices['cpu'] = {
            'matrix': np.array(eq_matrix[:matrix_size, :matrix_size]).copy(),
            'rhs': np.array(eq_rhs[:matrix_size]).copy(),
            'num_rows': matrix_size
        }
        print(f"[CPU] Captured equilibrium matrix at iteration 0 (size {matrix_size}x{matrix_size})")
        
        # Print matrix details
        print(f"[CPU] Matrix at iteration 0:")
        for i in range(min(4, matrix_size)):
            row_str = " ".join(f"{eq_matrix[i,j]:+.6e}" for j in range(min(4, matrix_size)))
            if matrix_size > 4:
                row_str += " ..."
            print(f"  Row {i}: {row_str} | RHS: {eq_rhs[i]:+.6e}")
        if matrix_size > 4:
            print(f"  ... ({matrix_size - 4} more rows)")
    
    return result

# Monkey patch
min_module.fill_equilibrium_system = patched_cpu_fill_equilibrium_system

def compare_matrices():
    """Compare equilibrium matrices for 4-phase vs 6-phase cases."""
    
    # Load database
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    
    conditions = {
        v.X('BI'): 0.1,
        v.T: 400,
        v.P: 101325
    }
    
    # Test cases
    test_cases = [
        (['FCC_A1', 'AU2BI_C15', 'BCC_A2', 'HCP_A3'], "4-phase case"),
        (['AU2BI_C15', 'BCC_A2', 'FCC_A1', 'HCP_A3', 'LIQUID', 'RHOMBOHEDRAL_A7'], "6-phase case (all)"),
    ]
    
    for phases, case_name in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing {case_name}")
        print(f"Phases: {phases}")
        print(f"{'='*60}")
        
        # Reset captured matrices
        captured_matrices['cpu'] = None
        captured_matrices['gpu'] = None
        
        # CPU calculation
        print(f"\n--- CPU Calculation ---")
        result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
        cpu_gm = result_cpu.GM.values[0,0,0,0]
        cpu_phases = result_cpu.Phase.values[0,0,0,0]
        cpu_np = result_cpu.NP.values[0,0,0,0]
        
        print(f"CPU GM: {cpu_gm:.6f}")
        print("CPU active phases:")
        for phase, amount in zip(cpu_phases, cpu_np):
            if amount > 1e-8 and phase != '':
                print(f"  {phase}: {amount:.6f}")
        
        # Store CPU matrix
        cpu_matrix_data = captured_matrices['cpu'].copy() if captured_matrices['cpu'] else None
        
        # GPU calculation
        print(f"\n--- GPU Calculation ---")
        
        # Need to capture GPU matrix from kernel output
        # For now, just run the calculation
        result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)
        gpu_gm = result_gpu.GM.values[0,0,0,0]
        gpu_phases = result_gpu.Phase.values[0,0,0,0]
        gpu_np = result_gpu.NP.values[0,0,0,0]
        
        print(f"\nGPU GM: {gpu_gm:.6f}")
        print("GPU active phases:")
        for phase, amount in zip(gpu_phases, gpu_np):
            if amount > 1e-8 and phase != '':
                print(f"  {phase}: {amount:.6f}")
        
        # Compare results
        gm_diff = abs(gpu_gm - cpu_gm)
        print(f"\nGM difference: {gm_diff:.2e}")
        
        if gm_diff > 1.0:
            print("❌ SIGNIFICANT DIFFERENCE - GPU found different equilibrium")
            
            # Analyze phase differences
            cpu_active_set = set(p for p, a in zip(cpu_phases, cpu_np) if a > 1e-8 and p != '')
            gpu_active_set = set(p for p, a in zip(gpu_phases, gpu_np) if a > 1e-8 and p != '')
            
            only_cpu = cpu_active_set - gpu_active_set
            only_gpu = gpu_active_set - cpu_active_set
            
            if only_cpu:
                print(f"  Phases only in CPU: {only_cpu}")
            if only_gpu:
                print(f"  Phases only in GPU: {only_gpu}")
                
            if 'HCP_A3' in only_gpu:
                print("\n🔍 HCP_A3 ANALYSIS:")
                print("  HCP_A3 is incorrectly active in GPU result")
                print("  This suggests the equilibrium matrix at iteration 0")
                print("  may be constructed differently due to the 0.5 sublattice")
                
        else:
            print("✅ Results match within tolerance")
        
        # Compare matrices if we have CPU data
        if cpu_matrix_data:
            print(f"\n--- Matrix Analysis ---")
            print(f"CPU matrix size: {cpu_matrix_data['num_rows']}x{cpu_matrix_data['num_rows']}")
            
            # Check for specific HCP_A3 related patterns
            # The matrix rows correspond to:
            # - Chemical potential equations (one per component)
            # - Mass balance equations
            # - Phase internal DOF equations
            
            # Look for unusual values that might indicate HCP_A3 issues
            cpu_matrix = cpu_matrix_data['matrix']
            
            # Check for very large or very small values
            max_val = np.max(np.abs(cpu_matrix))
            min_nonzero = np.min(np.abs(cpu_matrix[cpu_matrix != 0]))
            
            print(f"Matrix value range: [{min_nonzero:.2e}, {max_val:.2e}]")
            
            # Check condition number if possible
            try:
                cond = np.linalg.cond(cpu_matrix)
                print(f"Matrix condition number: {cond:.2e}")
            except:
                print("Could not compute condition number")

if __name__ == "__main__":
    compare_matrices()