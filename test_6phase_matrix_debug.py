#!/usr/bin/env python
"""Debug equilibrium matrix at iteration 0 for 6-phase case where GPU includes HCP_A3."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

# Monkey patch to capture iteration 0 details
import pycalphad.core.minimizer as min_module

# Storage for captured data
captured_data = {'cpu': {}, 'gpu': {}}
original_construct_equilibrium = min_module.construct_equilibrium_system

def patched_construct_equilibrium(state, eq_matrix, eq_rhs, cond_dict, hess_data, constr_jac_data, constr_data, chem_pot, parameters, callables, num_statevars, num_components, num_constraints, num_phases, cur_iter, newton, spec):
    result = original_construct_equilibrium(state, eq_matrix, eq_rhs, cond_dict, hess_data, constr_jac_data, constr_data, chem_pot, parameters, callables, num_statevars, num_components, num_constraints, num_phases, cur_iter, newton, spec)
    
    # Capture data at iteration 0
    if cur_iter == 0 and 'matrix_captured' not in captured_data['cpu']:
        print(f"\n[CPU ITERATION 0] Capturing equilibrium matrix...")
        
        # Get active phases
        active_phases = []
        phase_amounts = []
        for i in range(num_phases):
            if state.cs_states[i].is_stable:
                phase_name = state.cs_states[i].phase_name.decode('utf-8') if hasattr(state.cs_states[i].phase_name, 'decode') else str(state.cs_states[i].phase_name)
                phase_amt = state.cs_states[i].NP
                active_phases.append(phase_name)
                phase_amounts.append(phase_amt)
                print(f"  Active phase {i}: {phase_name} (NP={phase_amt:.6f})")
        
        captured_data['cpu']['active_phases'] = active_phases
        captured_data['cpu']['phase_amounts'] = phase_amounts
        captured_data['cpu']['num_active'] = len(active_phases)
        
        # Capture matrix size
        matrix_size = eq_matrix.shape[0]
        print(f"  Matrix size: {matrix_size}x{matrix_size}")
        
        # Print first few rows of matrix
        print(f"\n[CPU] Equilibrium matrix at iteration 0:")
        for i in range(min(6, matrix_size)):
            row_str = " ".join(f"{eq_matrix[i,j]:+.6e}" for j in range(min(6, matrix_size)))
            if matrix_size > 6:
                row_str += " ..."
            print(f"  Row {i}: {row_str} | RHS: {eq_rhs[i]:+.6e}")
        
        captured_data['cpu']['matrix_captured'] = True
        
        # Check for HCP_A3
        if 'HCP_A3' in active_phases:
            print(f"\n  ⚠️ CPU has HCP_A3 active at iteration 0!")
        else:
            print(f"\n  ✓ CPU does NOT have HCP_A3 active at iteration 0")
    
    return result

# Apply patch
min_module.construct_equilibrium_system = patched_construct_equilibrium

def test_6phase_matrix():
    """Test the 6-phase case where GPU incorrectly includes HCP_A3."""
    
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    
    # All 6 phases - this is where GPU fails
    phases = ['AU2BI_C15', 'BCC_A2', 'FCC_A1', 'HCP_A3', 'LIQUID', 'RHOMBOHEDRAL_A7']
    
    conditions = {
        v.X('BI'): 0.1,
        v.T: 400,
        v.P: 101325
    }
    
    print("="*70)
    print("6-PHASE CASE - EQUILIBRIUM MATRIX AT ITERATION 0")
    print("="*70)
    print(f"Phases: {phases}")
    print(f"Conditions: X(BI)=0.1, T=400K")
    
    # Check sublattices
    print("\nPhase sublattices:")
    for phase_name in phases:
        phase = dbf.phases[phase_name]
        print(f"  {phase_name}: {phase.sublattices} (sum={sum(phase.sublattices)})")
    
    # CPU calculation
    print("\n" + "="*50)
    print("CPU CALCULATION")
    print("="*50)
    
    captured_data['cpu'] = {}
    
    result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
    cpu_gm = result_cpu.GM.values[0,0,0,0]
    cpu_phases = result_cpu.Phase.values[0,0,0,0]
    cpu_np = result_cpu.NP.values[0,0,0,0]
    
    print(f"\nCPU Final result:")
    print(f"  GM: {cpu_gm:.6f}")
    print("  Active phases:")
    for phase, amount in zip(cpu_phases, cpu_np):
        if phase != '' and amount > 1e-8:
            print(f"    {phase}: {amount:.6f}")
    
    # GPU calculation
    print("\n" + "="*50)
    print("GPU CALCULATION")
    print("="*50)
    
    # Reset capture for GPU
    captured_data['gpu'] = {}
    
    # GPU verbose output will show its iteration 0 info
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
    print("\n" + "="*50)
    print("ANALYSIS")
    print("="*50)
    
    gm_diff = abs(gpu_gm - cpu_gm)
    print(f"GM difference: {gm_diff:.2e}")
    
    # Check phase differences
    cpu_active_set = set(p for p, a in zip(cpu_phases, cpu_np) if p != '' and a > 1e-8)
    gpu_active_set = set(p for p, a in zip(gpu_phases, gpu_np) if p != '' and a > 1e-8)
    
    if 'HCP_A3' in gpu_active_set and 'HCP_A3' not in cpu_active_set:
        print("\n❌ GPU incorrectly has HCP_A3 active!")
        print("This must have come from the starting point calculation,")
        print("since the equilibrium matrix construction should be identical.")
        
        # Check captured CPU data
        if 'active_phases' in captured_data['cpu']:
            print(f"\nCPU iteration 0 had {captured_data['cpu']['num_active']} active phases:")
            for phase, amt in zip(captured_data['cpu']['active_phases'], captured_data['cpu']['phase_amounts']):
                print(f"  {phase}: {amt:.6f}")
    
    print("\nThe equilibrium matrix at iteration 0 should reveal if the GPU")
    print("is starting with HCP_A3 active, which would explain the final result.")

if __name__ == "__main__":
    test_6phase_matrix()