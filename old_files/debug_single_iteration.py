#!/usr/bin/env python
"""
Compare a single iteration of CPU vs GPU equilibrium solver.
"""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings('ignore')

def debug_single_iteration():
    """Compare exactly one iteration of solver between CPU and GPU"""
    print("="*60)
    print("SINGLE ITERATION DEBUG: CPU vs GPU")
    print("="*60)
    
    # Use the same condition
    dbf = Database("NbTi.tdb")
    conditions = {v.X("TI"): 1e-10, v.T: 600, v.P: 101325}
    
    print("Condition: T=600K, X_TI=1e-10 (pure NB system)")
    print("Expected: Single BCC_A2 phase at equilibrium")
    
    # Run CPU equilibrium
    print(f"\n1. CPU EQUILIBRIUM RESULT:")
    cpu_result = equilibrium(dbf, ['NB', 'TI', 'VA'], ['LIQUID', 'BCC_A2'], 
                           conditions, gpu=False, verbose=False)
    
    cpu_gm = float(cpu_result.GM.values[0,0,0,0])
    cpu_mu = cpu_result.MU.values[0,0,0,0,:]
    cpu_np = cpu_result.NP.values[0,0,0,0,:]
    cpu_phases = cpu_result.Phase.values[0,0,0,0,:]
    
    print(f"  GM: {cpu_gm:.6f}")
    print(f"  MU: {cpu_mu}")
    print(f"  NP: {cpu_np}")
    print(f"  Phases: {cpu_phases}")
    
    # Get active phases
    cpu_active_mask = cpu_np > 1e-10
    cpu_active_phases = cpu_phases[cpu_active_mask]
    cpu_active_amounts = cpu_np[cpu_active_mask]
    print(f"  Active: {cpu_active_phases} with amounts {cpu_active_amounts}")
    
    # Run GPU equilibrium with limited iterations
    print(f"\n2. GPU EQUILIBRIUM RESULT:")
    
    try:
        gpu_result = equilibrium(dbf, ['NB', 'TI', 'VA'], ['LIQUID', 'BCC_A2'], 
                               conditions, gpu=True, verbose=False)
        
        gpu_gm = float(gpu_result.GM.values[0,0,0,0])
        gpu_mu = gpu_result.MU.values[0,0,0,0,:]
        gpu_np = gpu_result.NP.values[0,0,0,0,:]
        gpu_phases = gpu_result.Phase.values[0,0,0,0,:]
        
        print(f"  GM: {gpu_gm:.6f}")
        print(f"  MU: {gpu_mu}")
        print(f"  NP: {gpu_np}")
        print(f"  Phases: {gpu_phases}")
        
        gpu_active_mask = gpu_np > 1e-10
        gpu_active_phases = gpu_phases[gpu_active_mask]
        gpu_active_amounts = gpu_np[gpu_active_mask]
        print(f"  Active: {gpu_active_phases} with amounts {gpu_active_amounts}")
        
    except Exception as e:
        print(f"GPU failed: {e}")
        return
    
    # Compare results
    print(f"\n3. COMPARISON:")
    print(f"  GM difference: {abs(cpu_gm - gpu_gm):.10f}")
    print(f"  MU difference: {np.max(np.abs(cpu_mu[:2] - gpu_mu[:2])):.10f}")
    
    if abs(cpu_gm - gpu_gm) < 1e-6:
        print(f"  ✅ Results match within tolerance")
    else:
        print(f"  ❌ Results do not match")
        print(f"  Expected GM: {cpu_gm:.10f}")
        print(f"  Actual GM:   {gpu_gm:.10f}")
        
        # Analyze the starting points
        print(f"\n4. STARTING POINT ANALYSIS:")
        print(f"  The GPU should start with the same values as CPU final result")
        print(f"  since starting_point should give the equilibrium state.")
        print(f"  If GPU result differs, the solver is not converging correctly.")

def main():
    debug_single_iteration()

if __name__ == "__main__":
    main()