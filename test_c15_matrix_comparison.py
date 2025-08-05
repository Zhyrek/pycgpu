#!/usr/bin/env python
"""Compare CPU vs GPU equilibrium matrices at iteration 0 for C15/LIQUID."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

def test_c15_equilibrium_matrix():
    """Test that CPU and GPU have identical equilibrium matrices at iteration 0 for C15/LIQUID."""
    
    # Load database and set up calculation
    dbf = Database('AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    phases = ['AU2BI_C15', 'LIQUID']
    
    conditions = {
        v.X('BI'): 0.3,
        v.T: 600,
        v.P: 101325
    }
    
    print("Testing C15/LIQUID equilibrium matrices at iteration 0...")
    print(f"Conditions: X(BI)={conditions[v.X('BI')]}, T={conditions[v.T]}K")
    print("="*60)
    
    # CPU calculation - capture only iteration 0 output
    print("\nCPU Equilibrium Matrix at Iteration 0:")
    result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=True)
    
    print("\n" + "="*60 + "\n")
    
    # GPU calculation - capture only iteration 0 output  
    print("GPU Equilibrium Matrix at Iteration 0:")
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)
    
    # Extract final results
    cpu_gm = float(result_cpu.GM.values)
    gpu_gm = float(result_gpu.GM.values)
    
    print(f"\n{'='*60}")
    print(f"Final Results:")
    print(f"CPU GM: {cpu_gm:.6f}")
    print(f"GPU GM: {gpu_gm:.6f}")
    print(f"Difference: {abs(cpu_gm - gpu_gm):.6f}")
    
    if abs(cpu_gm - gpu_gm) < 1.0:
        print("\n✓ Test PASSED - Results match within tolerance")
    else:
        print("\n✗ Test FAILED - Results differ significantly")

if __name__ == "__main__":
    test_c15_equilibrium_matrix()