#!/usr/bin/env python
"""Check equilibrium matrix at iteration 0 for CPU vs GPU."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

def test_equilibrium_matrices():
    """Test that CPU and GPU have identical equilibrium matrices at iteration 0."""
    
    # Load database and set up calculation
    dbf = Database('AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    phases = ['FCC_A1', 'LIQUID']
    
    conditions = {
        v.X('BI'): 0.3,
        v.T: 600,
        v.P: 101325
    }
    
    print("Testing equilibrium matrices at iteration 0...")
    print(f"Conditions: X(BI)={conditions[v.X('BI')]}, T={conditions[v.T]}K")
    
    # CPU calculation
    print("\nCPU Equilibrium Matrix at Iteration 0:")
    result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=True)
    
    print("\n" + "="*60 + "\n")
    
    # GPU calculation
    print("GPU Equilibrium Matrix at Iteration 0:")
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)
    
    # Extract results
    cpu_gm = float(result_cpu.GM.values)
    gpu_gm = float(result_gpu.GM.values)
    
    print(f"\nFinal Results:")
    print(f"CPU GM: {cpu_gm:.6f}")
    print(f"GPU GM: {gpu_gm:.6f}")
    print(f"Difference: {abs(cpu_gm - gpu_gm):.6f}")

if __name__ == "__main__":
    test_equilibrium_matrices()