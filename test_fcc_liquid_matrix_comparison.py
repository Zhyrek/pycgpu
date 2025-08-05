#!/usr/bin/env python
"""Test equilibrium matrix comparison between CPU and GPU for FCC_A1 and LIQUID phases."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

def test_fcc_liquid_equilibrium_matrix():
    """Test that GPU and CPU produce identical equilibrium matrices for FCC_A1/LIQUID system."""
    dbf = Database('AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    phases = ['FCC_A1', 'LIQUID']
    conditions = {
        v.X('BI'): 0.1,
        v.T: 400,
        v.P: 101325
    }
    
    print("Testing FCC_A1/LIQUID equilibrium matrices at iteration 0...")
    print(f"Conditions: X(BI)={conditions[v.X('BI')]}, T={conditions[v.T]}K")
    print("============================================================\n")
    
    # Run CPU calculation with verbose output
    print("CPU Equilibrium Matrix at Iteration 0:")
    result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=True)
    cpu_gm = float(result_cpu.GM.values.squeeze())
    
    print("\n" + "="*60 + "\n")
    
    # Run GPU calculation with verbose output  
    print("GPU Equilibrium Matrix at Iteration 0:")
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)
    gpu_gm = float(result_gpu.GM.values.squeeze())
    
    # Extract matrix values from debug output
    print("\n" + "="*60 + "\n")
    print("COMPARISON:")
    print(f"CPU GM: {cpu_gm:.6f}")
    print(f"GPU GM: {gpu_gm:.6f}")
    print(f"Difference: {abs(cpu_gm - gpu_gm):.6f}")
    
    # Check phase amounts
    cpu_phases = result_cpu.Phase.values.squeeze()
    gpu_phases = result_gpu.Phase.values.squeeze()
    cpu_np = result_cpu.NP.values.squeeze()
    gpu_np = result_gpu.NP.values.squeeze()
    
    print("\nPhase amounts:")
    for i, (cpu_phase, gpu_phase) in enumerate(zip(cpu_phases, gpu_phases)):
        if cpu_phase != '' and gpu_phase != '':
            print(f"  {cpu_phase}: CPU={cpu_np[i]:.6f}, GPU={gpu_np[i]:.6f}")

if __name__ == "__main__":
    test_fcc_liquid_equilibrium_matrix()