#!/usr/bin/env python
"""Test GPU vs CPU for NbTi system."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

def test_nbti():
    """Test NbTi system."""
    
    # Load database and set up calculation
    dbf = Database('NbTi.tdb')
    comps = ['NB', 'TI']
    phases = ['LIQUID', 'BCC_A2']
    
    conditions = {
        v.X('TI'): (0.1, 0.9, 0.2),  # 0.1 to 0.9 in 0.2 increments
        v.T: (1000, 2000, 250),       # 1000 to 2000 in 250 increments
        v.P: 101325
    }
    
    print("Testing NbTi system (LIQUID + BCC_A2 phases)")
    print("Multiple conditions test...")
    
    # CPU calculation
    print("\nRunning CPU calculation...")
    result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False)
    cpu_gm = result_cpu.GM.values
    
    # GPU calculation
    print("Running GPU calculation...")
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True)
    gpu_gm = result_gpu.GM.values
    
    # Flatten and compare
    cpu_gm_flat = cpu_gm.flatten()
    gpu_gm_flat = gpu_gm.flatten()
    
    # Count passing conditions
    passed = 0
    total = len(cpu_gm_flat)
    
    for i in range(total):
        if not (np.isnan(cpu_gm_flat[i]) or np.isnan(gpu_gm_flat[i])):
            diff = abs(cpu_gm_flat[i] - gpu_gm_flat[i])
            if diff < 1.0:  # 1 J/mol tolerance
                passed += 1
    
    print(f"\nResults:")
    print(f"  Total conditions: {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {total - passed}")
    print(f"  Pass rate: {passed/total*100:.1f}%")
    
    # Show some examples
    print(f"\nExample comparisons:")
    for i in range(min(5, total)):
        if not (np.isnan(cpu_gm_flat[i]) or np.isnan(gpu_gm_flat[i])):
            diff = abs(cpu_gm_flat[i] - gpu_gm_flat[i])
            print(f"  Condition {i}: CPU={cpu_gm_flat[i]:.2f}, GPU={gpu_gm_flat[i]:.2f}, Diff={diff:.2f}")

if __name__ == "__main__":
    test_nbti()