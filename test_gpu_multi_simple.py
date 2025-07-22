#!/usr/bin/env python
"""Simple test for GPU multi-condition handling."""

import numpy as np
from pycalphad import Database, equilibrium
import pycalphad.variables as v

# Load database
db = Database('NbTi.tdb')
components = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']

# Test with just 3 conditions
conditions = {
    v.T: 600,
    v.P: 101325,
    v.X('TI'): [0.1, 0.5, 0.9],  # 3 specific compositions
    v.N: 1
}

print("Testing GPU with 3 conditions...")
print("Compositions: X(TI) = [0.1, 0.5, 0.9]")
print("=" * 60)

# GPU calculation
try:
    gpu_result = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 50}, gpu=True)
    print(f"GPU calculation successful!")
    print(f"Result shape: {gpu_result.GM.shape}")
    print(f"GM values shape: {gpu_result.GM.values.shape}")
    
    # Extract results
    print("\nGPU Results:")
    x_ti_list = [0.1, 0.5, 0.9]
    for i in range(len(x_ti_list)):
        if i < gpu_result.GM.shape[-1]:
            x_ti = x_ti_list[i]
            gm = gpu_result.GM.values[0, 0, 0, i]
            mu_shape = gpu_result.MU.values.shape
            print(f"  MU shape: {mu_shape}")
            if len(mu_shape) >= 5 and mu_shape[4] >= 2:
                mu_nb = gpu_result.MU.values[0, 0, 0, i, 0]
                mu_ti = gpu_result.MU.values[0, 0, 0, i, 1]
                print(f"  X(TI)={x_ti}: GM={gm:.3f}, MU(NB)={mu_nb:.3f}, MU(TI)={mu_ti:.3f}")
            else:
                print(f"  X(TI)={x_ti}: GM={gm:.3f}, MU data has unexpected shape")
            
            # Check phase information
            if hasattr(gpu_result, 'Phase'):
                phases_at_point = gpu_result.Phase.values[0, 0, 0, i, :]
                print(f"    Phases: {phases_at_point}")
            
            if hasattr(gpu_result, 'NP'):
                np_at_point = gpu_result.NP.values[0, 0, 0, i, :]
                print(f"    Phase amounts: {np_at_point}")
                
except Exception as e:
    print(f"GPU calculation FAILED: {e}")
    import traceback
    traceback.print_exc()

# Also test CPU for comparison
print("\n" + "=" * 60)
print("Testing CPU with same 3 conditions...")
try:
    cpu_result = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 50})
    print(f"CPU calculation successful!")
    print(f"Result shape: {cpu_result.GM.shape}")
    
    print("\nCPU Results:")
    for i, x_ti in enumerate(x_ti_list):
        gm = cpu_result.GM.values[0, 0, 0, i]
        mu_nb = cpu_result.MU.values[0, 0, 0, i, 0]
        mu_ti = cpu_result.MU.values[0, 0, 0, i, 1]
        print(f"  X(TI)={x_ti}: GM={gm:.3f}, MU(NB)={mu_nb:.3f}, MU(TI)={mu_ti:.3f}")
        
except Exception as e:
    print(f"CPU calculation FAILED: {e}")
    import traceback
    traceback.print_exc()