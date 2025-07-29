#!/usr/bin/env python
"""Compare GPU and CPU multi-condition results."""

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

print("Testing with 3 conditions at T=600K")
print("Compositions: X(TI) = [0.1, 0.5, 0.9]")
print("=" * 60)

# GPU calculation
print("\nGPU Calculation:")
try:
    gpu_result = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 50}, gpu=True, verbose=False)
    print("Success!")
    
    x_ti_list = [0.1, 0.5, 0.9]
    for i, x_ti in enumerate(x_ti_list):
        gm = gpu_result.GM.values[0, 0, 0, i]
        mu_nb = gpu_result.MU.values[0, 0, 0, i, 0]
        mu_ti = gpu_result.MU.values[0, 0, 0, i, 1]
        np_vals = gpu_result.NP.values[0, 0, 0, i, :]
        phases_present = [p for j, p in enumerate(gpu_result.Phase.values[0, 0, 0, i, :]) if np_vals[j] > 1e-12]
        phase_amounts = [np_vals[j] for j in range(len(np_vals)) if np_vals[j] > 1e-12]
        
        print(f"  X(TI)={x_ti}: GM={gm:.1f}, MU(NB)={mu_nb:.1f}, MU(TI)={mu_ti:.1f}")
        print(f"    Phases: {phases_present}, amounts: {[f'{a:.3f}' for a in phase_amounts]}")
except Exception as e:
    print(f"FAILED: {e}")

# CPU calculation
print("\nCPU Calculation:")
try:
    cpu_result = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 50})
    print("Success!")
    
    for i, x_ti in enumerate(x_ti_list):
        gm = cpu_result.GM.values[0, 0, 0, i]
        mu_nb = cpu_result.MU.values[0, 0, 0, i, 0]
        mu_ti = cpu_result.MU.values[0, 0, 0, i, 1]
        np_vals = cpu_result.NP.values[0, 0, 0, i, :]
        phases_present = [p for j, p in enumerate(cpu_result.Phase.values[0, 0, 0, i, :]) if np_vals[j] > 1e-12]
        phase_amounts = [np_vals[j] for j in range(len(np_vals)) if np_vals[j] > 1e-12]
        
        print(f"  X(TI)={x_ti}: GM={gm:.1f}, MU(NB)={mu_nb:.1f}, MU(TI)={mu_ti:.1f}")
        print(f"    Phases: {phases_present}, amounts: {[f'{a:.3f}' for a in phase_amounts]}")
except Exception as e:
    print(f"FAILED: {e}")

# Compare results
print("\n" + "=" * 60)
print("Comparison (GPU - CPU):")
try:
    for i, x_ti in enumerate(x_ti_list):
        gm_diff = gpu_result.GM.values[0, 0, 0, i] - cpu_result.GM.values[0, 0, 0, i]
        mu_nb_diff = gpu_result.MU.values[0, 0, 0, i, 0] - cpu_result.MU.values[0, 0, 0, i, 0]
        mu_ti_diff = gpu_result.MU.values[0, 0, 0, i, 1] - cpu_result.MU.values[0, 0, 0, i, 1]
        
        print(f"  X(TI)={x_ti}: ΔGM={gm_diff:.1f}, ΔMU(NB)={mu_nb_diff:.1f}, ΔMU(TI)={mu_ti_diff:.1f}")
        
        # Check if differences are within tolerance
        if abs(gm_diff) > 1.0 or abs(mu_nb_diff) > 1.0 or abs(mu_ti_diff) > 1.0:
            print(f"    WARNING: Large difference detected!")
except:
    print("  Cannot compare - one calculation failed")

print("\nTest complete!")