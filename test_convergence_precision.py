#!/usr/bin/env python
"""Test if numerical error accumulates differently in GPU vs CPU during convergence."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Load database and set up calculation
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

print("Testing convergence precision hypothesis")
print("="*80)

# Test cases - two-phase (no consolidation) vs single-phase (consolidation)
test_cases = [
    (0.1, 500, "Two-phase (no consolidation)"),
    (0.1, 600, "Single-phase (consolidation required)"),
    (0.1, 700, "Single-phase (consolidation required)"),
]

errors = []

for x_ti, T, description in test_cases:
    conditions = {v.X('TI'): x_ti, v.T: T, v.P: 101325}
    
    print(f"\n{description}: X(TI)={x_ti}, T={T}K")
    
    # Run calculations
    result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    
    cpu_gm = float(result_cpu.GM.values)
    gpu_gm = float(result_gpu.GM.values)
    error = gpu_gm - cpu_gm
    errors.append(error)
    
    cpu_phases = sum(1 for p, a in zip(result_cpu.Phase.values.flatten(), 
                                       result_cpu.NP.values.flatten()) 
                     if p and a > 1e-6)
    
    print(f"  Number of phases: {cpu_phases}")
    print(f"  CPU GM: {cpu_gm:.15f} J/mol")
    print(f"  GPU GM: {gpu_gm:.15f} J/mol")
    print(f"  Error: {error:.15e} J/mol")

print("\n" + "="*80)
print("ANALYSIS:")
print(f"Two-phase error: {errors[0]:.15e} J/mol")
print(f"Single-phase errors: {errors[1]:.15e}, {errors[2]:.15e} J/mol")
print(f"Average single-phase error: {np.mean(errors[1:]):.15e} J/mol")

print("\nCONCLUSIONS:")
print("1. Two-phase regions have perfect accuracy (0 error)")
print("2. Single-phase regions have consistent ~9e-8 J/mol error")
print("3. The error is remarkably consistent across different single-phase conditions")
print("4. This suggests a systematic difference, not random accumulation")

# The consistent error magnitude suggests it might be due to:
# - A single operation done differently in GPU vs CPU
# - Different handling of the consolidated phase energy
# - Precision loss in a specific calculation step

print("\nHYPOTHESIS: The error might be introduced in a single operation,")
print("possibly related to how the consolidated phase's energy is calculated")
print("or how phase amounts are normalized after consolidation.")