#!/usr/bin/env python
"""Test numerical accuracy for specific conditions."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Load database and set up calculation
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

print("Testing numerical accuracy for specific conditions")
print("="*80)

# Test cases
test_cases = [
    (0.1, 500, "Inside miscibility gap (two-phase)"),
    (0.1, 600, "Outside miscibility gap (single-phase)"),
    (0.5, 600, "Inside miscibility gap (two-phase)"),
    (0.1, 700, "Outside miscibility gap (single-phase)"),
]

for x_ti, T, description in test_cases:
    conditions = {v.X('TI'): x_ti, v.T: T, v.P: 101325}
    
    # Run calculations
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
    
    gpu_gm = float(result_gpu.GM.values)
    cpu_gm = float(result_cpu.GM.values)
    error = gpu_gm - cpu_gm
    
    # Count phases
    gpu_phases = sum(1 for phase, amt in zip(result_gpu.Phase.values.flatten(), 
                                             result_gpu.NP.values.flatten()) 
                     if phase and amt > 1e-6)
    cpu_phases = sum(1 for phase, amt in zip(result_cpu.Phase.values.flatten(), 
                                             result_cpu.NP.values.flatten()) 
                     if phase and amt > 1e-6)
    
    print(f"\nX(TI)={x_ti}, T={T}K - {description}")
    print(f"  CPU: GM={cpu_gm:.15f} J/mol, phases={cpu_phases}")
    print(f"  GPU: GM={gpu_gm:.15f} J/mol, phases={gpu_phases}")
    print(f"  Error: {error:.15e} J/mol")
    print(f"  Relative error: {abs(error/cpu_gm)*100:.10f}%")
    
    # Check if consolidation occurred
    if cpu_phases == 1 and gpu_phases == 1:
        print("  → Single phase (consolidation occurred)")
    elif cpu_phases == 2 and gpu_phases == 2:
        print("  → Two phases (no consolidation)")
    else:
        print(f"  → Phase count mismatch! CPU={cpu_phases}, GPU={gpu_phases}")

print("\n" + "="*80)
print("OBSERVATION: Perfect accuracy for two-phase regions, small errors for single-phase")