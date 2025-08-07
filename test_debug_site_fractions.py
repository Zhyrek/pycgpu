#!/usr/bin/env python
"""Debug test to trace site fractions issue."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

# Load database
dbf = Database('AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']

# Test with 4 phases
phases = ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'RHOMBOHEDRAL_A7']
print(f"Testing with 4 phases: {phases}")

# Single condition
conditions = {
    v.X('BI'): 0.3,
    v.T: 600,
    v.P: 101325
}

try:
    # CPU calculation
    print("\nRunning CPU calculation...")
    result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
    cpu_gm = float(result_cpu.GM.values)
    print(f"CPU GM: {cpu_gm:.2f} J/mol")
    
    # GPU calculation with verbose
    print("\nRunning GPU calculation with verbose=True...")
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)
    gpu_gm = float(result_gpu.GM.values)
    print(f"\nGPU GM: {gpu_gm:.2f} J/mol")
    print(f"Difference: {abs(cpu_gm - gpu_gm):.2f} J/mol")
    
except Exception as e:
    print(f"\nERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()