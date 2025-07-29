#!/usr/bin/env python
"""Quick test to see if dgelsd_device compiles and improves accuracy."""

import os
os.environ['PYCALPHAD_DEBUG_EQSOLVER'] = ''  # Disable debug output

from pycalphad import Database, equilibrium, variables as v
import numpy as np

# Load database
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'HCP_A3']

# Test a single condition
conditions = {v.T: 500, v.P: 101325, v.N: 1, v.X('TI'): 0.1}

print("Testing GPU with dgelsd_device implementation...")

try:
    # CPU calculation
    cpu_result = equilibrium(dbf, comps, phases, conditions, 
                           calc_opts={'pdens': 100}, verbose=False)
    cpu_gm = float(cpu_result.GM.values)
    print(f"CPU GM: {cpu_gm:.10f} J/mol")
    
    # GPU calculation  
    gpu_result = equilibrium(dbf, comps, phases, conditions, 
                           calc_opts={'pdens': 100}, verbose=False, gpu=True)
    gpu_gm = float(gpu_result.GM.values)
    print(f"GPU GM: {gpu_gm:.10f} J/mol")
    
    # Compare
    diff = abs(cpu_gm - gpu_gm)
    print(f"\nDifference: {diff:.10f} J/mol")
    print(f"Relative error: {diff/abs(cpu_gm)*100:.6f}%")
    
    if diff < 0.001:
        print("\n✓ EXCELLENT! Error < 0.001 J/mol")
    elif diff < 0.01:
        print("\n✓ Good! Error < 0.01 J/mol")
    else:
        print(f"\n✗ Error still large: {diff:.6f} J/mol")
        
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()