#!/usr/bin/env python3
import numpy as np
from pycalphad import Database, equilibrium, variables as v

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']

# Set conditions
eq_conditions = {v.N: 1, v.P: 101325, v.T: 1000, v.X('TI'): 0.4}

print("Testing GPU equilibrium calculation (quiet mode)...")
try:
    # Run GPU calculation
    gpu_result = equilibrium(db, comps, phases, eq_conditions, verbose=False, gpu=True, to='GM', calc_opts={'pdens': 50})
    gpu_gm = float(gpu_result.GM.values)
    print(f"\nGPU GM: {gpu_gm:.6f} J/mol")
    
    # Run CPU calculation for comparison
    cpu_result = equilibrium(db, comps, phases, eq_conditions, verbose=False, gpu=False, to='GM', calc_opts={'pdens': 50})
    cpu_gm = float(cpu_result.GM.values)
    print(f"CPU GM: {cpu_gm:.6f} J/mol")
    
    # Compare results
    diff = abs(gpu_gm - cpu_gm)
    print(f"\nDifference: {diff:.6f} J/mol")
    print(f"Match within 0.001 J tolerance: {diff < 0.001}")
    
    if diff < 0.001:
        print("\n✅ SUCCESS: GPU and CPU results match within tolerance!")
    else:
        print(f"\n❌ FAIL: Difference {diff:.6f} J/mol exceeds 0.001 J tolerance")
    
except Exception as e:
    print(f"\n❌ GPU failed with error: {type(e).__name__}: {str(e)}")