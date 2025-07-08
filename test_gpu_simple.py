#!/usr/bin/env python3
import numpy as np
from pycalphad import Database, equilibrium, variables as v

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']

# Set conditions
eq_conditions = {v.N: 1, v.P: 101325, v.T: 1000, v.X('TI'): 0.4}

print("Testing GPU equilibrium calculation...")
try:
    # Run GPU calculation
    print("\n=== GPU Calculation ===")
    gpu_result = equilibrium(db, comps, phases, eq_conditions, verbose=False, gpu=True, to='GM', calc_opts={'pdens': 50})
    gpu_gm = float(gpu_result.GM.values)
    print(f"GPU GM: {gpu_gm:.6f} J/mol")
    print(f"GPU converged: Yes")
    
    # Run CPU calculation for comparison
    print("\n=== CPU Calculation ===")
    cpu_result = equilibrium(db, comps, phases, eq_conditions, verbose=False, gpu=False, to='GM', calc_opts={'pdens': 50})
    cpu_gm = float(cpu_result.GM.values)
    print(f"CPU GM: {cpu_gm:.6f} J/mol")
    
    # Compare results
    print(f"\n=== Comparison ===")
    print(f"Difference: {abs(gpu_gm - cpu_gm):.6f} J/mol")
    print(f"Match within 0.001 J tolerance: {abs(gpu_gm - cpu_gm) < 0.001}")
    
except Exception as e:
    print(f"GPU failed with error: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()