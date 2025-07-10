#!/usr/bin/env python3
"""Final clean test of GPU vs CPU equilibrium"""
import os
import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.gpu.gpu_equilibrium import clear_gpu_cache

# Clear any cached GPU modules
clear_gpu_cache()

# Also try to clear any compiled cache files
os.system("rm -f *.ptx *.cubin 2>/dev/null")

# Disable debug output for clean comparison
os.environ['PYCALPHAD_DEBUG'] = '0'

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']

# Set conditions
conditions = {v.N: 1, v.P: 101325, v.T: 1000, v.X('TI'): 0.4}

print("=== Final GPU vs CPU Equilibrium Test ===")
print(f"Conditions: {conditions}")
print()

# Run CPU calculation
print("Running CPU calculation...")
cpu_result = equilibrium(db, comps, phases, conditions, verbose=False, gpu=False, calc_opts={'pdens': 50})
cpu_gm = float(cpu_result.GM.values.flatten()[0])
cpu_phases = [p for p in cpu_result.Phase.values.flatten() if p != '']
print(f"CPU GM: {cpu_gm:.6f} J/mol")
print(f"CPU phases: {cpu_phases}")

# Run GPU calculation
print("\nRunning GPU calculation...")
try:
    gpu_result = equilibrium(db, comps, phases, conditions, verbose=False, gpu=True, calc_opts={'pdens': 50})
    gpu_gm = float(gpu_result.GM.values.flatten()[0])
    gpu_phases = [p for p in gpu_result.Phase.values.flatten() if p != '']
    print(f"GPU GM: {gpu_gm:.6f} J/mol")
    print(f"GPU phases: {gpu_phases}")
    
    # Compare results
    print(f"\n=== Comparison ===")
    print(f"GM difference: {abs(gpu_gm - cpu_gm):.6f} J/mol")
    print(f"Match within 0.001 J tolerance: {abs(gpu_gm - cpu_gm) < 0.001}")
    
    if abs(gpu_gm - cpu_gm) > 0.001:
        print("\n⚠️  GPU and CPU results do not match!")
        print("The issue with mass jacobian calculation needs to be fixed.")
    else:
        print("\n✅ GPU and CPU results match!")
        
except Exception as e:
    print(f"\nGPU calculation failed: {e}")
    import traceback
    traceback.print_exc()