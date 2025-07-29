#!/usr/bin/env python
"""Test GPU equilibrium calculation with LIQUID and ALCU_ZETA phases - simple version."""

from pycalphad import Database, equilibrium
import pycalphad.variables as v
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load database
db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']
phases = ['LIQUID', 'ALCU_ZETA']

# Simple test conditions
conditions = {
    v.T: 1273.15,  # 1000°C
    v.P: 101325,
    v.X('AL'): 0.3,
    v.X('CU'): 0.3,
    v.N: 1
}

print("Testing GPU equilibrium with LIQUID and ALCU_ZETA phases")
print("="*60)
print(f"Temperature: {conditions[v.T] - 273.15}°C")
print(f"Phases: {phases}")
print(f"Composition: X(AL)={conditions[v.X('AL')]}, X(CU)={conditions[v.X('CU')]}")

# Test CPU
print("\nTesting CPU...")
try:
    cpu_result = equilibrium(db, components, phases, conditions, 
                           calc_opts={'pdens': 10}, 
                           verbose=False)
    print("✓ CPU SUCCESS")
    print(f"  GM = {cpu_result.GM.values[0]:.1f} J/mol")
    for phase in phases:
        np_val = cpu_result.NP.sel(phase=phase).values[0]
        if np_val > 1e-10:
            print(f"  {phase}: {np_val:.4f}")
except Exception as e:
    print(f"✗ CPU FAILED: {type(e).__name__}")

# Test GPU
print("\nTesting GPU...")
try:
    gpu_result = equilibrium(db, components, phases, conditions, 
                           calc_opts={'pdens': 10}, 
                           gpu=True,
                           verbose=False)
    print("✓ GPU SUCCESS")
    print(f"  GM = {gpu_result.GM.values[0]:.1f} J/mol")
    for phase in phases:
        np_val = gpu_result.NP.sel(phase=phase).values[0]
        if np_val > 1e-10:
            print(f"  {phase}: {np_val:.4f}")
            
    # Compare if both succeeded
    if 'cpu_result' in locals():
        diff = abs(cpu_result.GM.values[0] - gpu_result.GM.values[0])
        print(f"\nDifference: {diff:.6f} J/mol")
        
except Exception as e:
    print(f"✗ GPU FAILED: {type(e).__name__}")
    if 'nvcc' in str(e):
        print("  (nvcc compilation error)")
    else:
        print(f"  Error: {str(e)[:100]}...")