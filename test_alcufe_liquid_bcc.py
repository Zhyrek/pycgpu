#!/usr/bin/env python
"""Test LIQUID and BCC_A2 phases in Al-Cu-Fe ternary space."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

# Load Al-Cu-Fe database
dbf = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']
phases = ['LIQUID', 'BCC_A2']

print("Testing LIQUID and BCC_A2 in Al-Cu-Fe system")
print("=" * 60)

# Test different compositions
test_conditions = [
    {'T': 1000, 'P': 101325, 'X(AL)': 0.33, 'X(CU)': 0.33},  # Equimolar
    {'T': 1000, 'P': 101325, 'X(AL)': 0.5, 'X(CU)': 0.3},   # Al-rich
    {'T': 1000, 'P': 101325, 'X(AL)': 0.2, 'X(CU)': 0.5},   # Cu-rich
    {'T': 1000, 'P': 101325, 'X(AL)': 0.1, 'X(CU)': 0.1},   # Fe-rich
    {'T': 1500, 'P': 101325, 'X(AL)': 0.33, 'X(CU)': 0.33},  # Higher T
]

for i, conds in enumerate(test_conditions):
    print(f"\nTest {i+1}: T={conds['T']}K, X(AL)={conds.get('X(AL)', 0):.2f}, X(CU)={conds.get('X(CU)', 0):.2f}")
    
    try:
        # CPU calculation
        eq_cpu = equilibrium(dbf, components, phases, conds, verbose=False)
        cpu_gm = float(eq_cpu.GM.values.flatten()[0])
        print(f"  CPU GM: {cpu_gm:.2f} J/mol")
        
        # GPU calculation
        eq_gpu = equilibrium(dbf, components, phases, conds, verbose=False, gpu=True)
        gpu_gm = float(eq_gpu.GM.values.flatten()[0])
        print(f"  GPU GM: {gpu_gm:.2f} J/mol")
        
        diff = abs(gpu_gm - cpu_gm)
        print(f"  Difference: {diff:.2f} J/mol")
        
        if gpu_gm < -700 and gpu_gm > -800:
            print("  WARNING: GPU returned error code -777.0")
        
    except Exception as e:
        print(f"  ERROR: {str(e)}")

# Test with multiple conditions at once
print("\n" + "=" * 60)
print("Testing multiple conditions simultaneously:")

multi_conds = {
    v.T: [1000, 1500],
    v.P: 101325,
    v.X('AL'): 0.33,
    v.X('CU'): 0.33
}

try:
    # CPU calculation
    eq_cpu = equilibrium(dbf, components, phases, multi_conds, verbose=False)
    cpu_gm_values = eq_cpu.GM.values.flatten()
    
    # GPU calculation
    eq_gpu = equilibrium(dbf, components, phases, multi_conds, verbose=False, gpu=True)
    gpu_gm_values = eq_gpu.GM.values.flatten()
    
    print(f"\nResults for 2 conditions:")
    for j, (T, cpu_gm, gpu_gm) in enumerate(zip([1000, 1500], cpu_gm_values, gpu_gm_values)):
        diff = abs(gpu_gm - cpu_gm)
        print(f"  T={T}K: CPU={cpu_gm:.2f}, GPU={gpu_gm:.2f}, Diff={diff:.2f} J/mol")
        if gpu_gm < -700 and gpu_gm > -800:
            print(f"    WARNING: GPU returned error code -777.0")
            
except Exception as e:
    print(f"  ERROR: {str(e)}")