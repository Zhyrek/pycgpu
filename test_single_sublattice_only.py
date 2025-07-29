#!/usr/bin/env python
"""Test GPU vs CPU with just LIQUID phase (single sublattice)."""

from pycalphad import Database, equilibrium, variables as v
import warnings

warnings.filterwarnings("ignore")

# Load database
dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']

# Test with just LIQUID (single sublattice)
phases = ['LIQUID']

print(f"Testing GPU vs CPU with just LIQUID phase (single sublattice)")
print()

# Simple test condition
conditions = {v.T: 900, v.P: 101325, v.N: 1, v.X('CU'): 0.3, v.X('FE'): 0.0}

print(f"Test conditions: X(AL)=0.70, X(CU)=0.30, X(FE)=0.00, T=900K")

try:
    # CPU calculation
    print("\nRunning CPU...")
    cpu_result = equilibrium(dbf, comps, phases, conditions, 
                           calc_opts={'pdens': 10}, verbose=False)
    cpu_gm = float(cpu_result.GM.values)
    
    print(f"CPU SUCCESS: GM={cpu_gm:.2f} J/mol")
    
except Exception as e:
    print(f"CPU FAILED: {type(e).__name__}: {str(e)}")
    cpu_gm = None

try:
    # GPU calculation
    print("\nRunning GPU...")
    gpu_result = equilibrium(dbf, comps, phases, conditions, 
                           calc_opts={'pdens': 10}, verbose=False, gpu=True)
    gpu_gm = float(gpu_result.GM.values)
    
    print(f"GPU SUCCESS: GM={gpu_gm:.2f} J/mol")
    
    # Compare
    if cpu_gm is not None:
        diff = abs(gpu_gm - cpu_gm)
        print(f"\nDifference: {diff:.6f} J/mol - {'PASS' if diff < 1.0 else 'FAIL'}")
    
except Exception as e:
    print(f"GPU FAILED: {type(e).__name__}: {str(e)}")