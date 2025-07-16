#!/usr/bin/env python
"""Test to identify which specific conditions fail."""

import os
import sys
import warnings
warnings.filterwarnings('ignore')
os.environ['PYCALPHAD_DEBUG'] = '0'

from pycalphad import Database, equilibrium
import numpy as np

# Suppress output
class SuppressOutput:
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        return self
        
    def __exit__(self, *args):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr

# Load TDB
with SuppressOutput():
    tdb = Database('NbTi.tdb')
    
comps = ['NB', 'TI', 'VA']
phases = ['LIQUID', 'BCC_A2']

# Test conditions
test_conditions = [
    # Previously divergent case (should now pass)
    {'T': 1000, 'X(TI)': 0.01},
    
    # Other key cases
    {'T': 500, 'X(TI)': 0.001},
    {'T': 500, 'X(TI)': 0.5},
    {'T': 1000, 'X(TI)': 0.5},
    {'T': 1500, 'X(TI)': 0.5},
    {'T': 2000, 'X(TI)': 0.5},
]

print("Checking specific test cases...")
print("="*60)

tolerance = 1e-6
failed_cases = []

for conditions in test_conditions:
    conditions['P'] = 101325
    
    with SuppressOutput():
        try:
            # CPU calculation
            eq_cpu = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=False)
            cpu_gm = float(eq_cpu.GM.values.item())
            
            # GPU calculation
            eq_gpu = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=False)
            gpu_gm = float(eq_gpu.GM.values.item())
            
            diff = abs(cpu_gm - gpu_gm)
            
            if diff > tolerance:
                failed_cases.append({
                    'T': conditions['T'],
                    'X(TI)': conditions['X(TI)'],
                    'cpu_gm': cpu_gm,
                    'gpu_gm': gpu_gm,
                    'diff': diff,
                    'cpu_phases': eq_cpu.Phase.values.flatten(),
                    'gpu_phases': eq_gpu.Phase.values.flatten(),
                    'cpu_np': eq_cpu.NP.values.flatten(),
                    'gpu_np': eq_gpu.NP.values.flatten()
                })
                
        except Exception as e:
            print(f"ERROR at T={conditions['T']}K, X(TI)={conditions['X(TI)']}: {str(e)}")

if failed_cases:
    print(f"\nFAILED CASES ({len(failed_cases)} total):")
    print("-"*60)
    for fc in failed_cases:
        print(f"\nT={fc['T']}K, X(TI)={fc['X(TI)']:.3f}:")
        print(f"  CPU GM: {fc['cpu_gm']:.1f} J/mol")
        print(f"  GPU GM: {fc['gpu_gm']:.1f} J/mol")
        print(f"  Difference: {fc['diff']:.1e} J/mol")
        
        # Show phase information
        print("  CPU phases:", end='')
        for phase, amount in zip(fc['cpu_phases'], fc['cpu_np']):
            if amount > 1e-6:
                print(f" {phase}({amount:.3f})", end='')
        print()
        
        print("  GPU phases:", end='')
        for phase, amount in zip(fc['gpu_phases'], fc['gpu_np']):
            if amount > 1e-6:
                print(f" {phase}({amount:.3f})", end='')
        print()
else:
    print("\n✓ ALL TEST CASES PASSED!")