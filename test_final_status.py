#!/usr/bin/env python
"""Final status test - minimal output."""

import os
os.environ['PYCALPHAD_DEBUG'] = '0'

# Suppress all warnings and output
import warnings
warnings.filterwarnings('ignore')

from pycalphad import Database, equilibrium
from contextlib import redirect_stdout, redirect_stderr
import io

# Load database silently
with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    tdb = Database('NbTi.tdb')
    
comps = ['NB', 'TI', 'VA']
phases = ['LIQUID', 'BCC_A2']

# Test conditions
test_cases = [
    {'T': 1000, 'X(TI)': 0.01},  # Previously divergent case
    {'T': 1500, 'X(TI)': 0.5},   # Middle case
    {'T': 2000, 'X(TI)': 0.9},   # High Ti case
]

print("Testing GPU vs CPU match after consolidation fix...")
print("="*50)

failed = 0
tolerance = 1e-6

for conditions in test_cases:
    conditions['P'] = 101325
    
    try:
        # Redirect all output during calculations
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            eq_cpu = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=False)
            cpu_gm = float(eq_cpu.GM.values.item())
            
            eq_gpu = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=False)
            gpu_gm = float(eq_gpu.GM.values.item())
        
        diff = abs(cpu_gm - gpu_gm)
        
        if diff > tolerance:
            print(f"✗ T={conditions['T']}K, X(TI)={conditions['X(TI)']}: Diff={diff:.2e} J/mol")
            failed += 1
        else:
            print(f"✓ T={conditions['T']}K, X(TI)={conditions['X(TI)']}: PASS")
            
    except Exception as e:
        print(f"ERROR at T={conditions['T']}K, X(TI)={conditions['X(TI)']}: {str(e)}")
        failed += 1

print("="*50)
if failed == 0:
    print("✓ ALL TESTS PASSED!")
else:
    print(f"✗ {failed}/{len(test_cases)} tests failed")