#!/usr/bin/env python
"""Simple summary of GPU vs CPU match status."""

import os
os.environ['PYCALPHAD_DEBUG'] = '0'

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

from pycalphad import Database, equilibrium
from pycalphad.core.debug_output import reset_debug_session
import sys

# Suppress all output
old_stdout = sys.stdout
old_stderr = sys.stderr
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

try:
    # Load database
    tdb = Database('NbTi.tdb')
    comps = ['NB', 'TI', 'VA']
    phases = ['LIQUID', 'BCC_A2']
    
    # Test specific case that was previously divergent
    conditions = {'T': 1000, 'P': 101325, 'X(TI)': 0.01}
    
    # Reset debug
    reset_debug_session()
    
    # Run calculations
    eq_cpu = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=False)
    cpu_gm = eq_cpu.GM.values.item()
    
    eq_gpu = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=False)
    gpu_gm = eq_gpu.GM.values.item()
    
finally:
    # Restore output
    sys.stdout = old_stdout
    sys.stderr = old_stderr

# Show results
print("GPU vs CPU Test Result")
print("="*40)
print(f"Test case: T=1000K, X(TI)=0.01")
print(f"CPU GM: {cpu_gm:.6f} J/mol")
print(f"GPU GM: {gpu_gm:.6f} J/mol")
print(f"Difference: {abs(cpu_gm - gpu_gm):.2e} J/mol")
print(f"Tolerance: 1e-6 J/mol")

if abs(cpu_gm - gpu_gm) < 1e-6:
    print("\n✓ TEST PASSED - GPU matches CPU!")
    print("\nThe phase consolidation fix successfully resolved")
    print("the divergence for this test case.")
else:
    print("\n✗ TEST FAILED - GPU does not match CPU")
    print(f"Difference of {abs(cpu_gm - gpu_gm):.2e} exceeds tolerance")