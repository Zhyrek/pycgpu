#!/usr/bin/env python
"""Clean summary test."""

import os
import sys

# Disable all debug output
os.environ['PYCALPHAD_DEBUG'] = '0'

# Redirect stderr to devnull before imports
original_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

try:
    from pycalphad import Database, equilibrium
    import numpy as np
    import warnings
    warnings.filterwarnings('ignore')
    
    # Restore stderr for our output
    sys.stderr.close()
    sys.stderr = original_stderr
    
    # Load database
    tdb = Database('NbTi.tdb')
    comps = ['NB', 'TI', 'VA']
    phases = ['LIQUID', 'BCC_A2']
    
    # Test broad range
    temperatures = [500, 1000, 1500, 2000, 2500]
    ti_fractions = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99]
    
    total = 0
    passed = 0
    max_diff = 0.0
    
    # Capture all output
    from io import StringIO
    import contextlib
    
    for T in temperatures:
        for x_ti in ti_fractions:
            total += 1
            conditions = {'T': T, 'P': 101325, 'X(TI)': x_ti}
            
            # Suppress all output during calculation
            f = StringIO()
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                try:
                    eq_cpu = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=False)
                    cpu_gm = float(eq_cpu.GM.values.item())
                    
                    eq_gpu = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=False)
                    gpu_gm = float(eq_gpu.GM.values.item())
                    
                    diff = abs(cpu_gm - gpu_gm)
                    if diff < 1e-6:
                        passed += 1
                    if diff > max_diff:
                        max_diff = diff
                except:
                    pass
    
    print(f"Results: {passed}/{total} passed ({100*passed/total:.1f}%)")
    print(f"Max difference: {max_diff:.2e} J/mol")
    
except Exception as e:
    print(f"Error: {e}")
finally:
    if sys.stderr != original_stderr:
        sys.stderr.close()
        sys.stderr = original_stderr