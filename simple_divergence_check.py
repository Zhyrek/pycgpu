#!/usr/bin/env python
"""Simple check for CPU vs GPU divergences."""

from pycalphad import Database, equilibrium
from pycalphad.core.debug_output import reset_debug_session, close_debug_output
import numpy as np
import warnings
import sys
import io

warnings.filterwarnings('ignore')

# Load TDB
tdb = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['LIQUID', 'BCC_A2']

# Test conditions
test_conditions = [
    {'T': 1000, 'P': 101325, 'X(TI)': 0.001},
    {'T': 1000, 'P': 101325, 'X(TI)': 0.01},
    {'T': 1000, 'P': 101325, 'X(TI)': 0.1},
    {'T': 1000, 'P': 101325, 'X(TI)': 0.5},
    {'T': 1000, 'P': 101325, 'X(TI)': 0.9},
    {'T': 1000, 'P': 101325, 'X(TI)': 0.99},
    {'T': 1000, 'P': 101325, 'X(TI)': 0.999},
    {'T': 300, 'P': 101325, 'X(TI)': 0.5},
    {'T': 2000, 'P': 101325, 'X(TI)': 0.5},
    {'T': 3000, 'P': 101325, 'X(TI)': 0.5},
]

print("Condition | CPU GM | GPU GM | |ΔGM| | Status")
print("-" * 70)

for cond in test_conditions:
    # Suppress output
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    try:
        # CPU calculation
        reset_debug_session()
        eq_cpu = equilibrium(tdb, comps, phases, cond, gpu=False, verbose=False)
        cpu_gm = eq_cpu.GM.values.item()
        
        # GPU calculation
        reset_debug_session()
        eq_gpu = equilibrium(tdb, comps, phases, cond, gpu=True, verbose=False)
        gpu_gm = eq_gpu.GM.values.item()
        
        # Compare
        diff = abs(cpu_gm - gpu_gm)
        status = "OK" if diff < 1e-6 else "DIVERGED"
        
        sys.stdout = old_stdout
        print(f"T={cond['T']:4.0f} X={cond['X(TI)']:.3f} | {cpu_gm:11.2f} | {gpu_gm:11.2f} | {diff:.2e} | {status}")
        
    except Exception as e:
        sys.stdout = old_stdout
        print(f"T={cond['T']:4.0f} X={cond['X(TI)']:.3f} | ERROR: {str(e)[:30]}...")