#!/usr/bin/env python3
"""
Final summary of GPU/CPU comparison after fix
"""
import numpy as np
from pycalphad import Database, equilibrium
import warnings
warnings.filterwarnings('ignore')

# Redirect debug output
import sys
import os
sys.stderr = open(os.devnull, 'w')

db = Database('NbTi.tdb')

# Test conditions
test_conditions = [
    {'T': 300, 'P': 101325, 'X(TI)': 0.1},
    {'T': 300, 'P': 101325, 'X(TI)': 0.5},
    {'T': 300, 'P': 101325, 'X(TI)': 0.9},
    {'T': 600, 'P': 101325, 'X(TI)': 0.1},
    {'T': 600, 'P': 101325, 'X(TI)': 0.5},
    {'T': 600, 'P': 101325, 'X(TI)': 0.9},
    {'T': 1000, 'P': 101325, 'X(TI)': 0.1},
    {'T': 1000, 'P': 101325, 'X(TI)': 0.5},
    {'T': 1000, 'P': 101325, 'X(TI)': 0.9},
]

print("GPU/CPU Comparison After Phase Consolidation Fix")
print("="*70)
print(f"{'T (K)':>6} {'X(TI)':>6} {'CPU GM':>10} {'GPU GM':>10} {'ΔGM':>8} {'CPU X':>8} {'GPU X':>8} {'Status':>8}")
print("-"*70)

passed = 0
failed = 0

for cond in test_conditions:
    try:
        # Redirect stdout temporarily
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        gpu_eq = equilibrium(db, ['NB', 'TI'], 'BCC_A2', cond,
                           model=None, verbose=False,
                           calc_opts={'pdens': 50}, gpu=True)
        
        cpu_eq = equilibrium(db, ['NB', 'TI'], 'BCC_A2', cond,
                           model=None, verbose=False,
                           calc_opts={'pdens': 50}, gpu=False)
        
        sys.stdout = old_stdout
        
        gpu_gm = float(gpu_eq.GM.values[0])
        cpu_gm = float(cpu_eq.GM.values[0])
        gm_diff = abs(gpu_gm - cpu_gm)
        
        gpu_x = float(gpu_eq.X('BCC_A2', 'TI').values[0])
        cpu_x = float(cpu_eq.X('BCC_A2', 'TI').values[0])
        x_diff = abs(gpu_x - cpu_x)
        
        if gm_diff < 1.0 and x_diff < 0.001:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1
            
        print(f"{cond['T']:6.0f} {cond['X(TI)']:6.2f} {cpu_gm:10.1f} {gpu_gm:10.1f} {gm_diff:8.1f} {cpu_x:8.4f} {gpu_x:8.4f} {status:>8}")
        
    except Exception as e:
        print(f"{cond['T']:6.0f} {cond['X(TI)']:6.2f} {'ERROR':>10} {'ERROR':>10} {'ERROR':>8} {'ERROR':>8} {'ERROR':>8} {'ERROR':>8}")
        failed += 1

print("-"*70)
print(f"\nSummary: {passed}/{len(test_conditions)} passed ({100*passed/len(test_conditions):.0f}%)")

# Check the specific case that was failing
print(f"\nSpecific case X(TI)=0.1, T=600K:")
cond = {'T': 600, 'P': 101325, 'X(TI)': 0.1}

old_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')

gpu_eq = equilibrium(db, ['NB', 'TI'], 'BCC_A2', cond,
                   model=None, verbose=False,
                   calc_opts={'pdens': 50}, gpu=True)
cpu_eq = equilibrium(db, ['NB', 'TI'], 'BCC_A2', cond,
                   model=None, verbose=False,
                   calc_opts={'pdens': 50}, gpu=False)

sys.stdout = old_stdout

gpu_x = float(gpu_eq.X('BCC_A2', 'TI').values[0])
cpu_x = float(cpu_eq.X('BCC_A2', 'TI').values[0])

print(f"  Was: GPU X(TI) = 0.102653")
print(f"  Now: GPU X(TI) = {gpu_x:.6f}")
print(f"  CPU: X(TI) = {cpu_x:.6f}")

if abs(gpu_x - cpu_x) < 0.001:
    print(f"  ✓ FIXED! Difference = {abs(gpu_x - cpu_x):.6f}")
else:
    print(f"  ✗ Still failing. Difference = {abs(gpu_x - cpu_x):.6f}")

# Restore stderr
sys.stderr = sys.__stderr__