#!/usr/bin/env python3
"""
Quick test to verify the fix for single-phase region handling
Focus on conditions that were failing before
"""

import numpy as np
from pycalphad import Database, equilibrium

# Load database
db = Database('NbTi.tdb')

# Test specific conditions that were failing (single-phase regions)
test_conditions = [
    {'T': 600, 'P': 101325, 'X(TI)': 0.1},  # Single phase region
    {'T': 600, 'P': 101325, 'X(TI)': 0.5},  # Two-phase region (should still work)
    {'T': 600, 'P': 101325, 'X(TI)': 0.9},  # Single phase region
    {'T': 1000, 'P': 101325, 'X(TI)': 0.1}, # Single phase region at high T
    {'T': 1000, 'P': 101325, 'X(TI)': 0.9}, # Single phase region at high T
]

print("Testing specific conditions that were failing...")
print("="*80)
print(f"{'T (K)':>8} {'X(TI)':>8} {'CPU GM':>12} {'GPU GM':>12} {'Diff (J/mol)':>12} {'X_CPU':>8} {'X_GPU':>8} {'Status':>10}")
print("-"*80)

for cond in test_conditions:
    try:
        # GPU calculation
        gpu_eq = equilibrium(db, ['NB', 'TI'], 'BCC_A2',
                            cond, model=None, verbose=False,
                            calc_opts={'pdens': 50}, gpu=True)
        
        # CPU calculation
        cpu_eq = equilibrium(db, ['NB', 'TI'], 'BCC_A2',
                            cond, model=None, verbose=False,
                            calc_opts={'pdens': 50}, gpu=False)
        
        # Extract values
        gpu_gm = float(gpu_eq.GM.values[0])
        cpu_gm = float(cpu_eq.GM.values[0])
        diff = abs(gpu_gm - cpu_gm)
        
        gpu_x_ti = float(gpu_eq.X('BCC_A2', 'TI').values[0])
        cpu_x_ti = float(cpu_eq.X('BCC_A2', 'TI').values[0])
        x_diff = abs(gpu_x_ti - cpu_x_ti)
        
        # Check number of phases
        gpu_phases = gpu_eq.Phase.unique()
        cpu_phases = cpu_eq.Phase.unique()
        
        # Status
        if diff < 1.0 and x_diff < 0.001:
            status = "PASS"
        else:
            status = "FAIL"
            
        print(f"{cond['T']:8.0f} {cond['X(TI)']:8.2f} {cpu_gm:12.2f} {gpu_gm:12.2f} {diff:12.2f} {cpu_x_ti:8.4f} {gpu_x_ti:8.4f} {status:>10}")
        
        # Additional info for failures
        if status == "FAIL" or True:  # Always show phase info for debugging
            print(f"         CPU phases: {list(cpu_phases)}, GPU phases: {list(gpu_phases)}")
            
    except Exception as e:
        print(f"{cond['T']:8.0f} {cond['X(TI)']:8.2f} {'ERROR':>12} {'ERROR':>12} {'ERROR':>12} {'ERROR':>8} {'ERROR':>8} {'ERROR':>10}")
        print(f"         Error: {str(e)}")

print("-"*80)
print("\nDone!")