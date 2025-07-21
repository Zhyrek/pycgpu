#!/usr/bin/env python3
"""
Test GPU vs CPU equilibrium calculations after fixing phase consolidation
"""

import numpy as np
from pycalphad import Database, equilibrium
import matplotlib.pyplot as plt
import sys

# Load database
db = Database('NbTi.tdb')

# Create test conditions
temps = np.array([300, 400, 500, 600, 700, 800, 900, 1000, 1100])
x_ti_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

# Mesh grid
T_grid, X_grid = np.meshgrid(temps, x_ti_values)
conditions = []
for i in range(len(temps)):
    for j in range(len(x_ti_values)):
        conditions.append({
            'T': T_grid[j, i],
            'P': 101325,
            'X(TI)': X_grid[j, i]
        })

# Run equilibrium calculations
print("Running GPU equilibrium calculations...")
gpu_results = []
cpu_results = []

total_conditions = len(conditions)
passed = 0
failed = 0

print("\nCondition Results:")
print("="*80)
print(f"{'T (K)':>8} {'X(TI)':>8} {'CPU GM':>12} {'GPU GM':>12} {'Diff (J/mol)':>12} {'Status':>10}")
print("-"*80)

for idx, cond in enumerate(conditions):
    try:
        # GPU calculation
        gpu_eq = equilibrium(db, ['NB', 'TI'], 'BCC_A2',
                            cond, model=None, verbose=False,
                            calc_opts={'pdens': 50}, gpu=True)
        
        # CPU calculation
        cpu_eq = equilibrium(db, ['NB', 'TI'], 'BCC_A2',
                            cond, model=None, verbose=False,
                            calc_opts={'pdens': 50}, gpu=False)
        
        # Extract GM values
        gpu_gm = float(gpu_eq.GM.values[0])
        cpu_gm = float(cpu_eq.GM.values[0])
        diff = abs(gpu_gm - cpu_gm)
        
        # Check composition
        gpu_x_ti = float(gpu_eq.X('BCC_A2', 'TI').values[0])
        cpu_x_ti = float(cpu_eq.X('BCC_A2', 'TI').values[0])
        x_diff = abs(gpu_x_ti - cpu_x_ti)
        
        # Status based on both energy and composition
        if diff < 1.0 and x_diff < 0.001:  # < 1 J/mol and < 0.1% composition
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1
            
        print(f"{cond['T']:8.0f} {cond['X(TI)']:8.2f} {cpu_gm:12.2f} {gpu_gm:12.2f} {diff:12.2f} {status:>10}")
        
        # Additional info for failures
        if status == "FAIL":
            print(f"         X(TI): CPU={cpu_x_ti:.6f}, GPU={gpu_x_ti:.6f}, diff={x_diff:.6f}")
            
    except Exception as e:
        print(f"{cond['T']:8.0f} {cond['X(TI)']:8.2f} {'ERROR':>12} {'ERROR':>12} {'ERROR':>12} {'ERROR':>10}")
        print(f"         Error: {str(e)}")
        failed += 1

print("-"*80)
print(f"\nSummary: {passed}/{total_conditions} passed ({100*passed/total_conditions:.1f}%)")
print(f"Failed conditions: {failed}")

# Identify patterns in failures
print("\n\nAnalyzing failure patterns...")
failure_temps = []
failure_x_ti = []
for idx, cond in enumerate(conditions):
    try:
        gpu_eq = equilibrium(db, ['NB', 'TI'], 'BCC_A2',
                            cond, model=None, verbose=False,
                            calc_opts={'pdens': 50}, gpu=True)
        cpu_eq = equilibrium(db, ['NB', 'TI'], 'BCC_A2',
                            cond, model=None, verbose=False,
                            calc_opts={'pdens': 50}, gpu=False)
        
        gpu_gm = float(gpu_eq.GM.values[0])
        cpu_gm = float(cpu_eq.GM.values[0])
        diff = abs(gpu_gm - cpu_gm)
        
        if diff >= 1.0:
            failure_temps.append(cond['T'])
            failure_x_ti.append(cond['X(TI)'])
    except:
        pass

if failure_temps:
    print(f"\nFailures occur at:")
    print(f"  Temperature range: {min(failure_temps):.0f}K - {max(failure_temps):.0f}K")
    print(f"  X(TI) range: {min(failure_x_ti):.2f} - {max(failure_x_ti):.2f}")

print("\nDone!")