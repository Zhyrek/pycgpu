#!/usr/bin/env python
"""Test CPU vs GPU across multiple conditions including the previously problematic ones."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test conditions including the problematic 0.5, 700K
test_conditions = [
    {'x_ti': 0.05, 'temp': 500},
    {'x_ti': 0.05, 'temp': 600},
    {'x_ti': 0.05, 'temp': 700},
    {'x_ti': 0.05, 'temp': 800},
    {'x_ti': 0.05, 'temp': 900},
    {'x_ti': 0.05, 'temp': 1000},
    {'x_ti': 0.1, 'temp': 500},
    {'x_ti': 0.1, 'temp': 600},
    {'x_ti': 0.1, 'temp': 700},
    {'x_ti': 0.1, 'temp': 800},
    {'x_ti': 0.1, 'temp': 900},
    {'x_ti': 0.1, 'temp': 1000},
    {'x_ti': 0.5, 'temp': 500},
    {'x_ti': 0.5, 'temp': 600},
    {'x_ti': 0.5, 'temp': 700},  # Previously problematic
    {'x_ti': 0.5, 'temp': 800},
    {'x_ti': 0.5, 'temp': 900},
    {'x_ti': 0.5, 'temp': 1000},
    {'x_ti': 0.9, 'temp': 500},
    {'x_ti': 0.9, 'temp': 600},
    {'x_ti': 0.9, 'temp': 700},
    {'x_ti': 0.9, 'temp': 800},
    {'x_ti': 0.9, 'temp': 900},
    {'x_ti': 0.9, 'temp': 1000},
    {'x_ti': 0.95, 'temp': 500},
    {'x_ti': 0.95, 'temp': 600},
    {'x_ti': 0.95, 'temp': 700},
    {'x_ti': 0.95, 'temp': 800},
    {'x_ti': 0.95, 'temp': 900},
    {'x_ti': 0.95, 'temp': 1000},
]

# Open file for writing results
with open('/tmp/output.txt', 'w') as f:
    f.write("CPU vs GPU Comprehensive Test Results\n")
    f.write("=" * 80 + "\n")
    f.write(f"Testing {len(test_conditions)} conditions\n\n")
    f.write(f"{'X(TI)':<10}{'T (K)':<10}{'CPU GM':<15}{'GPU GM':<15}{'Abs Error':<15}{'CPU Phases':<12}{'GPU Phases':<12}{'Status':<10}\n")
    f.write("-" * 110 + "\n")
    
    max_error = 0.0
    max_error_condition = None
    errors_over_10 = []
    
    for cond in test_conditions:
        x_ti = cond['x_ti']
        temp = cond['temp']
        
        conditions = {v.X('TI'): x_ti, v.T: temp, v.P: 101325}
        
        try:
            # CPU calculation
            result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
            cpu_gm = result_cpu.GM.values.flatten()[0]
            cpu_phases = result_cpu.NP.values.flatten()
            cpu_active = sum(1 for p in cpu_phases if p > 1e-12)
            
            # GPU calculation
            result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
            gpu_gm = result_gpu.GM.values.flatten()[0]
            gpu_phases = result_gpu.NP.values.flatten()
            gpu_active = sum(1 for p in gpu_phases if p > 1e-12)
            
            # Calculate error
            error = abs(cpu_gm - gpu_gm)
            
            # Track maximum error
            if error > max_error:
                max_error = error
                max_error_condition = (x_ti, temp)
            
            # Track errors over 10 J/mol
            if error > 10.0:
                errors_over_10.append((x_ti, temp, error))
            
            # Determine status
            if cpu_active == gpu_active and error < 10.0:
                status = "✓ PASS"
            elif cpu_active != gpu_active:
                status = "✗ PHASES"
            else:
                status = "⚠ ERROR"
            
            # Write results
            f.write(f"{x_ti:<10.2f}{temp:<10}{cpu_gm:<15.1f}{gpu_gm:<15.1f}{error:<15.6f}{cpu_active:<12}{gpu_active:<12}{status:<10}\n")
            
        except Exception as e:
            f.write(f"{x_ti:<10.2f}{temp:<10}{'ERROR':<15}{'ERROR':<15}{'N/A':<15}{'N/A':<12}{'N/A':<12}{'✗ FAILED':<10}\n")
            f.write(f"  Error: {str(e)}\n")
    
    # Summary statistics
    f.write("\n" + "=" * 80 + "\n")
    f.write("SUMMARY\n")
    f.write("=" * 80 + "\n")
    f.write(f"Maximum absolute error: {max_error:.6f} J/mol\n")
    if max_error_condition:
        f.write(f"  at X(TI)={max_error_condition[0]}, T={max_error_condition[1]}K\n")
    f.write(f"\nConditions with error > 10 J/mol: {len(errors_over_10)}\n")
    if errors_over_10:
        for x_ti, temp, error in errors_over_10:
            f.write(f"  X(TI)={x_ti}, T={temp}K: {error:.1f} J/mol\n")
    else:
        f.write("  None! All errors are under 10 J/mol\n")
    
    # Special note about previously problematic condition
    f.write(f"\nPreviously problematic condition (X(TI)=0.5, T=700K):\n")
    for cond in test_conditions:
        if cond['x_ti'] == 0.5 and cond['temp'] == 700:
            conditions = {v.X('TI'): 0.5, v.T: 700, v.P: 101325}
            result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
            result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
            cpu_gm = result_cpu.GM.values.flatten()[0]
            gpu_gm = result_gpu.GM.values.flatten()[0]
            error = abs(cpu_gm - gpu_gm)
            f.write(f"  CPU: {cpu_gm:.1f} J/mol\n")
            f.write(f"  GPU: {gpu_gm:.1f} J/mol\n")
            f.write(f"  Error: {error:.6f} J/mol (was 1547 J/mol before fix)\n")
            break

print("Running comprehensive test...")