#!/usr/bin/env python
"""Trace X(TI)=0.9 divergence between CPU and GPU to find first mismatch."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Test a specific problematic condition
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Use X(TI)=0.9, T=600K which shows 19 J/mol error
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("TRACING X(TI)=0.9, T=600K DIVERGENCE")
print("=" * 80)

# First, let's capture the starting point (which should be identical)
from pycalphad import calculate
from pycalphad.core.lower_convex_hull import lower_convex_hull

# Get the starting point
print("\n1. STARTING POINT (should be identical for CPU and GPU)")
print("-" * 80)

# This is what both CPU and GPU start with
calc_result = calculate(dbf, comps, phases, T=600, P=101325)
print(f"Calculate result shape: {calc_result.GM.shape}")

# Get the hull result which provides the starting guess
hull_result = lower_convex_hull(calc_result, conditions)
print(f"\nStarting point from lower_convex_hull:")
print(f"  Phase fractions: {hull_result.Phase.values.flatten()}")
print(f"  NP values: {hull_result.NP.values.flatten()}")

# Extract initial compositions
if hasattr(hull_result, 'X'):
    x_values = hull_result.X.sel(component='TI').values.flatten()
    print(f"  X(TI) per phase: {x_values}")

# Get site fractions
if hasattr(hull_result, 'Y'):
    y_shape = hull_result.Y.shape
    print(f"  Y shape: {y_shape}")
    # For BCC_A2, site fractions are direct
    for i in range(len(phases)):
        if hull_result.NP.values.flatten()[i] > 1e-12:
            phase_name = phases[i].name
            # Get site fractions for this phase
            y_ti = hull_result.Y.isel(vertex=i).sel(internal_dof='Y(BCC_A2,0,TI)').values.item() if 'Y(BCC_A2,0,TI)' in hull_result.Y.internal_dof else None
            if y_ti is not None:
                print(f"  Phase {i} ({phase_name}): Y(TI) = {y_ti:.6f}")

print("\nInitial phase amounts:")
phase_amounts = hull_result.NP.values.flatten()
for i, amount in enumerate(phase_amounts):
    if amount > 1e-12:
        print(f"  Phase {i}: {amount:.6f}")

# Save this data to compare with GPU's interpretation
with open('/tmp/trace_output.txt', 'w') as f:
    f.write("STARTING POINT DATA\n")
    f.write("=" * 80 + "\n")
    f.write(f"Condition: X(TI)=0.9, T=600K\n")
    f.write(f"Phase amounts: {phase_amounts}\n")
    f.write(f"Number of active phases: {sum(1 for a in phase_amounts if a > 1e-12)}\n")
    
    # Check if this is a two-phase starting point
    active_phases = [(i, phase_amounts[i]) for i in range(len(phase_amounts)) if phase_amounts[i] > 1e-12]
    f.write(f"Active phases: {active_phases}\n")
    
    if len(active_phases) == 2:
        f.write("\nTwo-phase starting point detected - checking compositions:\n")
        for i, amount in active_phases:
            if hasattr(hull_result, 'Y'):
                y_ti = hull_result.Y.isel(vertex=i).sel(internal_dof='Y(BCC_A2,0,TI)').values.item() if 'Y(BCC_A2,0,TI)' in hull_result.Y.internal_dof else None
                if y_ti is not None:
                    f.write(f"  Phase {i}: Y(TI) = {y_ti:.6f}, amount = {amount:.6f}\n")

print("\nStarting point saved to /tmp/trace_output.txt")
print("\nNow running CPU calculation with detailed tracing...")

# Enable detailed CPU debug output
os.environ['PYCALPHAD_DEBUG_EQSOLVER'] = '1'

# Run CPU calculation
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=True)

print("\nCPU calculation complete. Extracting key values...")

# Extract CPU results
cpu_gm = result_cpu.GM.values.flatten()[0]
cpu_phases = result_cpu.NP.values.flatten()
cpu_active = sum(1 for p in cpu_phases if p > 1e-12)

print(f"\nCPU Final Results:")
print(f"  GM: {cpu_gm:.6f} J/mol")
print(f"  Active phases: {cpu_active}")
print(f"  Phase fractions: {[f'{p:.6f}' for p in cpu_phases if p > 1e-12]}")

# Save CPU trace info
with open('/tmp/trace_output.txt', 'a') as f:
    f.write("\n\nCPU FINAL RESULTS\n")
    f.write("=" * 80 + "\n")
    f.write(f"GM: {cpu_gm:.6f} J/mol\n")
    f.write(f"Active phases: {cpu_active}\n")
    f.write(f"Phase fractions: {cpu_phases}\n")

print("\nNow running GPU calculation...")

# Run GPU calculation
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)

gpu_gm = result_gpu.GM.values.flatten()[0]
gpu_phases = result_gpu.NP.values.flatten()
gpu_active = sum(1 for p in gpu_phases if p > 1e-12)

print(f"\nGPU Final Results:")
print(f"  GM: {gpu_gm:.6f} J/mol")
print(f"  Active phases: {gpu_active}")
print(f"  Phase fractions: {[f'{p:.6f}' for p in gpu_phases if p > 1e-12]}")

error = abs(cpu_gm - gpu_gm)
print(f"\nFinal error: {error:.6f} J/mol")

# Save comparison
with open('/tmp/trace_output.txt', 'a') as f:
    f.write("\n\nGPU FINAL RESULTS\n")
    f.write("=" * 80 + "\n")
    f.write(f"GM: {gpu_gm:.6f} J/mol\n")
    f.write(f"Active phases: {gpu_active}\n")
    f.write(f"Phase fractions: {gpu_phases}\n")
    f.write(f"\nERROR: {error:.6f} J/mol\n")
    
    if cpu_active != gpu_active:
        f.write("\nWARNING: Different number of active phases!\n")
        f.write(f"CPU has {cpu_active} phases, GPU has {gpu_active} phases\n")

print("\nAnalysis saved to /tmp/trace_output.txt")
print("\nTo find divergence point, look for:")
print("1. First iteration where phase amounts differ")
print("2. First energy calculation mismatch")
print("3. First Hessian difference")
print("4. First gradient mismatch")