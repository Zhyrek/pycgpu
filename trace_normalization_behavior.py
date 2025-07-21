#!/usr/bin/env python
"""Trace phase amount normalization behavior during consolidation."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import sys
import io

# Load database and set up calculation
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test single-phase condition that requires consolidation
conditions = {v.X('TI'): 0.1, v.T: 600, v.P: 101325}

print("Tracing phase amount normalization during consolidation")
print("="*80)
print(f"Condition: X(TI)={conditions[v.X('TI')]}, T={conditions[v.T]}K")
print()

# First, run CPU calculation
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
cpu_gm = float(result_cpu.GM.values)
cpu_phases = result_cpu.Phase.values.flatten()
cpu_np = result_cpu.NP.values.flatten()

print("CPU Result:")
print(f"  GM = {cpu_gm:.15f} J/mol")
print(f"  Active phases: {[(p, a) for p, a in zip(cpu_phases, cpu_np) if p and a > 1e-6]}")
print(f"  Phase amount sum: {sum(a for a in cpu_np if a > 1e-6):.15f}")
print()

# Run GPU with verbose output
print("GPU Calculation:")
print("-"*60)

# Capture GPU output
gpu_output = io.StringIO()
old_stdout = sys.stdout
sys.stdout = gpu_output
try:
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)
finally:
    sys.stdout = old_stdout

gpu_text = gpu_output.getvalue()
gpu_gm = float(result_gpu.GM.values)
gpu_phases = result_gpu.Phase.values.flatten()
gpu_np = result_gpu.NP.values.flatten()

print(f"GPU Final GM = {gpu_gm:.15f} J/mol")
print(f"GPU Active phases: {[(p, a) for p, a in zip(gpu_phases, gpu_np) if p and a > 1e-6]}")
print(f"GPU Phase amount sum: {sum(a for a in gpu_np if a > 1e-6):.15f}")
print()

# Look for normalization events in GPU output
lines = gpu_text.split('\n')
print("Searching for normalization events in GPU output:")
print("-"*60)

normalization_found = False
consolidation_found = False
phase_sum_lines = []

for i, line in enumerate(lines):
    # Look for normalization
    if "normalization" in line.lower() or "normalize" in line.lower():
        print(f"Line {i}: {line}")
        normalization_found = True
        # Print context
        for j in range(max(0, i-3), min(len(lines), i+5)):
            if "phase" in lines[j].lower() and ("sum" in lines[j] or "amount" in lines[j]):
                print(f"  Context [{j-i:+d}]: {lines[j]}")
    
    # Look for phase sum
    if "phase_amount_sum" in line or "phase_amt_sum" in line or "sum(phase_amt)" in line:
        phase_sum_lines.append((i, line))
    
    # Look for consolidation
    if "Consolidated phases" in line or "CONSOLIDATION" in line:
        consolidation_found = True
        print(f"\nCONSOLIDATION at line {i}: {line}")
        # Print detailed context
        for j in range(max(0, i-5), min(len(lines), i+10)):
            if any(keyword in lines[j] for keyword in ["phase_amt", "amount=", "sum", "total"]):
                print(f"  Context [{j-i:+d}]: {lines[j]}")

# Print all phase sum observations
if phase_sum_lines:
    print("\nPhase sum observations:")
    for i, line in phase_sum_lines:
        print(f"  Line {i}: {line}")

# Analyze the pattern
print("\n" + "="*80)
print("ANALYSIS:")
print(f"- CPU phase amount sum: {sum(a for a in cpu_np if a > 1e-6):.15f}")
print(f"- GPU phase amount sum: {sum(a for a in gpu_np if a > 1e-6):.15f}")
print(f"- Normalization found in GPU: {normalization_found}")
print(f"- Consolidation found in GPU: {consolidation_found}")
print(f"- Error in GM: {gpu_gm - cpu_gm:.15e} J/mol")

# Check if the issue is that GPU normalizes to 1.0 after consolidation
if abs(sum(a for a in gpu_np if a > 1e-6) - 1.0) < 1e-10:
    print("\n** GPU phase amounts sum to exactly 1.0 - this suggests normalization after consolidation **")
    print("** This is likely the source of the numerical error **")