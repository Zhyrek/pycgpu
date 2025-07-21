#!/usr/bin/env python
"""Detailed trace of X(TI)=0.1, T=600K to find first CPU/GPU deviation."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import sys
import io
import re

# Load database and set up calculation
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test condition: X(TI)=0.1, T=600K
conditions = {v.X('TI'): 0.1, v.T: 600, v.P: 101325}

print("="*80)
print("DETAILED TRACE: X(TI)=0.1, T=600K")
print("="*80)

# Function to extract numerical values from trace output
def extract_trace_values(output_text, label):
    """Extract values from trace output for comparison."""
    values = {}
    
    # Extract iteration data
    iter_pattern = rf"{label} TRACE.*?AFTER ITERATION (\d+).*?END ITERATION \1"
    iterations = re.findall(iter_pattern, output_text, re.DOTALL)
    
    for i, iter_text in enumerate(iterations):
        iter_num = re.search(r"AFTER ITERATION (\d+)", iter_text).group(1)
        
        # Extract key values
        chem_pot_match = re.search(r"Chemical potentials: \[([-.\d\se]+)\]", iter_text)
        if chem_pot_match:
            values[f"iter_{iter_num}_chem_pot"] = [float(x) for x in chem_pot_match.group(1).split()]
        
        mass_res_match = re.search(r"Mass residual: ([-.\d\se]+)", iter_text)
        if mass_res_match:
            values[f"iter_{iter_num}_mass_residual"] = float(mass_res_match.group(1))
        
        # Extract phase data
        phase_matches = re.findall(r"Phase (\d+).*?NP \(mole fraction\): ([-.\d\se]+).*?phase_amt \(formula units\): ([-.\d\se]+).*?energy: ([-.\d\se]+).*?Site fractions: \[([-.\d\s]+)\]", iter_text, re.DOTALL)
        for phase_data in phase_matches:
            phase_idx = phase_data[0]
            values[f"iter_{iter_num}_phase_{phase_idx}_NP"] = float(phase_data[1])
            values[f"iter_{iter_num}_phase_{phase_idx}_amt"] = float(phase_data[2])
            values[f"iter_{iter_num}_phase_{phase_idx}_energy"] = float(phase_data[3])
            values[f"iter_{iter_num}_phase_{phase_idx}_Y"] = [float(x) for x in phase_data[4].split()]
    
    return values

# Capture CPU output
print("\nRUNNING CPU CALCULATION...")
cpu_output = io.StringIO()
old_stdout = sys.stdout
sys.stdout = cpu_output
try:
    result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=True)
finally:
    sys.stdout = old_stdout

cpu_text = cpu_output.getvalue()
cpu_values = extract_trace_values(cpu_text, "CPU")

# Print CPU iterations summary
print("\nCPU ITERATIONS SUMMARY:")
cpu_iters = sorted([k for k in cpu_values.keys() if k.startswith("iter_") and "_chem_pot" in k])
for iter_key in cpu_iters:
    iter_num = iter_key.split("_")[1]
    print(f"\nIteration {iter_num}:")
    print(f"  Chemical potentials: {cpu_values.get(iter_key, 'N/A')}")
    print(f"  Mass residual: {cpu_values.get(f'iter_{iter_num}_mass_residual', 'N/A')}")
    
    # Print phase data
    phase_keys = sorted([k for k in cpu_values.keys() if f"iter_{iter_num}_phase_" in k and "_NP" in k])
    for phase_key in phase_keys:
        phase_num = phase_key.split("_")[3]
        print(f"  Phase {phase_num}:")
        print(f"    NP: {cpu_values.get(phase_key, 'N/A')}")
        print(f"    Energy: {cpu_values.get(f'iter_{iter_num}_phase_{phase_num}_energy', 'N/A')}")
        print(f"    Y: {cpu_values.get(f'iter_{iter_num}_phase_{phase_num}_Y', 'N/A')}")

# Capture GPU output
print("\n" + "="*80)
print("RUNNING GPU CALCULATION...")
gpu_output = io.StringIO()
sys.stdout = gpu_output
try:
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)
finally:
    sys.stdout = old_stdout

gpu_text = gpu_output.getvalue()
gpu_values = extract_trace_values(gpu_text, "GPU")

# Print GPU iterations summary
print("\nGPU ITERATIONS SUMMARY:")
gpu_iters = sorted([k for k in gpu_values.keys() if k.startswith("iter_") and "_chem_pot" in k])
for iter_key in gpu_iters:
    iter_num = iter_key.split("_")[1]
    print(f"\nIteration {iter_num}:")
    print(f"  Chemical potentials: {gpu_values.get(iter_key, 'N/A')}")
    print(f"  Mass residual: {gpu_values.get(f'iter_{iter_num}_mass_residual', 'N/A')}")
    
    # Print phase data
    phase_keys = sorted([k for k in gpu_values.keys() if f"iter_{iter_num}_phase_" in k and "_NP" in k])
    for phase_key in phase_keys:
        phase_num = phase_key.split("_")[3]
        print(f"  Phase {phase_num}:")
        print(f"    NP: {gpu_values.get(phase_key, 'N/A')}")
        print(f"    Energy: {gpu_values.get(f'iter_{iter_num}_phase_{phase_num}_energy', 'N/A')}")
        print(f"    Y: {gpu_values.get(f'iter_{iter_num}_phase_{phase_num}_Y', 'N/A')}")

# Compare values
print("\n" + "="*80)
print("COMPARING CPU vs GPU VALUES:")
print("="*80)

# Find first deviation
first_deviation = None
for key in sorted(set(cpu_values.keys()) | set(gpu_values.keys())):
    cpu_val = cpu_values.get(key, "MISSING")
    gpu_val = gpu_values.get(key, "MISSING")
    
    if cpu_val != "MISSING" and gpu_val != "MISSING":
        if isinstance(cpu_val, list) and isinstance(gpu_val, list):
            # Compare lists
            if len(cpu_val) == len(gpu_val):
                diffs = [abs(c - g) for c, g in zip(cpu_val, gpu_val)]
                max_diff = max(diffs)
                if max_diff > 1e-6:
                    print(f"\n{key}:")
                    print(f"  CPU: {cpu_val}")
                    print(f"  GPU: {gpu_val}")
                    print(f"  Max diff: {max_diff}")
                    if first_deviation is None:
                        first_deviation = (key, cpu_val, gpu_val, max_diff)
        else:
            # Compare scalars
            diff = abs(cpu_val - gpu_val)
            if diff > 1e-6:
                print(f"\n{key}:")
                print(f"  CPU: {cpu_val}")
                print(f"  GPU: {gpu_val}")
                print(f"  Diff: {diff}")
                if first_deviation is None:
                    first_deviation = (key, cpu_val, gpu_val, diff)

if first_deviation:
    print("\n" + "="*80)
    print("FIRST DEVIATION FOUND:")
    print(f"  Key: {first_deviation[0]}")
    print(f"  CPU: {first_deviation[1]}")
    print(f"  GPU: {first_deviation[2]}")
    print(f"  Difference: {first_deviation[3]}")
else:
    print("\nNo significant deviations found in traced values!")

# Final GM comparison
print("\n" + "="*80)
print("FINAL RESULTS:")
cpu_gm = result_cpu.GM.values.flatten()[0]
gpu_gm = result_gpu.GM.values.flatten()[0]
print(f"CPU GM: {cpu_gm:.6f} J/mol")
print(f"GPU GM: {gpu_gm:.6f} J/mol")
print(f"Difference: {gpu_gm - cpu_gm:.6f} J/mol")