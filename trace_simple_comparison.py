#!/usr/bin/env python
"""Simple comparison of CPU vs GPU trace outputs for X(TI)=0.1, T=600K."""

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
print("TRACE COMPARISON: X(TI)=0.1, T=600K")
print("="*80)

# Capture CPU output
print("\nCAPTURING CPU OUTPUT...")
cpu_output = io.StringIO()
old_stdout = sys.stdout
sys.stdout = cpu_output
try:
    result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=True)
finally:
    sys.stdout = old_stdout

cpu_text = cpu_output.getvalue()

# Capture GPU output
print("CAPTURING GPU OUTPUT...")
gpu_output = io.StringIO()
sys.stdout = gpu_output
try:
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)
finally:
    sys.stdout = old_stdout

gpu_text = gpu_output.getvalue()

# Extract key patterns from both outputs
def extract_patterns(text, label):
    """Extract key numerical patterns from output."""
    results = []
    
    # Look for chemical potentials
    for match in re.finditer(r"Chemical potentials: \[([-\d.e+\s,]+)\]", text):
        values = [float(x) for x in match.group(1).replace(',', '').split()]
        results.append(f"{label} Chemical potentials: {values}")
    
    # Look for phase compositions
    for match in re.finditer(r"phase_compositions: \[([-\d.e+\s,]+)\]", text):
        values = [float(x) for x in match.group(1).replace(',', '').split()]
        results.append(f"{label} Phase compositions: {values}")
    
    # Look for site fractions
    for match in re.finditer(r"Site fractions: \[([-\d.e+\s,]+)\]", text):
        values = [float(x) for x in match.group(1).replace(',', '').split()]
        results.append(f"{label} Site fractions: {values}")
    
    # Look for energy values
    for match in re.finditer(r"energy: ([-\d.e+]+)", text):
        energy = float(match.group(1))
        results.append(f"{label} Energy: {energy}")
    
    # Look for phase amounts
    for match in re.finditer(r"phase_amt.*?: ([-\d.e+]+)", text):
        amt = float(match.group(1))
        results.append(f"{label} Phase amount: {amt}")
    
    # Look for mass residuals
    for match in re.finditer(r"Mass residual: ([-\d.e+]+)", text):
        residual = float(match.group(1))
        results.append(f"{label} Mass residual: {residual}")
    
    # Look for Hessian values
    for match in re.finditer(r"hess\[(\d+),(\d+)\] = ([-\d.e+]+)", text):
        i, j, val = match.groups()
        results.append(f"{label} Hessian[{i},{j}]: {float(val)}")
    
    # Look for gradient values
    for match in re.finditer(r"gradient values: \[([-\d.e+\s,]+)\]", text):
        values = [float(x) for x in match.group(1).replace(',', '').split()]
        results.append(f"{label} Gradient: {values}")
    
    # Look for c_G values
    for match in re.finditer(r"c_G values: \[([-\d.e+\s,]+)\]", text):
        values = [float(x) for x in match.group(1).replace(',', '').split()]
        results.append(f"{label} c_G: {values}")
    
    # Look for equilibrium matrix entries
    for match in re.finditer(r"Row \d+: ([+\-\d.e+\s]+)\| RHS: ([+\-\d.e+]+)", text):
        row_vals = [float(x) for x in match.group(1).split()]
        rhs = float(match.group(2))
        results.append(f"{label} Matrix row: {row_vals} | RHS: {rhs}")
    
    return results

# Extract patterns
cpu_patterns = extract_patterns(cpu_text, "CPU")
gpu_patterns = extract_patterns(gpu_text, "GPU")

# Compare line by line to find first difference
print("\nCOMPARING OUTPUTS...")
print("-"*80)

# Group patterns by type for easier comparison
def group_patterns(patterns):
    groups = {}
    for p in patterns:
        key = p.split(":")[0].strip()
        if key not in groups:
            groups[key] = []
        groups[key].append(p)
    return groups

cpu_groups = group_patterns(cpu_patterns)
gpu_groups = group_patterns(gpu_patterns)

# Compare each type
first_diff_found = False
for pattern_type in sorted(set(cpu_groups.keys()) | set(gpu_groups.keys())):
    cpu_items = cpu_groups.get(pattern_type, [])
    gpu_items = gpu_groups.get(pattern_type, [])
    
    if len(cpu_items) != len(gpu_items):
        print(f"\nDIFFERENCE in count of {pattern_type}:")
        print(f"  CPU: {len(cpu_items)} items")
        print(f"  GPU: {len(gpu_items)} items")
        if not first_diff_found:
            print("  ^^^ FIRST DIFFERENCE FOUND ^^^")
            first_diff_found = True
    else:
        # Compare values
        for i, (cpu_item, gpu_item) in enumerate(zip(cpu_items, gpu_items)):
            # Extract numerical values
            cpu_nums = re.findall(r"[-\d.e+]+", cpu_item.split(":")[-1])
            gpu_nums = re.findall(r"[-\d.e+]+", gpu_item.split(":")[-1])
            
            if cpu_nums and gpu_nums and len(cpu_nums) == len(gpu_nums):
                cpu_vals = [float(x) for x in cpu_nums]
                gpu_vals = [float(x) for x in gpu_nums]
                
                # Check for differences
                max_diff = 0
                for cv, gv in zip(cpu_vals, gpu_vals):
                    diff = abs(cv - gv)
                    max_diff = max(max_diff, diff)
                
                if max_diff > 1e-6:
                    print(f"\nDIFFERENCE in {pattern_type} (item {i}):")
                    print(f"  CPU: {cpu_vals}")
                    print(f"  GPU: {gpu_vals}")
                    print(f"  Max diff: {max_diff}")
                    if not first_diff_found:
                        print("  ^^^ FIRST DIFFERENCE FOUND ^^^")
                        first_diff_found = True

# Final results
print("\n" + "="*80)
print("FINAL RESULTS:")
cpu_gm = result_cpu.GM.values.flatten()[0]
gpu_gm = result_gpu.GM.values.flatten()[0]
print(f"CPU GM: {cpu_gm:.6f} J/mol")
print(f"GPU GM: {gpu_gm:.6f} J/mol")
print(f"Difference: {gpu_gm - cpu_gm:.6f} J/mol")