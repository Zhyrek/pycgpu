#!/usr/bin/env python
"""Extract and compare actual values at the divergence iteration from debug output."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import re

print("EXTRACTING ACTUAL VALUES AT DIVERGENCE ITERATION")
print("=" * 80)

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

# Capture debug output
import subprocess
import sys

# Run with debug output captured
print("\nCapturing CPU debug output...")
cpu_output = subprocess.check_output([sys.executable, '-c', """
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}
result = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
"""], stderr=subprocess.STDOUT, text=True)

print("\nCapturing GPU debug output...")
gpu_output = subprocess.check_output([sys.executable, '-c', """
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}
result = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
"""], stderr=subprocess.STDOUT, text=True)

# Parse iteration 1 values from CPU
print("\n" + "="*80)
print("CPU VALUES AT ITERATION 1:")
print("="*80)

# Extract CPU c_G values
cpu_cg_matches = re.findall(r'\[CPU c_G DEBUG\].*?c_G = \[([-\d.e+]+)\s+([-\d.e+]+)\]', cpu_output)
if len(cpu_cg_matches) >= 3:  # Should have iter 0 phase 0, iter 0 phase 1, iter 1 phase 0
    print(f"\nIteration 0:")
    print(f"  Phase 0 c_G: [{cpu_cg_matches[0][0]}, {cpu_cg_matches[0][1]}]")
    print(f"  Phase 1 c_G: [{cpu_cg_matches[1][0]}, {cpu_cg_matches[1][1]}]")
    print(f"\nIteration 1:")
    print(f"  Phase 0 c_G: [{cpu_cg_matches[2][0]}, {cpu_cg_matches[2][1]}]")

# Extract CPU RHS values
cpu_rhs_matches = re.findall(r'RHS contribution: ([-\d.e+]+)', cpu_output)
if cpu_rhs_matches:
    print(f"\nRHS contributions at iteration 1:")
    for i, rhs in enumerate(cpu_rhs_matches[2:4]):  # Get iteration 1 values
        print(f"  Phase {i}: {rhs}")

# Extract matrix values
cpu_matrix_match = re.search(r'Equilibrium matrix at iteration 1.*?\n((?:.*?\n){4})', cpu_output)
if cpu_matrix_match:
    print("\nEquilibrium matrix at iteration 1:")
    print(cpu_matrix_match.group(1).strip())

# Extract consolidation info
if "CONSOLIDATING phases" in cpu_output:
    print("\nCPU: PHASES CONSOLIDATED at iteration 1")

print("\n" + "="*80)
print("GPU VALUES AT ITERATION 1:")
print("="*80)

# Extract GPU c_G values  
gpu_cg_matches = re.findall(r'c_G\[0\] = ([-\d.e+]+).*?c_G\[1\] = ([-\d.e+]+)', gpu_output)
if gpu_cg_matches:
    print(f"\nIteration 0:")
    if len(gpu_cg_matches) >= 2:
        print(f"  Phase 0 c_G: [{gpu_cg_matches[0][0]}, {gpu_cg_matches[0][1]}]")
        print(f"  Phase 1 c_G: [{gpu_cg_matches[1][0]}, {gpu_cg_matches[1][1]}]")

# Extract GPU RHS
gpu_rhs_matches = re.findall(r'RHS contribution: ([-\d.e+]+)', gpu_output)
if gpu_rhs_matches:
    print(f"\nRHS contributions:")
    for i, rhs in enumerate(gpu_rhs_matches[:2]):
        print(f"  Phase {i}: {rhs}")

# Check for phase removal
gpu_phase_removal = re.search(r'Phase (\d+) amount became very small.*?at iteration (\d+)', gpu_output)
if gpu_phase_removal:
    print(f"\nGPU: Phase {gpu_phase_removal.group(1)} REMOVED at iteration {gpu_phase_removal.group(2)}")

# Extract final compositions
cpu_final_match = re.search(r'CPU result: X\(TI\) = ([\d.]+)', cpu_output)
gpu_final_match = re.search(r'GPU result: X\(TI\) = ([\d.]+)', gpu_output)

print("\n" + "="*80)
print("KEY DIFFERENCE:")
print("="*80)
print("\nAfter iteration 0/1:")
print("- CPU consolidates phases and gets single phase with X(TI) ≈ 0.9031")
print("- GPU removes phase but keeps X(TI) ≈ 0.8983 from remaining phase")
print("\nThis is why GPU can never reach target of 0.900000!")

if cpu_final_match and gpu_final_match:
    print(f"\nFinal results:")
    print(f"  CPU: X(TI) = {cpu_final_match.group(1)}")
    print(f"  GPU: X(TI) = {gpu_final_match.group(1)}")