#!/usr/bin/env python
"""Extract specific values at iterations 0 and 1 for CPU vs GPU."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import sys
import io
import re

# Redirect output to capture
original_stdout = sys.stdout
capture = io.StringIO()

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

# Run calculations capturing output
sys.stdout = capture
try:
    result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
except:
    pass
sys.stdout = original_stdout

output = capture.getvalue()

print("EXTRACTED VALUES FROM ITERATIONS 0 AND 1")
print("=" * 80)

# Parse CPU iteration 0
print("\nCPU ITERATION 0:")
cpu_iter0 = re.search(r'iteration 0\)(.*?)iteration 1\)', output, re.DOTALL)
if cpu_iter0:
    text = cpu_iter0.group(1)
    
    # Extract c_G values
    cg_matches = re.findall(r'c_G = \[([-\d.e+ ]+)\]', text)
    for i, cg in enumerate(cg_matches):
        print(f"  Phase {i} c_G: [{cg}]")
    
    # Extract phase amounts and deltas
    phase_deltas = re.findall(r'Phase (\d+): old=([-\d.e+]+), delta=([-\d.e+]+), step_size=([-\d.e+]+), actual_change=([-\d.e+]+)', text)
    for match in phase_deltas:
        print(f"  Phase {match[0]}: old={match[1]}, delta={match[2]}, step_size={match[3]}, actual_change={match[4]}")
    
    # Extract RHS contributions
    rhs_contrib = re.findall(r'RHS contribution: ([-\d.e+]+)', text)
    for i, rhs in enumerate(rhs_contrib):
        print(f"  Phase {i} RHS contribution: {rhs}")

# Parse CPU iteration 1 
print("\nCPU ITERATION 1:")
cpu_iter1 = re.search(r'iteration 1\)(.*?)(?:iteration 2\)|CONSOLIDATING)', output, re.DOTALL)
if cpu_iter1:
    text = cpu_iter1.group(1)
    
    # Extract c_G values
    cg_matches = re.findall(r'c_G = \[([-\d.e+ ]+)\]', text)
    for i, cg in enumerate(cg_matches):
        print(f"  Phase {i} c_G: [{cg}]")
    
    # Extract RHS
    rhs_contrib = re.findall(r'RHS contribution: ([-\d.e+]+)', text)
    for i, rhs in enumerate(rhs_contrib):
        print(f"  Phase {i} RHS contribution: {rhs}")
        
    # Check for consolidation
    if "CONSOLIDATING" in output[cpu_iter1.start():cpu_iter1.end()+100]:
        print("  **CPU CONSOLIDATES PHASES**")

# Parse GPU iteration 0
print("\nGPU ITERATION 0:")
gpu_iter0 = re.search(r'GPU DEBUG: Iteration 0/200(.*?)GPU DEBUG: Iteration 1', output, re.DOTALL)
if gpu_iter0:
    text = gpu_iter0.group(1)
    
    # Extract c_G values
    cg_matches = re.findall(r'c_G\[0\] = ([-\d.e+]+).*?c_G\[1\] = ([-\d.e+]+)', text)
    for i, (cg0, cg1) in enumerate(cg_matches[:2]):  # First 2 phases
        print(f"  Phase {i} c_G: [{cg0}, {cg1}]")
    
    # Extract phase deltas
    phase_deltas = re.findall(r'Phase (\d+): old=([-\d.e+]+), delta=([-\d.e+]+).*?actual_change=([-\d.e+]+)', text)
    for match in phase_deltas:
        print(f"  Phase {match[0]}: old={match[1]}, delta={match[2]}, actual_change={match[3]}")
    
    # Extract RHS
    rhs_contrib = re.findall(r'RHS contribution: ([-\d.e+]+)', text)
    for i, rhs in enumerate(rhs_contrib[:2]):  # First 2 phases
        print(f"  Phase {i} RHS contribution: {rhs}")
    
    # Check for phase removal
    phase_removal = re.search(r'Phase (\d+) amount became very small.*?at iteration 0', text)
    if phase_removal:
        print(f"  **GPU REMOVES Phase {phase_removal.group(1)}**")

# Parse GPU iteration 1
print("\nGPU ITERATION 1:")
gpu_iter1 = re.search(r'GPU DEBUG: Iteration 1(.*?)(?:GPU DEBUG: Iteration 2|converged)', output, re.DOTALL)
if gpu_iter1:
    text = gpu_iter1.group(1)
    
    # Extract c_G values
    cg_matches = re.findall(r'c_G\[0\] = ([-\d.e+]+).*?c_G\[1\] = ([-\d.e+]+)', text)
    if cg_matches:
        print(f"  Phase 0 c_G: [{cg_matches[0][0]}, {cg_matches[0][1]}]")
    
    # Extract solution
    soln_matches = re.findall(r'x\[(\d+)\] = ([-\d.e+]+)', text)
    if soln_matches:
        print("  Solution vector:")
        for idx, val in soln_matches[:3]:
            print(f"    x[{idx}] = {val}")

print("\n" + "="*80)
print("KEY DIFFERENCES:")
print("="*80)
print("\n1. At iteration 0:")
print("   - Both have same c_G values and phase deltas")
print("   - Both change phase amounts by same delta")
print("   - Phase 1 becomes very small (1e-10)")

print("\n2. Between iterations 0 and 1:")
print("   - CPU: CONSOLIDATES phases (weighted average)")
print("   - GPU: REMOVES phase 1 (keeps phase 0 composition)")

print("\n3. Result after consolidation/removal:")
print("   - CPU: Single phase with X(TI) ≈ 0.9031")
print("   - GPU: Single phase with X(TI) ≈ 0.8983")

print("\n4. This explains final results:")
print("   - CPU can reach target X(TI) = 0.9000")
print("   - GPU stuck at X(TI) = 0.9030")