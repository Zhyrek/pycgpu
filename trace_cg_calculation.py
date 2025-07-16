#!/usr/bin/env python
"""Trace c_G calculation differences between CPU and GPU."""

from pycalphad import Database, equilibrium
from pycalphad.core.debug_output import reset_debug_session, close_debug_output
import numpy as np
import re

# Load TDB
tdb = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['LIQUID', 'BCC_A2']

# Single condition
conditions = {
    'T': 1000,
    'P': 101325,
    'X(TI)': 0.5
}

# Capture output to analyze
import io
import sys

# Run CPU calculation with verbose output
cpu_output = io.StringIO()
old_stdout = sys.stdout
sys.stdout = cpu_output
reset_debug_session()
eq_cpu = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=True)
sys.stdout = old_stdout
cpu_text = cpu_output.getvalue()

# Run GPU calculation with verbose output  
gpu_output = io.StringIO()
sys.stdout = gpu_output
reset_debug_session()
eq_gpu = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=True)
sys.stdout = old_stdout
gpu_text = gpu_output.getvalue()

print("="*80)
print("C_G CALCULATION COMPARISON")
print("="*80)

# Extract c_G calculation details from CPU output
cpu_cg_pattern = r"\[CPU c_G DEBUG\] Phase (\d+) calculation:(.*?)After c_G calc, c_G = \[(.*?)\]"
cpu_cg_matches = list(re.finditer(cpu_cg_pattern, cpu_text, re.DOTALL))

print("\nCPU c_G calculations:")
for match in cpu_cg_matches:
    phase_idx = int(match.group(1))
    details = match.group(2)
    final_cg = match.group(3).strip()
    
    print(f"\n  Phase {phase_idx}:")
    
    # Extract gradient values
    grad_match = re.search(r"gradient values: \[(.*?)\]", details)
    if grad_match:
        print(f"    Gradients: [{grad_match.group(1)}]")
    
    # Extract e_matrix diagonal
    ematrix_match = re.search(r"full_e_matrix diagonal: \[(.*?)\]", details)
    if ematrix_match:
        print(f"    E-matrix diagonal: [{ematrix_match.group(1)}]")
    
    # Extract individual c_G calculations
    cg_calc_pattern = r"c_G\[0\] -= ([-+]?\d+\.?\d*e[+-]?\d+) \* ([-+]?\d+\.?\d*e[+-]?\d+) = ([-+]?\d+\.?\d*e[+-]?\d+)"
    cg_calcs = re.findall(cg_calc_pattern, details)
    if cg_calcs:
        print("    c_G[0] calculation steps:")
        for i, (ematrix_val, grad_val, product) in enumerate(cg_calcs):
            print(f"      Step {i}: {ematrix_val} * {grad_val} = {product}")
    
    print(f"    Final c_G: [{final_cg}]")

# Extract c_G values from GPU output
gpu_cg_pattern = r"GPU DEBUG: Phase (\d+) c_G values \(iter \d+\):(.*?)(?=GPU DEBUG:|$)"
gpu_cg_matches = list(re.finditer(gpu_cg_pattern, gpu_text, re.DOTALL))

print("\n\nGPU c_G calculations:")
for match in gpu_cg_matches:
    phase_idx = int(match.group(1))
    details = match.group(2)
    
    print(f"\n  Phase {phase_idx}:")
    
    # Extract gradient values
    grad_match = re.search(r"gradient values: \[(.*?)\]", details)
    if grad_match:
        print(f"    Gradients: [{grad_match.group(1)}]")
    
    # Extract e_matrix diagonal
    ematrix_match = re.search(r"full_e_matrix diagonal: \[(.*?)\]", details)
    if ematrix_match:
        print(f"    E-matrix diagonal: [{ematrix_match.group(1)}]")
    
    # Extract c_G values
    cg_val_pattern = r"c_G\[(\d+)\] = ([-+]?\d+\.?\d*e[+-]?\d+)"
    cg_vals = re.findall(cg_val_pattern, details)
    if cg_vals:
        print("    c_G values:")
        for idx, val in cg_vals:
            print(f"      c_G[{idx}] = {val}")

# Look for specific c_G calc steps in GPU output
gpu_calc_pattern = r"c_G\[(\d+)\] calc: full_e_matrix\[(\d+),(\d+)\]=([-+]?\d+\.?\d*e[+-]?\d+) \* grad\[(\d+)\]=([-+]?\d+\.?\d*e[+-]?\d+) = ([-+]?\d+\.?\d*e[+-]?\d+)"
gpu_calc_matches = list(re.finditer(gpu_calc_pattern, gpu_text))

if gpu_calc_matches:
    print("\n\nGPU c_G calculation steps:")
    current_phase = None
    for match in gpu_calc_matches:
        cg_idx = int(match.group(1))
        i = int(match.group(2))
        j = int(match.group(3))
        ematrix_val = match.group(4)
        grad_idx = int(match.group(5))
        grad_val = match.group(6)
        product = match.group(7)
        
        # Determine which phase this is for based on context
        if cg_idx == 0 and i == 0 and j == 0:
            if current_phase is None or current_phase == 1:
                current_phase = 0
            else:
                current_phase = 1
            print(f"\n  Phase {current_phase}:")
        
        print(f"    c_G[{cg_idx}] calc step [{i},{j}]: {ematrix_val} * grad[{grad_idx}]={grad_val} = {product}")

# Compare specific values
print("\n\n" + "="*80)
print("NUMERICAL COMPARISON")
print("="*80)

# Extract first iteration c_G values for comparison
cpu_phase0_cg = None
gpu_phase0_cg = None

# CPU Phase 0 c_G
cpu_p0_match = re.search(r"\[CPU c_G DEBUG\] Phase 0 calculation:.*?After c_G calc, c_G = \[(.*?)\]", cpu_text, re.DOTALL)
if cpu_p0_match:
    cg_vals = cpu_p0_match.group(1).strip().split()
    if len(cg_vals) >= 2:
        cpu_phase0_cg = [float(cg_vals[0]), float(cg_vals[1])]
        print(f"\nCPU Phase 0 c_G: {cpu_phase0_cg}")

# GPU Phase 0 c_G
gpu_p0_matches = re.findall(r"GPU DEBUG: Phase 0 c_G values.*?c_G\[0\] = ([-+]?\d+\.?\d*e[+-]?\d+).*?c_G\[1\] = ([-+]?\d+\.?\d*e[+-]?\d+)", gpu_text, re.DOTALL)
if gpu_p0_matches:
    gpu_phase0_cg = [float(gpu_p0_matches[0][0]), float(gpu_p0_matches[0][1])]
    print(f"GPU Phase 0 c_G: {gpu_phase0_cg}")

if cpu_phase0_cg and gpu_phase0_cg:
    print(f"\nDifferences in Phase 0 c_G:")
    for i in range(2):
        diff = abs(cpu_phase0_cg[i] - gpu_phase0_cg[i])
        rel_err = diff / abs(cpu_phase0_cg[i]) if cpu_phase0_cg[i] != 0 else float('inf')
        print(f"  c_G[{i}]: CPU={cpu_phase0_cg[i]:.15e}, GPU={gpu_phase0_cg[i]:.15e}")
        print(f"    Difference: {diff:.15e} ({rel_err*100:.3e}%)")

# Also check the gradient values that go into c_G calculation
print("\n\nGRADIENT COMPARISON:")

# Extract CPU gradients
cpu_grad_pattern = r"\[CPU\] Phase (\d+) gradient \(iteration 0\): \[(.*?)\]"
cpu_grad_matches = re.findall(cpu_grad_pattern, cpu_text)
if cpu_grad_matches:
    for phase_idx, grad_str in cpu_grad_matches[:2]:  # First 2 phases
        grads = [float(x) for x in grad_str.split()]
        if len(grads) >= 5:
            print(f"\nCPU Phase {phase_idx} gradients: Y(NB)={grads[3]:.6e}, Y(TI)={grads[4]:.6e}")

# Extract GPU gradients
gpu_grad_pattern = r"GPU DEBUG: Phase (\d+) gradients before c_G calculation:.*?grad\[3\] = ([-+]?\d+\.?\d*e[+-]?\d+).*?grad\[4\] = ([-+]?\d+\.?\d*e[+-]?\d+)"
gpu_grad_matches = re.findall(gpu_grad_pattern, gpu_text, re.DOTALL)
if gpu_grad_matches:
    for phase_idx, grad3, grad4 in gpu_grad_matches[:2]:  # First 2 phases
        print(f"GPU Phase {phase_idx} gradients: Y(NB)={grad3}, Y(TI)={grad4}")