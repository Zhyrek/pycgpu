#!/usr/bin/env python
"""Trace gradient calculation differences between CPU and GPU."""

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

# Run both calculations
cpu_output = io.StringIO()
old_stdout = sys.stdout
sys.stdout = cpu_output
reset_debug_session()
eq_cpu = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=True)
sys.stdout = old_stdout
cpu_text = cpu_output.getvalue()

gpu_output = io.StringIO()
sys.stdout = gpu_output
reset_debug_session()
eq_gpu = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=True)
sys.stdout = old_stdout
gpu_text = gpu_output.getvalue()

print("="*80)
print("GRADIENT CALCULATION TRACE")
print("="*80)

# Find the first gradient calculations for both CPU and GPU
# CPU format: "[CPU] Phase 0 gradient (iteration 0): [0. 0. -74.61288142 -40279.83760282 -35129.19972857]"
cpu_grad_pattern = r"\[CPU\] Phase (\d+) gradient \(iteration 0\): \[(.*?)\]"
cpu_grad_matches = re.findall(cpu_grad_pattern, cpu_text)

# GPU format: "GPU DEBUG: Phase 0 gradients before c_G calculation:"
# followed by "grad[3] = -4.027983760282061e+04"
gpu_grad_pattern = r"GPU DEBUG: Phase (\d+) gradients before c_G calculation:(.*?)(?=GPU DEBUG:|$)"
gpu_grad_matches = list(re.finditer(gpu_grad_pattern, gpu_text, re.DOTALL))

print("\nINITIAL GRADIENT VALUES (Iteration 0):")
print("-" * 60)

# Process CPU gradients
cpu_gradients = {}
for phase_str, grad_str in cpu_grad_matches[:2]:  # First 2 phases
    phase_idx = int(phase_str)
    grads = [float(x) for x in grad_str.split()]
    cpu_gradients[phase_idx] = grads
    if len(grads) >= 5:
        print(f"\nCPU Phase {phase_idx}:")
        print(f"  Full gradient: {grads}")
        print(f"  Site fraction gradients: Y(NB)={grads[3]:.15e}, Y(TI)={grads[4]:.15e}")

# Process GPU gradients
gpu_gradients = {}
for match in gpu_grad_matches[:2]:  # First 2 phases
    phase_idx = int(match.group(1))
    details = match.group(2)
    
    # Extract individual gradient values
    grad_vals = {}
    grad_val_pattern = r"grad\[(\d+)\] = ([-+]?\d+\.?\d*e[+-]?\d+)"
    for grad_match in re.finditer(grad_val_pattern, details):
        idx = int(grad_match.group(1))
        val = float(grad_match.group(2))
        grad_vals[idx] = val
    
    if grad_vals:
        gpu_gradients[phase_idx] = grad_vals
        print(f"\nGPU Phase {phase_idx}:")
        print(f"  Gradient values found: {grad_vals}")
        if 3 in grad_vals and 4 in grad_vals:
            print(f"  Site fraction gradients: Y(NB)={grad_vals[3]:.15e}, Y(TI)={grad_vals[4]:.15e}")

# Compare the values
print("\n\nGRADIENT COMPARISON:")
print("-" * 60)

for phase_idx in sorted(set(cpu_gradients.keys()) & set(gpu_gradients.keys())):
    print(f"\nPhase {phase_idx}:")
    cpu_grad = cpu_gradients[phase_idx]
    gpu_grad = gpu_gradients[phase_idx]
    
    # Compare site fraction gradients (indices 3 and 4 for CPU, should match GPU)
    if len(cpu_grad) >= 5 and 3 in gpu_grad and 4 in gpu_grad:
        print("  Y(NB) gradient:")
        print(f"    CPU: {cpu_grad[3]:.15e}")
        print(f"    GPU: {gpu_grad[3]:.15e}")
        diff_nb = abs(cpu_grad[3] - gpu_grad[3])
        print(f"    Difference: {diff_nb:.15e} ({diff_nb/abs(cpu_grad[3])*100:.3e}%)")
        
        print("  Y(TI) gradient:")
        print(f"    CPU: {cpu_grad[4]:.15e}")
        print(f"    GPU: {gpu_grad[4]:.15e}")
        diff_ti = abs(cpu_grad[4] - gpu_grad[4])
        print(f"    Difference: {diff_ti:.15e} ({diff_ti/abs(cpu_grad[4])*100:.3e}%)")

# Check if gradients are being called with the same inputs
print("\n\nDOF VALUES BEFORE GRADIENT CALCULATION:")
print("-" * 60)

# CPU DOF values (from phase record updates)
cpu_dof_pattern = r"Phase (\d+) gradient \(iteration 0\).*?phase_{} DOF: \[(.*?)\]"
# GPU DOF values
gpu_dof_pattern = r"GPU DEBUG: workspace DOF before formulamole_grad: \[(.*?)\]"

gpu_dof_matches = re.findall(gpu_dof_pattern, gpu_text)
if gpu_dof_matches:
    print("\nGPU workspace DOF values found:")
    for i, dof_str in enumerate(gpu_dof_matches[:2]):
        dof_vals = [float(x) for x in dof_str.split(',')]
        print(f"  Phase {i}: {dof_vals}")

# Look for site fraction values specifically
print("\n\nSITE FRACTION VALUES AT GRADIENT CALCULATION:")
print("-" * 60)

# GPU site fractions
gpu_sf_pattern = r"GPU DEBUG: Phase (\d+) current site fractions: \[(.*?)\]"
gpu_sf_matches = re.findall(gpu_sf_pattern, gpu_text)
for phase_str, sf_str in gpu_sf_matches[:2]:
    phase_idx = int(phase_str)
    sf_vals = [float(x) for x in sf_str.split(',')]
    print(f"\nGPU Phase {phase_idx} site fractions: {sf_vals}")

# Check masses which affect gradient calculation
print("\n\nMASSES (PHASE AMOUNTS):")
print("-" * 60)

# CPU masses
cpu_mass_pattern = r"\[CPU\] Phase (\d+) masses: \[\[(.*?)\]\]"
cpu_mass_matches = re.findall(cpu_mass_pattern, cpu_text)
for phase_str, mass_str in cpu_mass_matches[:2]:
    phase_idx = int(phase_str)
    masses = [float(x) for x in mass_str.split()]
    print(f"\nCPU Phase {phase_idx} masses: {masses}")

# GPU masses
gpu_mass_pattern = r"GPU DEBUG: Phase (\d+).*?masses: \[(.*?)\]"
gpu_mass_matches = re.findall(gpu_mass_pattern, gpu_text)
for phase_str, mass_str in gpu_mass_matches[:2]:
    phase_idx = int(phase_str)
    masses = [float(x) for x in mass_str.split(',')]
    print(f"GPU Phase {phase_idx} masses: {masses}")