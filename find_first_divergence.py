#!/usr/bin/env python
"""Find the first point where CPU and GPU calculations diverge."""

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

print("="*80)
print("FINDING FIRST DIVERGENCE POINT")
print("="*80)

# Run CPU calculation silently first
import io
import sys
cpu_output = io.StringIO()
old_stdout = sys.stdout
sys.stdout = cpu_output
reset_debug_session()
eq_cpu = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=True)
sys.stdout = old_stdout
cpu_text = cpu_output.getvalue()

# Run GPU calculation silently
gpu_output = io.StringIO()
sys.stdout = gpu_output
reset_debug_session()
eq_gpu = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=True)
sys.stdout = old_stdout
gpu_text = gpu_output.getvalue()

# Extract all numerical values we can compare
def extract_numerical_values(text, is_gpu=False):
    """Extract key numerical values from debug output."""
    values = {}
    
    # Initial phase amounts
    if is_gpu:
        pattern = r"GPU DEBUG: Processing phase (\d+).*?amount=([\d.e+-]+)"
        matches = re.findall(pattern, text)
        for phase_str, amount_str in matches:
            values[f'initial_phase_{phase_str}_amount'] = float(amount_str)
    else:
        pattern = r"\[CPU\] Phase (\d+) masses: \[\[([\d.e+-]+)"
        matches = re.findall(pattern, text)
        for phase_str, mass_str in matches:
            values[f'initial_phase_{phase_str}_amount'] = float(mass_str)
    
    # Gradient values (iteration 0)
    if is_gpu:
        pattern = r"GPU DEBUG: Phase (\d+) gradients before c_G calculation:.*?grad\[3\] = ([-\d.e+]+).*?grad\[4\] = ([-\d.e+]+)"
        matches = re.findall(pattern, text, re.DOTALL)
        for phase_str, grad3, grad4 in matches:
            if f'phase_{phase_str}_grad_Y_NB' not in values:  # Only first iteration
                values[f'phase_{phase_str}_grad_Y_NB'] = float(grad3)
                values[f'phase_{phase_str}_grad_Y_TI'] = float(grad4)
    else:
        pattern = r"\[CPU\] Phase (\d+) gradient \(iteration 0\): \[(.*?)\]"
        matches = re.findall(pattern, text)
        for phase_str, grad_str in matches:
            grads = [float(x) for x in grad_str.split()]
            if len(grads) >= 5:
                values[f'phase_{phase_str}_grad_Y_NB'] = grads[3]
                values[f'phase_{phase_str}_grad_Y_TI'] = grads[4]
    
    # Energy values
    if is_gpu:
        pattern = r"phase_(\d+)_energy: ([-\d.e+]+)"
        matches = re.findall(pattern, text)
        for phase_str, energy_str in matches[:2]:  # First iteration only
            values[f'phase_{phase_str}_energy'] = float(energy_str)
    else:
        pattern = r"phase_(\d+)_\w+_energy: ([-\d.e+]+)"
        matches = re.findall(pattern, text)
        for phase_str, energy_str in matches[:2]:  # First iteration only
            values[f'phase_{phase_str}_energy'] = float(energy_str)
    
    # c_G values
    if is_gpu:
        pattern = r"GPU DEBUG: Phase (\d+) c_G values.*?c_G\[0\] = ([-\d.e+]+).*?c_G\[1\] = ([-\d.e+]+)"
        matches = re.findall(pattern, text, re.DOTALL)
        for phase_str, cg0, cg1 in matches[:2]:  # First iteration
            values[f'phase_{phase_str}_c_G_0'] = float(cg0)
            values[f'phase_{phase_str}_c_G_1'] = float(cg1)
    else:
        pattern = r"\[CPU c_G DEBUG\] Phase (\d+).*?After c_G calc, c_G = \[([-\d.e+ ]+)\]"
        matches = re.findall(pattern, text, re.DOTALL)
        for phase_str, cg_str in matches[:2]:  # First iteration
            cg_vals = [float(x) for x in cg_str.split()]
            if len(cg_vals) >= 2:
                values[f'phase_{phase_str}_c_G_0'] = cg_vals[0]
                values[f'phase_{phase_str}_c_G_1'] = cg_vals[1]
    
    # Chemical potentials
    if is_gpu:
        pattern = r"Thread 0 chemical_potentials\[0\] = ([-\d.e+]+).*?Thread 0 chemical_potentials\[1\] = ([-\d.e+]+)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            values['chem_pot_0'] = float(match.group(1))
            values['chem_pot_1'] = float(match.group(2))
    else:
        pattern = r"chemical_potentials: \[([-\d.e+ ]+)\]"
        match = re.search(pattern, text)
        if match:
            chem_pots = [float(x) for x in match.group(1).split()]
            if len(chem_pots) >= 2:
                values['chem_pot_0'] = chem_pots[0]
                values['chem_pot_1'] = chem_pots[1]
    
    return values

# Extract values
cpu_values = extract_numerical_values(cpu_text, is_gpu=False)
gpu_values = extract_numerical_values(gpu_text, is_gpu=True)

# Compare values
print("\nCOMPARING EXTRACTED VALUES:")
print("-" * 80)

# Find all common keys
common_keys = sorted(set(cpu_values.keys()) & set(gpu_values.keys()))

max_key_len = max(len(k) for k in common_keys) if common_keys else 0

differences = []
for key in common_keys:
    cpu_val = cpu_values[key]
    gpu_val = gpu_values[key]
    abs_diff = abs(cpu_val - gpu_val)
    rel_diff = abs_diff / abs(cpu_val) if cpu_val != 0 else float('inf')
    
    differences.append((key, cpu_val, gpu_val, abs_diff, rel_diff))

# Sort by relative difference to find largest discrepancies
differences.sort(key=lambda x: x[4], reverse=True)

# Print all differences
print(f"{'Value':<{max_key_len}} | {'CPU':>20} | {'GPU':>20} | {'Abs Diff':>15} | {'Rel Diff %':>12}")
print("-" * (max_key_len + 71))

for key, cpu_val, gpu_val, abs_diff, rel_diff in differences:
    if abs_diff > 1e-15:  # Only show non-negligible differences
        print(f"{key:<{max_key_len}} | {cpu_val:>20.12e} | {gpu_val:>20.12e} | {abs_diff:>15.6e} | {rel_diff*100:>11.6f}%")

# Find the FIRST divergence point
print("\n\nFIRST DIVERGENCE ANALYSIS:")
print("-" * 80)

# Check if initial values match
if 'phase_0_energy' in cpu_values and 'phase_0_energy' in gpu_values:
    cpu_e0 = cpu_values['phase_0_energy']
    gpu_e0 = gpu_values['phase_0_energy']
    diff_e0 = abs(cpu_e0 - gpu_e0)
    print(f"\nInitial Phase 0 Energy:")
    print(f"  CPU: {cpu_e0:.15e}")
    print(f"  GPU: {gpu_e0:.15e}")
    print(f"  Difference: {diff_e0:.15e}")
    if diff_e0 > 1e-10:
        print("  *** DIVERGENCE FOUND IN INITIAL ENERGY CALCULATION ***")

# Check gradient values
print("\nGradient Values (should be identical with same inputs):")
for phase in [0, 1]:
    for comp in ['NB', 'TI']:
        key = f'phase_{phase}_grad_Y_{comp}'
        if key in cpu_values and key in gpu_values:
            cpu_grad = cpu_values[key]
            gpu_grad = gpu_values[key]
            diff = abs(cpu_grad - gpu_grad)
            print(f"\n  Phase {phase} Y({comp}) gradient:")
            print(f"    CPU: {cpu_grad:.15e}")
            print(f"    GPU: {gpu_grad:.15e}")
            print(f"    Difference: {diff:.15e}")
            if diff > 1e-10:
                print("    *** GRADIENT DIVERGENCE FOUND ***")

# Final energies
print("\n\nFINAL RESULTS:")
print("-" * 80)
print(f"CPU Final GM: {eq_cpu.GM.values.item():.15e}")
print(f"GPU Final GM: {eq_gpu.GM.values.item():.15e}")
print(f"Difference: {abs(eq_cpu.GM.values.item() - eq_gpu.GM.values.item()):.15e}")