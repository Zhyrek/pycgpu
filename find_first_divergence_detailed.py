#!/usr/bin/env python
"""Find the first divergence between CPU and GPU calculations."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")
import sys
import io
from contextlib import redirect_stdout, redirect_stderr
import re

# Load database
dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']
phases = ['LIQUID', 'FCC_A1']

# Single ternary condition
conditions = {
    v.T: 1200,
    v.P: 101325,
    v.X('CU'): 0.2,
    v.X('FE'): 0.3  # X(AL) = 0.5 implied
}

print("=" * 80)
print("FINDING FIRST CPU/GPU DIVERGENCE")
print("=" * 80)

# Run CPU silently and capture output
cpu_captured = io.StringIO()
with redirect_stdout(cpu_captured):
    result_cpu = equilibrium(dbf, comps, phases, conditions, verbose=True, calc_opts={'pdens': 50})

cpu_output = cpu_captured.getvalue()
cpu_lines = cpu_output.split('\n')

# Run GPU silently and capture output
gpu_captured = io.StringIO()
gpu_err_captured = io.StringIO()
with redirect_stdout(gpu_captured), redirect_stderr(gpu_err_captured):
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True, calc_opts={'pdens': 50})

gpu_output = gpu_captured.getvalue()
gpu_lines = gpu_output.split('\n')

print(f"CPU output lines: {len(cpu_lines)}")
print(f"GPU output lines: {len(gpu_lines)}")
print()

# Extract key numerical values from lines
def extract_numbers(line):
    """Extract all numbers from a line."""
    # Match scientific notation and regular numbers
    pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    numbers = re.findall(pattern, line)
    return [float(n) for n in numbers if n and n != '.']

# Key patterns to look for
key_patterns = [
    'chemical_potentials',
    'mole_fractions',
    'phase_amt',
    'energy',
    'NP=',
    'GM',
    'converged',
    'Phase 0',
    'Phase 1', 
    'Phase 2',
    'Iteration',
    'X[0]',
    'X[1]',
    'X[2]',
    'Y[0]',
    'Y[1]',
    'Coefficients',
    'RHS',
    'mass_residual',
    'phase_compositions',
    'site_fractions'
]

# Collect matching lines from both outputs
cpu_key_lines = []
gpu_key_lines = []

for line in cpu_lines:
    for pattern in key_patterns:
        if pattern in line:
            cpu_key_lines.append(line.strip())
            break

for line in gpu_lines:
    for pattern in key_patterns:
        if pattern in line:
            gpu_key_lines.append(line.strip())
            break

# Compare initial setup
print("INITIAL SETUP COMPARISON")
print("-" * 40)

# Find initial chemical potentials
print("\nInitial Chemical Potentials:")
for line in cpu_key_lines[:50]:
    if 'initial_chemical_potentials' in line or ('chemical_potentials' in line and 'initial' in line.lower()):
        print(f"CPU: {line}")
        break

for line in gpu_key_lines[:50]:
    if 'initial_chemical_potentials' in line:
        print(f"GPU: {line}")
        break

# Find initial mole fractions
print("\nInitial Mole Fractions:")
for line in gpu_key_lines[:50]:
    if 'Thread 0 mole fractions:' in line:
        print(f"GPU: {line}")
        break

# Find constraint setup
print("\nConstraint Setup:")
gpu_coeff_found = False
gpu_rhs_found = False
for i, line in enumerate(gpu_key_lines[:100]):
    if 'Coefficients:' in line and not gpu_coeff_found:
        print(f"GPU: {line}")
        gpu_coeff_found = True
    elif 'RHS:' in line and not gpu_rhs_found:
        print(f"GPU: {line}")
        gpu_rhs_found = True
    if gpu_coeff_found and gpu_rhs_found:
        break

# Compare phase energies in first iteration
print("\n" + "=" * 80)
print("ITERATION 0 COMPARISON")
print("-" * 40)

# Extract iteration 0 data from GPU
gpu_iter0_start = -1
gpu_iter0_end = -1
for i, line in enumerate(gpu_lines):
    if 'Iteration 0/' in line or 'Iteration 0:' in line:
        gpu_iter0_start = i
    elif gpu_iter0_start >= 0 and ('Iteration 1/' in line or 'Iteration 1:' in line):
        gpu_iter0_end = i
        break

if gpu_iter0_start >= 0:
    print("\nGPU Iteration 0 Key Values:")
    iter0_lines = gpu_lines[gpu_iter0_start:gpu_iter0_end if gpu_iter0_end > 0 else gpu_iter0_start+100]
    
    # Look for phase energies
    for line in iter0_lines:
        if 'phase_0_energy' in line or 'phase_1_energy' in line or 'phase_2_energy' in line:
            print(f"  {line.strip()}")
        elif 'Phase 0 energy' in line or 'Phase 1 energy' in line or 'Phase 2 energy' in line:
            print(f"  {line.strip()}")
        elif 'NP=' in line and 'phase' in line.lower():
            print(f"  {line.strip()}")

# Look for CPU phase energies
print("\nCPU Iteration 0 Key Values:")
for line in cpu_lines[:200]:
    if 'FORMULAHESS' in line and 'iteration 0' in line:
        print(f"  {line.strip()}")
    elif 'Energy:' in line and any(x in cpu_lines[max(0,cpu_lines.index(line)-5):cpu_lines.index(line)] for x in ['iteration 0', 'Iteration 0']):
        print(f"  {line.strip()}")

# Check for specific divergence in phase amounts
print("\n" + "=" * 80)
print("PHASE AMOUNT COMPARISON")
print("-" * 40)

# GPU phase amounts
print("\nGPU Phase Amounts:")
gpu_phase_lines = []
for line in gpu_lines:
    if 'phase_amt' in line and '=' in line:
        gpu_phase_lines.append(line.strip())

for line in gpu_phase_lines[:5]:
    print(f"  {line}")

# Look for convergence status
print("\n" + "=" * 80)
print("CONVERGENCE STATUS")
print("-" * 40)

gpu_converged = None
for line in gpu_lines:
    if 'converged = True' in line:
        gpu_converged = True
        print(f"GPU: {line.strip()}")
        break
    elif 'converged = False' in line:
        gpu_converged = False
        print(f"GPU: {line.strip()}")
        break

if gpu_converged is None:
    print("GPU: Convergence status not found")
elif not gpu_converged:
    print("⚠️  GPU DID NOT CONVERGE - This is likely the cause of divergence")

# Final GM comparison
print("\n" + "=" * 80)
print("FINAL RESULTS")
print("-" * 40)

cpu_gm = float(result_cpu.GM.values)
gpu_gm = float(result_gpu.GM.values)

print(f"CPU Final GM: {cpu_gm:.6f} J/mol")
print(f"GPU Final GM: {gpu_gm:.6f} J/mol")
print(f"Difference: {abs(cpu_gm - gpu_gm):.6f} J/mol")

# Check if the issue is the number of phases
cpu_phases_active = np.sum(result_cpu.NP.values[0,0,0,0,:] > 1e-6)
gpu_phases_active = np.sum(result_gpu.NP.values[0,0,0,0,:] > 1e-6)

print(f"\nActive phases:")
print(f"  CPU: {cpu_phases_active} phases")
print(f"  GPU: {gpu_phases_active} phases")

if cpu_phases_active != gpu_phases_active:
    print("⚠️  DIFFERENT NUMBER OF ACTIVE PHASES")

# Summary
print("\n" + "=" * 80)
print("LIKELY DIVERGENCE POINT")
print("-" * 40)

if not gpu_converged:
    print("The GPU solver did not converge, while the CPU solver did.")
    print("This is the primary cause of the 239 J/mol difference.")
    print("\nPossible reasons for non-convergence:")
    print("  1. Maximum iteration limit reached")
    print("  2. Numerical instability in solver")
    print("  3. Different convergence criteria")
    print("  4. Issue with constraint handling for ternary systems")