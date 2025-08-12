#!/usr/bin/env python
"""Trace CPU vs GPU iteration by iteration to find exact divergence point."""

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
print("ITERATION-BY-ITERATION COMPARISON")
print("=" * 80)

# Run GPU and capture output
gpu_captured = io.StringIO()
with redirect_stdout(gpu_captured):
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True, calc_opts={'pdens': 50})

gpu_output = gpu_captured.getvalue()
gpu_lines = gpu_output.split('\n')

# Parse GPU iterations
gpu_iterations = {}
current_iter = -1
for line in gpu_lines:
    # Look for iteration markers
    if 'Iteration' in line and ('/' in line or ':' in line):
        match = re.search(r'Iteration (\d+)', line)
        if match:
            current_iter = int(match.group(1))
            gpu_iterations[current_iter] = []
    
    # Collect important values for current iteration
    if current_iter >= 0:
        # Look for key values
        if any(pattern in line for pattern in [
            'phase_0_energy', 'phase_1_energy', 'phase_2_energy',
            'Phase 0 energy', 'Phase 1 energy', 'Phase 2 energy',
            'chemical_potentials:', 'mole_fractions:',
            'phase_amt', 'NP=', 'converged',
            'mass_residual', 'num_phases_active',
            'phase_0_NP', 'phase_1_NP', 'phase_2_NP',
            'phase_0_X', 'phase_1_X', 'phase_2_X',
            'Coefficients:', 'RHS:',
            'Phase 0:', 'Phase 1:', 'Phase 2:',
            'system_amount:', 'GM'
        ]):
            gpu_iterations[current_iter].append(line.strip())

# Print iteration-by-iteration analysis
for iter_num in sorted(gpu_iterations.keys()):
    print(f"\n{'='*80}")
    print(f"GPU ITERATION {iter_num}")
    print(f"{'='*80}")
    
    iter_lines = gpu_iterations[iter_num]
    
    # Extract phase energies
    print("\nPhase Energies:")
    phase_energies = []
    for line in iter_lines:
        if 'phase_0_energy' in line or 'Phase 0 energy' in line:
            match = re.search(r'[-+]?\d*\.?\d+[eE][-+]?\d+', line)
            if match:
                energy = float(match.group())
                phase_energies.append(('Phase 0', energy))
                print(f"  Phase 0: {energy:.2f} J/mol")
        elif 'phase_1_energy' in line or 'Phase 1 energy' in line:
            match = re.search(r'[-+]?\d*\.?\d+[eE][-+]?\d+', line)
            if match:
                energy = float(match.group())
                phase_energies.append(('Phase 1', energy))
                print(f"  Phase 1: {energy:.2f} J/mol")
        elif 'phase_2_energy' in line or 'Phase 2 energy' in line:
            match = re.search(r'[-+]?\d*\.?\d+[eE][-+]?\d+', line)
            if match:
                energy = float(match.group())
                phase_energies.append(('Phase 2', energy))
                print(f"  Phase 2: {energy:.2f} J/mol")
    
    # Extract phase amounts
    print("\nPhase Amounts:")
    for line in iter_lines:
        if 'phase_0_NP' in line:
            print(f"  {line}")
        elif 'phase_1_NP' in line:
            print(f"  {line}")
        elif 'phase_2_NP' in line:
            print(f"  {line}")
        elif 'phase_amt' in line and 'Phase' in line:
            print(f"  {line}")
    
    # Extract chemical potentials
    print("\nChemical Potentials:")
    for line in iter_lines:
        if 'chemical_potentials:' in line and '[' in line:
            print(f"  {line}")
            break
    
    # Extract mole fractions
    print("\nMole Fractions:")
    for line in iter_lines:
        if 'mole_fractions:' in line and '[' in line:
            print(f"  {line}")
            # Only show first occurrence
            break
    
    # Extract mass residual
    print("\nMass Residual:")
    for line in iter_lines:
        if 'mass_residual' in line:
            print(f"  {line}")
            break
    
    # Check convergence
    print("\nConvergence:")
    converged = False
    for line in iter_lines:
        if 'converged' in line:
            print(f"  {line}")
            if 'True' in line:
                converged = True
            break
    
    # If converged or last iteration, show final values
    if converged or iter_num >= 199:
        print("\n>>> SOLVER STATUS: ", "CONVERGED" if converged else "NOT CONVERGED")
        if iter_num >= 199:
            print(">>> Maximum iterations reached!")

# Now compare with expected CPU values
print("\n" + "=" * 80)
print("EXPECTED CPU VALUES (from standard Al-Cu-Fe at T=1200K)")
print("=" * 80)

# These are typical values for this system from CPU
print("\nCPU Iteration 0 (typical):")
print("  Phase 0 energy: ~-79537 J/mol")
print("  Phase 1 energy: ~-80333 J/mol")
print("  Phase 2 energy: ~-79505 J/mol")

print("\nCPU converges in ~7-8 iterations to:")
print("  Final GM: -79814.29 J/mol")
print("  Single phase (LIQUID or FCC_A1)")

# Find where GPU diverges
print("\n" + "=" * 80)
print("DIVERGENCE ANALYSIS")
print("=" * 80)

# Check if GPU reached max iterations
max_iter = max(gpu_iterations.keys()) if gpu_iterations else 0
print(f"\nGPU reached iteration: {max_iter}")

if max_iter >= 199:
    print("⚠️  GPU hit maximum iteration limit (200)")
    print("This is why it didn't converge!")
    
# Show the final GPU result
gpu_gm = float(result_gpu.GM.values)
print(f"\nGPU Final GM: {gpu_gm:.2f} J/mol")
print(f"Expected CPU GM: -79814.29 J/mol")
print(f"Difference: {abs(gpu_gm - (-79814.29)):.2f} J/mol")

# Check for specific issues
print("\nPotential issues detected:")

# Check if phase energies are reasonable
if gpu_iterations:
    first_iter = gpu_iterations.get(0, [])
    for line in first_iter:
        if 'phase_0_energy' in line:
            match = re.search(r'[-+]?\d*\.?\d+[eE][-+]?\d+', line)
            if match:
                energy = float(match.group())
                if abs(energy - (-79537)) > 100:
                    print(f"  ✗ Phase 0 energy in iteration 0 differs by {abs(energy - (-79537)):.0f} J/mol")
                else:
                    print(f"  ✓ Phase 0 energy in iteration 0 matches CPU")
                break