#!/usr/bin/env python
"""Comprehensive trace of CPU vs GPU divergence for ternary system."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

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
print("COMPREHENSIVE TERNARY DIVERGENCE TRACE")
print("=" * 80)
print(f"Components: {comps}")
print(f"Conditions: T={conditions[v.T]}K, X(CU)={conditions[v.X('CU')]}, X(FE)={conditions[v.X('FE')]}, X(AL)={1-conditions[v.X('CU')]-conditions[v.X('FE')]}")
print()

# Run CPU with verbose output
print("=" * 80)
print("CPU CALCULATION")
print("=" * 80)

cpu_captured = io.StringIO()
with redirect_stdout(cpu_captured):
    result_cpu = equilibrium(dbf, comps, phases, conditions, verbose=True, calc_opts={'pdens': 50})

cpu_output = cpu_captured.getvalue()
cpu_gm = float(result_cpu.GM.values)

# Extract key CPU values
print("\nCPU Key Values:")
print(f"  Final GM: {cpu_gm:.6f} J/mol")

# Look for phase amounts and compositions in CPU output
cpu_phases = []
for line in cpu_output.split('\n'):
    if 'Phase' in line and 'NP=' in line:
        cpu_phases.append(line.strip())

if cpu_phases:
    print("  Phase amounts:")
    for phase in cpu_phases[:3]:  # Show first 3
        print(f"    {phase}")

# Run GPU with verbose output
print("\n" + "=" * 80)
print("GPU CALCULATION")
print("=" * 80)

gpu_captured = io.StringIO()
gpu_err_captured = io.StringIO()
with redirect_stdout(gpu_captured), redirect_stderr(gpu_err_captured):
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True, calc_opts={'pdens': 50})

gpu_output = gpu_captured.getvalue()
gpu_err = gpu_err_captured.getvalue()
gpu_gm = float(result_gpu.GM.values)

# Extract key GPU values
print("\nGPU Key Values:")
print(f"  Final GM: {gpu_gm:.6f} J/mol")

# Look for initial mole fractions
print("\n  Initial mole fractions:")
for line in gpu_output.split('\n'):
    if 'Thread 0 mole fractions:' in line:
        print(f"    {line.strip()}")
        break

# Look for final converged values
print("\n  Final converged state:")
converged_lines = []
final_gm_lines = []
for line in gpu_output.split('\n'):
    if 'converged = True' in line or 'converged = False' in line:
        converged_lines.append(line.strip())
    if 'final_gm_calc' in line and '=' in line:
        final_gm_lines.append(line.strip())

if converged_lines:
    print(f"    {converged_lines[-1]}")
if final_gm_lines:
    print(f"    {final_gm_lines[-1]}")

# Look for phase amounts at convergence
print("\n  GPU Phase amounts at convergence:")
phase_amt_lines = []
for i, line in enumerate(gpu_output.split('\n')):
    if 'phase_amt' in line and '=' in line and 'Phase' in line:
        phase_amt_lines.append(line.strip())

for line in phase_amt_lines[-5:]:  # Show last 5
    print(f"    {line}")

# Compare initial conditions setup
print("\n" + "=" * 80)
print("INITIAL CONDITIONS COMPARISON")
print("=" * 80)

# Extract constraint setup from GPU
print("\nGPU Constraint Setup:")
for line in gpu_output.split('\n'):
    if 'Coefficients:' in line and '[' in line:
        print(f"  {line.strip()}")
    if 'RHS:' in line and '[' in line:
        print(f"  {line.strip()}")
        break

# Extract first iteration values
print("\n" + "=" * 80)
print("FIRST ITERATION COMPARISON")
print("=" * 80)

print("\nGPU Iteration 0 values:")
in_iter_0 = False
iter_0_lines = []
for line in gpu_output.split('\n'):
    if 'Iteration 0/' in line:
        in_iter_0 = True
    elif 'Iteration 1/' in line:
        in_iter_0 = False
        break
    elif in_iter_0 and ('phase_' in line.lower() or 'energy' in line or 'NP=' in line):
        iter_0_lines.append(line.strip())

for line in iter_0_lines[:10]:  # Show first 10 relevant lines
    print(f"  {line}")

# Look for specific divergence points
print("\n" + "=" * 80)
print("DIVERGENCE ANALYSIS")
print("=" * 80)

print(f"\nFinal GM Difference: {abs(cpu_gm - gpu_gm):.6f} J/mol")

# Check phase identification
print("\nPhase identification check:")
gpu_phase_lines = []
for line in gpu_output.split('\n'):
    if 'Phase 0' in line and 'energy' in line:
        gpu_phase_lines.append(line.strip())
    elif 'Phase 1' in line and 'energy' in line:
        gpu_phase_lines.append(line.strip())

for line in gpu_phase_lines[:4]:
    print(f"  {line}")

# Check for any error messages or warnings
print("\nGPU Warnings/Errors:")
error_keywords = ['ERROR', 'WARNING', 'Failed', 'Invalid', 'NaN', 'Inf']
gpu_errors = []
for line in (gpu_output + gpu_err).split('\n'):
    for keyword in error_keywords:
        if keyword in line:
            gpu_errors.append(line.strip())
            break

if gpu_errors:
    for error in gpu_errors[:5]:
        print(f"  {error}")
else:
    print("  No errors detected")

# Check mass balance
print("\nMass balance check:")
mass_residual_lines = []
for line in gpu_output.split('\n'):
    if 'mass_residual' in line and ':' in line:
        mass_residual_lines.append(line.strip())

if mass_residual_lines:
    print(f"  Initial: {mass_residual_lines[0] if mass_residual_lines else 'N/A'}")
    print(f"  Final: {mass_residual_lines[-1] if mass_residual_lines else 'N/A'}")

# Check chemical potentials
print("\nChemical potentials comparison:")
gpu_mu_lines = []
for line in gpu_output.split('\n'):
    if 'chemical_potentials:' in line and '[' in line:
        gpu_mu_lines.append(line.strip())

if gpu_mu_lines:
    print(f"  Initial GPU: {gpu_mu_lines[0] if gpu_mu_lines else 'N/A'}")
    print(f"  Final GPU: {gpu_mu_lines[-1] if len(gpu_mu_lines) > 1 else 'N/A'}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"CPU GM: {cpu_gm:.2f} J/mol")
print(f"GPU GM: {gpu_gm:.2f} J/mol")
print(f"Difference: {abs(cpu_gm - gpu_gm):.2f} J/mol")

if abs(cpu_gm - gpu_gm) < 1.0:
    print("✓ Results match within tolerance")
else:
    print("✗ Significant divergence detected")
    print("\nPotential issues to investigate:")
    print("  1. Check if initial mole fractions are correct")
    print("  2. Verify constraint coefficients match CPU")
    print("  3. Check phase identification and amounts")
    print("  4. Verify chemical potential calculations")
    print("  5. Check for numerical precision issues in solver")