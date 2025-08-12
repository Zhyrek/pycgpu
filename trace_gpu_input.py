#!/usr/bin/env python
"""Trace what GPU gets as input and compare to expected values."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

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
print("EXPECTED VALUES")
print("=" * 80)
print(f"Components: {comps}")
print(f"Non-VA components: {[c for c in comps if c != 'VA']}")
print(f"Number of components: {len(comps)} (including VA)")
print(f"Number of non-VA components: {len([c for c in comps if c != 'VA'])}")
print(f"\nConditions:")
print(f"  T = {conditions[v.T]} K")
print(f"  P = {conditions[v.P]} Pa")
print(f"  X(CU) = {conditions[v.X('CU')]}")
print(f"  X(FE) = {conditions[v.X('FE')]}")
print(f"  X(AL) = {1.0 - conditions[v.X('CU')] - conditions[v.X('FE')]} (implicit)")
print(f"\nExpected mole fractions array:")
print(f"  [X(AL), X(CU), X(FE), X(VA)] = [0.5, 0.2, 0.3, 0.0]")

print("\n" + "=" * 80)
print("GPU DEBUG OUTPUT (Key lines)")
print("=" * 80)

# Run GPU with verbose and extract key lines
import sys
import io
from contextlib import redirect_stdout

captured = io.StringIO()
with redirect_stdout(captured):
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True, calc_opts={'pdens': 50})

output = captured.getvalue()

# Extract and display key debug lines
key_patterns = [
    ('Components setup:', 'num_components'),
    ('Mole fractions:', 'mole_fractions'),
    ('Constraints:', 'constraint'),
    ('Chemical potentials:', 'chemical_potentials'),
    ('Thread 0 mole fractions:', 'Thread 0 mole fractions'),
    ('Coefficients:', 'Coefficients'),
    ('RHS:', 'RHS'),
    ('Phase amounts:', 'phase_amt'),
    ('Converged:', 'converged'),
    ('Final GM:', 'final_gm_calc')
]

for section, pattern in key_patterns:
    print(f"\n{section}")
    count = 0
    for line in output.split('\n'):
        if pattern.lower() in line.lower():
            print(f"  {line.strip()}")
            count += 1
            if count >= 5:  # Limit output per section
                break

# Compare final results
print("\n" + "=" * 80)
print("RESULTS COMPARISON")
print("=" * 80)

result_cpu = equilibrium(dbf, comps, phases, conditions, calc_opts={'pdens': 50})
cpu_gm = float(result_cpu.GM.values)
gpu_gm = float(result_gpu.GM.values)

print(f"CPU GM: {cpu_gm:.2f} J/mol")
print(f"GPU GM: {gpu_gm:.2f} J/mol") 
print(f"Difference: {abs(cpu_gm - gpu_gm):.2f} J/mol")

# Check for obvious issues
print("\n" + "=" * 80)
print("POTENTIAL ISSUES")
print("=" * 80)

# Look for wrong mole fraction values
wrong_mf_found = False
for line in output.split('\n'):
    if 'mole_fractions: [0.800000' in line:
        print("✗ Found incorrect mole fractions: [0.8, 0.1, 0.1, 0.0]")
        print("  Expected: [0.5, 0.2, 0.3, 0.0]")
        print(f"  Line: {line.strip()}")
        wrong_mf_found = True
        break

if not wrong_mf_found:
    print("✓ Mole fractions appear to be set correctly")

# Check convergence
if 'converged = False' in output:
    print("✗ GPU solver did not converge")
elif 'converged = True' in output:
    print("✓ GPU solver converged")
else:
    print("? Convergence status unclear")