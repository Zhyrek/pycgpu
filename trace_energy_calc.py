#!/usr/bin/env python
"""Trace the exact energy calculation for X(TI)=0.9, T=600K."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, calculate, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Get the model for BCC_A2
from pycalphad import Model
mod = Model(dbf, comps, 'BCC_A2')

# The site fractions from the trace
y_nb = 0.101694915254245
y_ti = 0.898305084745755

# State variables
n = 1.0
p = 101325.0
t = 600.0

print("TESTING ENERGY CALCULATION DIRECTLY")
print("=" * 60)
print(f"Phase: BCC_A2")
print(f"State variables: N={n}, P={p}, T={t}")
print(f"Site fractions: Y(NB)={y_nb:.15f}, Y(TI)={y_ti:.15f}")

# Create the variable dict for energy evaluation
state_vars = {v.N: n, v.P: p, v.T: t}

# For BCC_A2, the site fraction variables are Y(BCC_A2,0,NB) and Y(BCC_A2,0,TI)
site_frac_vars = {
    v.Y('BCC_A2', 0, 'NB'): y_nb,
    v.Y('BCC_A2', 0, 'TI'): y_ti
}

# Combine all variables
all_vars = {**state_vars, **site_frac_vars}

# Evaluate the energy
gm_sympy = mod.GM.xreplace(all_vars)
energy_sympy = float(gm_sympy)

print(f"\nDirect SymPy evaluation: {energy_sympy:.15f} J/mol")

# Now let's check the compiled function
# Get the compiled energy function
energy_func = mod.energy

# The order of variables for the compiled function
# Based on the trace, it seems to be [N, P, T, Y(NB), Y(TI)]
dof = np.array([n, p, t, y_nb, y_ti])

print(f"\nDOF array for compiled function: {dof}")

# Call the compiled function
try:
    energy_compiled = energy_func(dof)
    print(f"Compiled function result: {energy_compiled:.15f} J/mol")
except Exception as e:
    print(f"Error calling compiled function: {e}")

# Compare with the values from the trace
cpu_energy = -19941.00371858601
gpu_energy = -19941.003719

print(f"\nComparison with trace values:")
print(f"  CPU energy: {cpu_energy:.15f}")
print(f"  GPU energy: {gpu_energy:.15f}")
print(f"  SymPy eval: {energy_sympy:.15f}")

# Check differences
print(f"\nDifferences:")
print(f"  GPU - CPU: {gpu_energy - cpu_energy:.15e}")
print(f"  SymPy - CPU: {energy_sympy - cpu_energy:.15e}")
print(f"  SymPy - GPU: {energy_sympy - gpu_energy:.15e}")

# Let's also check what happens with tiny perturbations
print("\n" + "=" * 60)
print("SENSITIVITY ANALYSIS")
print("=" * 60)

# Perturb Y(TI) by machine epsilon
epsilon = np.finfo(float).eps
y_ti_perturbed = y_ti * (1 + epsilon)

dof_perturbed = np.array([n, p, t, y_nb, y_ti_perturbed])
try:
    energy_perturbed = energy_func(dof_perturbed)
    delta = energy_perturbed - energy_compiled
    print(f"Energy change from {epsilon:.3e} relative change in Y(TI): {delta:.15e} J/mol")
except:
    pass

# Check the actual formulaobj code being used
print("\n" + "=" * 60)
print("FORMULA OBJECT INFO")
print("=" * 60)
print(f"Energy function type: {type(energy_func)}")
print(f"Model GM expression variables: {sorted(mod.GM.free_symbols, key=str)}")