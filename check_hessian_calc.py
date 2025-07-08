#!/usr/bin/env python
"""Check what Hessian values the CPU and GPU calculate."""

import numpy as np
from pycalphad import Database, calculate, variables as v

# Simple test
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']

# Calculate at specific composition
calc_result = calculate(dbf, comps, phases, T=1000, P=101325, 
                       points={'BCC_A2': [[0.5, 0.5]]}, 
                       output='GM')

print(f"Calculated GM at Y(NB)=0.5, Y(TI)=0.5: {calc_result.GM.values}")

# Now let's check what the equilibrium calculation shows
from pycalphad import equilibrium

print("\n" + "="*60)
print("Running equilibrium with mole fraction constraint...")
conditions = {v.T: 1000, v.P: 101325, v.X('TI'): 0.5}

# First CPU
print("\nCPU calculation:")
cpu_eq = equilibrium(dbf, comps, phases, conditions, verbose=False)
print(f"Result: X(TI) = {cpu_eq.X.sel(component='TI').values.flatten()[0]:.6f}")
print(f"Phase amounts: {cpu_eq.NP.values.flatten()}")

# Then GPU
print("\nGPU calculation:")
gpu_eq = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
print(f"Result: X(TI) = {gpu_eq.X.sel(component='TI').values.flatten()[0]:.6f}")
print(f"Phase amounts: {gpu_eq.NP.values.flatten()}")