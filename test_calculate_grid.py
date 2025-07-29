#!/usr/bin/env python
"""Check what grid points calculate generates."""

import numpy as np
from pycalphad import Database, calculate, variables as v

# Load database
db = Database('NbTi.tdb')
components = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'HCP_A3']

# Test conditions
T = 500
P = 101325

print("Checking calculate grid generation...")
print("="*80)

# Run calculate with explicit points
print("\nRunning calculate with points={'BCC_A2': 50, 'HCP_A3': 50}...")
calc_result = calculate(db, components, phases, T=T, P=P, 
                       output='GM', points={'BCC_A2': 50, 'HCP_A3': 50})

print(f"\nResult shape: {calc_result.GM.shape}")
print(f"Dimensions: {dict(calc_result.dims)}")

# Check phases
print(f"\nPhases in result: {np.unique(calc_result.Phase.values)}")

# Count points per phase
for phase in ['BCC_A2', 'HCP_A3']:
    mask = calc_result.Phase.values == phase
    count = np.sum(mask)
    print(f"{phase}: {count} points")

# Check if we have Y data
if 'Y' in calc_result:
    print(f"\nY array shape: {calc_result.Y.shape}")
    print(f"Y dimensions: {list(calc_result.Y.dims)}")
else:
    print("\nNo Y data in calculate result")

# Try different approach
print("\n" + "="*80)
print("Trying calculate without specifying points...")
calc_result2 = calculate(db, components, phases, T=T, P=P, output='GM')

print(f"\nResult shape: {calc_result2.GM.shape}")
print(f"Total points: {calc_result2.GM.size}")

# Count points per phase
for phase in ['BCC_A2', 'HCP_A3']:
    mask = calc_result2.Phase.values == phase
    count = np.sum(mask)
    print(f"{phase}: {count} points")

print("\n" + "="*80)
print("Summary:")
print("- Calculate is only generating 1 point per phase")
print("- This prevents finding the miscibility gap")
print("- The issue may be in the grid generation or pdens parameter")
print("="*80)