#!/usr/bin/env python
"""Debug how pycalphad structures multi-condition arrays for ternary systems."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

# Load database
dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']
phases = ['LIQUID', 'FCC_A1']

# Define conditions with ranges
conditions = {
    v.X('CU'): (0.1, 0.5, 0.1),  # 0.1 to 0.5 in 0.1 increments -> [0.1, 0.2, 0.3, 0.4, 0.5] = 5 values
    v.X('FE'): (0.1, 0.4, 0.1),  # 0.1 to 0.4 in 0.1 increments -> [0.1, 0.2, 0.3, 0.4] = 4 values  
    v.T: (600, 1200, 200),       # 600 to 1200 in 200 increments -> [600, 800, 1000, 1200] = 4 values
    v.P: 101325
}

# Analyze how conditions are structured
print("Analyzing condition structure:")
print(f"  X(CU) range: {conditions[v.X('CU')]}")
print(f"  X(FE) range: {conditions[v.X('FE')]}")
print(f"  T range: {conditions[v.T]}")

# Check what arrays are created
x_cu_values = np.arange(0.1, 0.5 + 0.1, 0.1)
x_fe_values = np.arange(0.1, 0.4 + 0.1, 0.1)
temp_values = np.arange(600, 1200 + 200, 200)

print(f"\nExpanded arrays:")
print(f"  X(CU) values: {x_cu_values} (length={len(x_cu_values)})")
print(f"  X(FE) values: {x_fe_values} (length={len(x_fe_values)})")
print(f"  T values: {temp_values} (length={len(temp_values)})")

# Calculate total conditions
total_conditions = len(x_cu_values) * len(x_fe_values) * len(temp_values)
print(f"\nTotal condition combinations: {total_conditions}")

# Check how equilibrium expands these
from pycalphad.core.workspace import Workspace

# Create a workspace to see how it structures conditions
wks = Workspace(dbf, comps, phases, conditions)

print(f"\nWorkspace conditions:")
for key, value in wks.conditions.items():
    if hasattr(value, '__len__'):
        print(f"  {key}: array of length {len(value)}")
        if len(value) <= 10:
            print(f"    Values: {value}")
    else:
        print(f"  {key}: {value}")

# Try to understand the condition structure by looking at internal arrays
print(f"\nAnalyzing condition arrays after workspace creation:")
for key, value in wks.conditions.items():
    if hasattr(key, 'species') and key.species != 'VA':
        arr = np.asarray(value).flatten()
        print(f"  {key}: shape={arr.shape}, first 10 values: {arr[:10]}")
    elif key == v.T:
        arr = np.asarray(value).flatten()
        print(f"  {key}: shape={arr.shape}, first 10 values: {arr[:10]}")