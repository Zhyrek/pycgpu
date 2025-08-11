#!/usr/bin/env python
"""Test how conditions are laid out for ternary systems."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

# Load database
dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']
phases = ['LIQUID', 'FCC_A1']

# Define conditions
conditions = {
    v.X('CU'): (0.1, 0.5, 0.1),  
    v.X('FE'): (0.1, 0.4, 0.1),  
    v.T: (600, 1200, 200),       
    v.P: 101325
}

print("Testing equilibrium with ranges:")
result = equilibrium(dbf, comps, phases, conditions, gpu=False, calc_opts={'pdens': 60})

print(f"\nResult dataset dimensions: {result.dims}")
print(f"Result dataset coordinates:")
for coord in result.coords:
    if coord in ['T', 'X_CU', 'X_FE']:
        vals = result.coords[coord].values
        print(f"  {coord}: {vals} (length={len(vals)})")

# The key insight: equilibrium creates a CARTESIAN PRODUCT of all conditions
print(f"\nShape of GM: {result.GM.shape}")
print(f"Total conditions calculated: {result.GM.size}")

# Check how conditions are indexed
print("\nFirst few condition combinations:")
for i in range(min(10, result.GM.size)):
    # Get indices in each dimension
    temp_idx = i // (len(result.coords['X_CU']) * len(result.coords['X_FE']))
    remainder = i % (len(result.coords['X_CU']) * len(result.coords['X_FE']))
    cu_idx = remainder // len(result.coords['X_FE'])
    fe_idx = remainder % len(result.coords['X_FE'])
    
    if temp_idx < len(result.coords['T']):
        t = result.coords['T'].values[temp_idx]
        x_cu = result.coords['X_CU'].values[cu_idx] 
        x_fe = result.coords['X_FE'].values[fe_idx]
        print(f"  Condition {i}: T={t:.0f}K, X(CU)={x_cu:.1f}, X(FE)={x_fe:.1f}")