#!/usr/bin/env python
"""Check vertex names in equilibrium result."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

# Load database
db = Database('NbTi.tdb')
components = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'HCP_A3']

# Test conditions
conditions = {
    v.T: 500,
    v.P: 101325,
    v.X('TI'): 0.1,
    v.N: 1
}

# Run equilibrium
eq = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 50}, verbose=False)

print("Available dimensions in equilibrium result:")
for dim in eq.dims:
    print(f"  {dim}: {eq.dims[dim]}")

print("\nVertex values:")
print(eq.coords['vertex'].values)

print("\nY array shape:", eq.Y.shape)
print("\nY data for first point:")
print(eq.Y.values[0])

# Get composition data properly
print("\nGetting composition data:")
for i, phase in enumerate(eq.Phase.values[0]):
    if eq.NP.values[0][i] > 0:
        print(f"\nPhase {i}: {phase}")
        print(f"  Amount: {eq.NP.values[0][i]}")
        print(f"  Y values: {eq.Y.values[0][i]}")
        
        # Get Ti composition
        ti_idx = list(eq.coords['component'].values).index('TI')
        x_ti = eq.Y.values[0][i][ti_idx]
        print(f"  X(TI) = {x_ti}")