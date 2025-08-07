#!/usr/bin/env python
"""Debug MU extraction issue."""

import numpy as np
from pycalphad import Database, Workspace, calculate, variables as v
from pycalphad.core.starting_point import starting_point

# Load database
dbf = Database('AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']

# Test with 6 phases and identical conditions
phases = ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'AU2BI_HEX', 'RHOMBOHEDRAL_A7', 'BCC_A2']
conditions = {v.X('BI'): [0.3, 0.3], v.T: 600, v.P: 101325}

print("Testing MU extraction with 6 phases and 2 identical conditions")

# Create workspace
wks = Workspace(dbf, comps, phases, conditions)
state_variables = wks.phase_record_factory.state_variables

# Calculate grid - unpack conditions properly
cond_dict = {}
for key, val in conditions.items():
    cond_dict[str(key)] = val
    
grid = calculate(dbf, comps, phases, mode='numpy', output='GM',
                fake_points=True, parameters=wks.parameters.unwrap(),
                to_xarray=True, **cond_dict)

# Get starting point
props = starting_point(conditions, state_variables, wks.phase_record_factory, grid)

print(f"\nStarting point properties:")
print(f"MU shape: {props.MU.shape}")
print(f"MU dims: {props.MU.dims}")

# Check actual MU values
print(f"\nMU values array:")
print(props.MU.values)

# Simulate the GPU extraction logic
gm_array = props.GM.values
print(f"\ngm_array.shape = {gm_array.shape}")

for cond_idx in range(2):
    multi_idx = np.unravel_index(cond_idx, gm_array.shape)
    print(f"\nCondition {cond_idx}:")
    print(f"  multi_idx = {multi_idx}")
    
    # Direct array access
    mu_direct = props.MU.values[multi_idx]
    print(f"  props.MU.values[{multi_idx}] = {mu_direct}")
    
    # Using xarray indexing
    try:
        # Try using isel with the dimension names
        mu_xr = props.MU.isel(points=cond_idx).values
        print(f"  props.MU.isel(points={cond_idx}).values = {mu_xr}")
    except:
        print(f"  props.MU.isel failed")
        
print("\nConclusion: Are the MU values the same?")
mu0 = props.MU.values[(0,0,0,0)]
mu1 = props.MU.values[(0,0,0,1)]
print(f"MU[0] == MU[1]: {np.array_equal(mu0, mu1)}")
print(f"MU[0]: {mu0}")
print(f"MU[1]: {mu1}")