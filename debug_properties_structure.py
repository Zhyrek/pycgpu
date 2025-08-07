#!/usr/bin/env python
"""Debug the structure of properties from starting_point."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
from pycalphad.core.workspace import Workspace
from pycalphad.core.starting_point import starting_point

# Load database
dbf = Database('important_tests/AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
phases = filter_phases(dbf, comps)

# Create conditions
conditions = {
    v.X('BI'): [0.1, 0.2, 0.3],
    v.T: [400, 500],
    v.P: 101325
}

# Create workspace
wks = Workspace(database=dbf, components=comps, phases=phases, conditions=conditions)

# Get properties from starting_point
properties = wks.eq

print("Properties structure:")
print(f"Type: {type(properties)}")
print(f"Shape: {getattr(properties, 'shape', 'No shape attribute')}")
print(f"Attributes: {[a for a in dir(properties) if not a.startswith('_')]}")

# Check specific properties
for attr in ['GM', 'MU', 'NP', 'Phase', 'X', 'Y']:
    if hasattr(properties, attr):
        prop = getattr(properties, attr)
        print(f"\n{attr}:")
        print(f"  Type: {type(prop)}")
        print(f"  Shape: {getattr(prop, 'shape', 'No shape')}")
        if hasattr(prop, 'values'):
            print(f"  Values shape: {prop.values.shape}")
            print(f"  Values sample: {prop.values.flat[:6]}")
        elif hasattr(prop, '__len__'):
            print(f"  Length: {len(prop)}")
            print(f"  Sample: {prop[:min(3, len(prop))]}")

# Check how to properly index
print("\n\nIndexing test:")
gm_array = properties.GM
print(f"GM shape: {gm_array.shape}")
print(f"Expected conditions: 3 x 2 = 6")

# Test unravel_index
for cond_idx in range(6):
    multi_idx = np.unravel_index(cond_idx, gm_array.shape)
    print(f"\nCondition {cond_idx} -> multi_idx {multi_idx}")
    
    # Check phases
    phase_val = properties.Phase[multi_idx]
    print(f"  Phases: {phase_val}")
    
    # Count non-empty phases
    num_phases = np.sum(phase_val != '')
    print(f"  Num phases: {num_phases}")
    
    # Check NP
    np_val = properties.NP[multi_idx]
    print(f"  NP: {np_val}")
    
    # Check if any phase amounts are valid
    valid_amounts = ~np.isnan(np_val)
    print(f"  Valid amounts: {valid_amounts}")
    
    # MU values
    mu_val = properties.MU[multi_idx]
    print(f"  MU: {mu_val}")