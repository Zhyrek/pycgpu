#!/usr/bin/env python
"""Check parameter usage in the test case"""

import numpy as np
from pycalphad import Database, Workspace
import pycalphad.variables as v

# Simple binary system for testing
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'HCP_A3', 'LIQUID']
conds = {v.T: 1000, v.P: 101325, v.X('TI'): 0.4}

# Create workspace to inspect parameters
wks = Workspace(database=dbf, components=comps, phases=phases, 
                conditions=conds, models=None, parameters=None,
                calc_opts={'pdens': 10}, verbose=False)

print("=== PARAMETER INSPECTION ===")
print(f"Phase record factory type: {type(wks.phase_record_factory)}")

if hasattr(wks.phase_record_factory, 'param_values'):
    print(f"Number of parameters: {len(wks.phase_record_factory.param_values)}")
    print(f"Parameter values shape: {wks.phase_record_factory.param_values.shape}")
    print(f"Parameter values: {wks.phase_record_factory.param_values}")
    
if hasattr(wks.phase_record_factory, 'param_symbols'):
    print(f"Parameter symbols: {wks.phase_record_factory.param_symbols}")

# Check specific phase records
for phase_name in phases:
    print(f"\n=== Phase: {phase_name} ===")
    phase_rec = wks.phase_record_factory[phase_name]
    
    if hasattr(phase_rec, 'parameters'):
        print(f"  Phase record parameters shape: {phase_rec.parameters.shape}")
        print(f"  Phase record parameters: {phase_rec.parameters}")
    
    print(f"  Number of state variables: {phase_rec.num_statevars}")
    print(f"  Phase DOF: {phase_rec.phase_dof}")
    print(f"  Total variables: {len(phase_rec.variables)}")
    print(f"  State variables: {phase_rec.state_variables}")
    print(f"  Site fractions: {phase_rec.variables}")