#!/usr/bin/env python
"""Debug DOF array indexing differences between CPU and GPU"""

import numpy as np
from pycalphad import Database, Workspace
import pycalphad.variables as v

# Simple binary system for testing
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']
conds = {v.T: 1000, v.P: 101325, v.X('TI'): 0.4}

# Create workspace
wks = Workspace(database=dbf, components=comps, phases=phases, 
                conditions=conds, models=None, parameters=None,
                calc_opts={'pdens': 10}, verbose=False)

print("=== DOF ARRAY STRUCTURE ===")

# Get the BCC_A2 model
model = wks.models['BCC_A2']
phase_rec = wks.phase_record_factory['BCC_A2']

print(f"\nModel state variables: {model.state_variables}")
print(f"Model site fractions: {model.site_fractions}")
print(f"Total model variables: {len(model.state_variables) + len(model.site_fractions)}")

print(f"\nPhase record state variables: {phase_rec.state_variables}")
print(f"Phase record num_statevars: {phase_rec.num_statevars}")
print(f"Phase record phase_dof: {phase_rec.phase_dof}")
print(f"Phase record variables: {phase_rec.variables}")

# Create a sample DOF array like the equilibrium solver would
dof = np.zeros(phase_rec.num_statevars + phase_rec.phase_dof)
dof[0] = 1.0  # N
dof[1] = 101325  # P
dof[2] = 1000  # T
dof[3] = 0.6  # Y(BCC_A2,0,NB)
dof[4] = 0.4  # Y(BCC_A2,0,TI)

print(f"\nSample DOF array: {dof}")
print("\nExpected indexing:")
for i, var in enumerate(phase_rec.state_variables + phase_rec.variables):
    print(f"  dof[{i}] = {dof[i]:.6f} ({var})")

# Check what the model expects
print(f"\nModel expects variables in this order:")
ordered_vars = model.state_variables + model.site_fractions
for i, var in enumerate(ordered_vars):
    print(f"  x[{i}] = {var}")

# Compare with phase record factory state variables
print(f"\nPhase record factory state variables: {wks.phase_record_factory.state_variables}")

# Key insight: The model may expect fewer state variables than the phase record!
print(f"\nCRITICAL: Model uses {len(model.state_variables)} state vars, phase rec has {phase_rec.num_statevars}")