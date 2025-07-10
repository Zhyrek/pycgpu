#!/usr/bin/env python3
"""Check workspace state variables"""
from pycalphad import Database, variables as v
from pycalphad.core.workspace import Workspace

# Load database
db = Database('NbTi.tdb')
components = ['NB', 'TI']
phases = ['BCC_A2']

# Create workspace
wks = Workspace(db, components, phases, {})

print("=== Workspace State Variables ===")
print(f"Components: {wks.components}")
print(f"Phases: {wks.phases}")

# Check phase record factory
if hasattr(wks, 'phase_record_factory'):
    prf = wks.phase_record_factory
    if hasattr(prf, 'state_variables'):
        print(f"\nPhase record factory state variables: {prf.state_variables}")
    else:
        print("\nPhase record factory has no state_variables attribute")
else:
    print("\nWorkspace has no phase_record_factory")

# Check model variables
model = wks.models['BCC_A2']
print(f"\nModel variables:")
print(f"  Site fractions: {model.site_fractions}")
print(f"  State variables: {model.state_variables}")

# Check if pressure is in state variables
print(f"\nIs pressure in state variables? {v.P in model.state_variables}")
print(f"Is temperature in state variables? {v.T in model.state_variables}")
print(f"Is amount in state variables? {v.N in model.state_variables}")

# Get all variables used in model
all_vars = set()
all_vars.update(model.state_variables)
all_vars.update(model.site_fractions)
print(f"\nAll model variables: {all_vars}")
print(f"Number of variables: {len(all_vars)}")

# Expected order
print("\n=== Expected Variable Order ===")
print("CPU expects: [N, P, T, Y_NB, Y_TI]")
print("GPU generates: [N, T, Y_NB, Y_TI] (missing P!)")