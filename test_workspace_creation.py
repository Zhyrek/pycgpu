#!/usr/bin/env python3
"""Test workspace creation with conditions"""
from pycalphad import Database, variables as v
from pycalphad.core.workspace import Workspace
from pycalphad.core.utils import get_state_variables

# Load database
db = Database('NbTi.tdb')
components = ['NB', 'TI']
phases = ['BCC_A2']

# Test conditions
conditions = {v.N: 1, v.P: 101325, v.T: 1000, v.X('TI'): 0.4}

print("=== Testing Workspace Creation ===")
print(f"Conditions: {conditions}")

# Create workspace
wks = Workspace(db, components, phases, conditions)

print(f"\nWorkspace conditions: {wks.conditions}")
print(f"Workspace phase_record_factory: {wks.phase_record_factory}")

if wks.phase_record_factory:
    print(f"\nPhase record factory state variables: {wks.phase_record_factory.state_variables}")
    
    # Test get_state_variables directly
    print("\n=== Testing get_state_variables ===")
    sv_from_models = get_state_variables(models=wks.models, conds=None)
    print(f"State vars from models only: {sv_from_models}")
    
    sv_from_conds = get_state_variables(models=None, conds=conditions)
    print(f"State vars from conditions only: {sv_from_conds}")
    
    sv_combined = get_state_variables(models=wks.models, conds=conditions)
    print(f"State vars combined: {sv_combined}")

# Check if pressure is missing somewhere
print("\n=== Checking for Pressure ===")
print(f"Is P in conditions? {v.P in conditions}")
print(f"P value in conditions: {conditions.get(v.P, 'NOT FOUND')}")

# Check model state variables
model = wks.models['BCC_A2']
print(f"\nBCC_A2 model state variables: {model.state_variables}")
print(f"Does model use pressure? {v.P in model.state_variables}")