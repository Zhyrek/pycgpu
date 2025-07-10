#!/usr/bin/env python3
"""Check workspace attributes"""

from pycalphad import Database
from pycalphad.core.workspace import Workspace
import pycalphad.variables as v

# Create a simple workspace
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']
conds = {v.T: 1800, v.P: 101325, v.X('TI'): 0.3}

wks = Workspace(database=db, components=comps, phases=phases, conditions=conds)

# Check attributes
print("Workspace attributes:")
for attr in dir(wks):
    if 'state' in attr.lower() or 'var' in attr.lower():
        print(f"  {attr}: {getattr(wks, attr, 'N/A')}")

# Check what we need
print("\nLooking for state variables...")
if hasattr(wks, 'statevars'):
    print(f"  wks.statevars: {wks.statevars}")
    print(f"  len(wks.statevars): {len(wks.statevars)}")

# Check the models
if hasattr(wks, 'models'):
    model = wks.models['BCC_A2']
    print(f"\nModel variables: {model.variables}")
    print(f"Number of model variables: {len(model.variables)}")
    print(f"First 3 are state variables: {model.variables[:3]}")