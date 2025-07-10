#!/usr/bin/env python3
"""Check variable indices in the model"""

from pycalphad import Database
from pycalphad.model import Model
from pycalphad.core.workspace import Workspace
from pycalphad.gpu.gpu_codegen import notebook_get_all_syms_for_model
import pycalphad.variables as v

# Create workspace and model
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']
conds = {v.T: 1800, v.P: 101325, v.X('TI'): 0.3}

wks = Workspace(database=db, components=comps, phases=phases, conditions=conds)
model = wks.models['BCC_A2']

# Get the ordered symbols
ordered_symbols = notebook_get_all_syms_for_model(model, wks)

print("Ordered symbols for differentiation:")
for i, sym in enumerate(ordered_symbols):
    print(f"  Index {i}: {sym}")

print(f"\nTotal symbols: {len(ordered_symbols)}")
print(f"Site fraction variables start at index: 1 (after T)")