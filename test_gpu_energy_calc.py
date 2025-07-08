#!/usr/bin/env python
"""Test GPU energy calculation directly"""

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

model = wks.models['BCC_A2']
phase_rec = wks.phase_record_factory['BCC_A2']

print("=== TESTING ENERGY CALCULATION ===")

# Create DOF array as GPU would see it
dof = np.array([1.0, 101325, 1000, 0.610169491525422, 0.389830508474579])
print(f"DOF array: {dof}")

# Calculate energy using CPU phase record
energy_out = np.array([0.0])
phase_rec.obj(energy_out, dof)
print(f"CPU GM energy: {energy_out[0]}")

# Also test formula energy
formula_energy_out = np.array([0.0])
phase_rec.formulaobj(formula_energy_out, dof)
print(f"CPU G energy: {formula_energy_out[0]}")

# Print model energy expression
print(f"\nModel GM expression: {model.GM}")
print(f"Model G expression: {model.G}")

# Check if expressions have any special functions that might cause NaN
import symengine as se
print(f"\nGM free symbols: {model.GM.free_symbols}")
print(f"G free symbols: {model.G.free_symbols}")

# Test with just the variables the model expects
model_dof = np.array([1000, 0.610169491525422, 0.389830508474579])
print(f"\nModel-only DOF: {model_dof}")
try:
    from pycalphad.codegen.sympydiff_utils import build_functions
    # Build function with model's variables only
    model_funcs = build_functions(model.GM, tuple(model.state_variables + model.site_fractions), 
                                 parameters=[], include_grad=False, include_hess=False)
    model_energy = model_funcs.func(model_dof)
    print(f"Model-variables-only GM energy: {model_energy}")
except Exception as e:
    print(f"Model-only calc failed: {e}")