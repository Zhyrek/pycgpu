#!/usr/bin/env python
"""Test what variables CPU compiled functions expect"""

import numpy as np
from pycalphad import Database, Workspace
import pycalphad.variables as v
from pycalphad.codegen.sympydiff_utils import build_functions

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
prf = wks.phase_record_factory

print("=== CPU FUNCTION VARIABLE ANALYSIS ===")
print(f"Model state variables: {model.state_variables}")
print(f"Model site fractions: {model.site_fractions}")
print(f"PRF state variables: {prf.state_variables}")

# Check what variables are used to build functions
print(f"\nVariables used to build CPU functions:")
print(f"  {prf.state_variables + model.site_fractions}")

# Build a test function to see what it expects
print("\nBuilding test function for model.G...")
G_funcs = build_functions(model.G, tuple(prf.state_variables + model.site_fractions), 
                         parameters=prf.param_symbols, include_grad=False, include_hess=True)

# Test with different DOF arrays
print("\nTesting function with different inputs:")

# Test 1: Full DOF array [N, P, T, Y(NB), Y(TI)]
dof1 = np.array([1.0, 101325, 1000, 0.6, 0.4])
result1 = G_funcs.func(dof1)
print(f"Full DOF array {dof1}: G = {result1}")

# Test 2: Model-only array [T, Y(NB), Y(TI)]
dof2 = np.array([1000, 0.6, 0.4])
try:
    result2 = G_funcs.func(dof2)
    print(f"Model-only array {dof2}: G = {result2}")
except Exception as e:
    print(f"Model-only array failed: {e}")

# Check the actual symengine function
print(f"\nG expression free symbols: {model.G.free_symbols}")
print(f"Does G depend on N? {v.N in model.G.free_symbols}")
print(f"Does G depend on P? {v.P in model.G.free_symbols}")
print(f"Does G depend on T? {v.T in model.G.free_symbols}")