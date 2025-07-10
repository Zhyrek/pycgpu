#!/usr/bin/env python3
"""Test what variable names the GPU code generation uses"""
from pycalphad import Database, Model
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.variables import T, P, N
from pycalphad.gpu.gpu_codegen import notebook_get_all_sym_names_for_model, notebook_get_all_syms_for_model

# Load database and create model
db = Database('NbTi.tdb')
mod = Model(db, ['NB', 'TI', 'VA'], 'BCC_A2')

# Create phase record factory with proper conditions
conditions = {T: 1000, P: 101325, N: 1}
prf = PhaseRecordFactory(db, ['NB', 'TI', 'VA'], conditions, {'BCC_A2': mod})

# Create minimal workspace
class MinimalWorkspace:
    def __init__(self, prf):
        self.components = ['NB', 'TI', 'VA']
        self.phase_record_factory = prf
        
wks = MinimalWorkspace(prf)

print("=== GPU Variable Name Mapping ===")
names = notebook_get_all_sym_names_for_model(mod, wks)
syms = notebook_get_all_syms_for_model(mod, wks)

print(f"Symbol names: {names}")
print(f"Symbols: {syms}")
print(f"\nMapping:")
for i, (name, sym) in enumerate(zip(names, syms)):
    print(f"  x[{i}] = {name} ({sym})")

# Now let's test with a workspace that doesn't have phase_record_factory
print("\n=== Without phase_record_factory ===")
class BareWorkspace:
    def __init__(self):
        self.components = ['NB', 'TI', 'VA']
        
bare_wks = BareWorkspace()
names_bare = notebook_get_all_sym_names_for_model(mod, bare_wks)
syms_bare = notebook_get_all_syms_for_model(mod, bare_wks)

print(f"Symbol names: {names_bare}")
print(f"Symbols: {syms_bare}")

# This shows the issue - without phase_record_factory, it falls back to model.state_variables
# which only includes T, not [N, P, T]