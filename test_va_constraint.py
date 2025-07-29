#!/usr/bin/env python
"""Test constraint handling for phases with VA in sublattices."""

import numpy as np
from pycalphad import Database, Model
from pycalphad.core.workspace import Workspace
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.gpu.gpu_codegen import notebook_source_from_expr, notebook_model_c_func_name_prefix
import pycalphad.variables as v

# Test with Al-Cu-Fe 
db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']

# Create model for FCC_A1
model = Model(db, components, 'FCC_A1')

print("FCC_A1 phase analysis:")
print("="*60)
print(f"Site fractions: {model.site_fractions}")
print(f"Internal constraints: {model.get_internal_constraints()}")

# Look at the constraint expressions in detail
print("\nDetailed constraint analysis:")
for i, constraint in enumerate(model.get_internal_constraints()):
    print(f"\nConstraint {i}: {constraint}")
    print(f"  Type: {type(constraint)}")
    print(f"  Free symbols: {constraint.free_symbols}")
    
    # Check if it references VA site fraction
    for symbol in constraint.free_symbols:
        if 'VA' in str(symbol):
            print(f"  ⚠️  References VA site fraction: {symbol}")

# Try to generate C code for the constraints
print("\n" + "="*60)
print("Attempting to generate C code for constraints:")

# Create minimal workspace
conditions = {v.P: 101325, v.T: 1000, v.N: 1}
state_variables = sorted([v.P, v.T, v.N], key=str)
models = {'FCC_A1': model}
prf = PhaseRecordFactory(db, components, state_variables, models)

class MinimalWorkspace:
    def __init__(self):
        self.phases = ['FCC_A1']
        self.components = [v.Species(c) for c in components]
        self.models = models
        self.conditions = conditions
        self.verbose = False
        self.phase_record_factory = prf

wks = MinimalWorkspace()

# Generate constraint functions
try:
    from pycalphad.gpu.gpu_codegen import _nb_internal_cons_func_from_model
    constraint_code = _nb_internal_cons_func_from_model(model, 0, wks, validate=True, verbose=True)
    print("\nGenerated constraint function:")
    print(constraint_code[:500] + "..." if len(constraint_code) > 500 else constraint_code)
    
    # Look for specific patterns in the code
    if 'x[6]' in constraint_code:
        print("\n⚠️  Code references x[6] which is Y(FCC_A1,1,VA)")
        
    # Check if constraint is always satisfied
    if '-1 + x[6]' in constraint_code or '-1.0 + x[6]' in constraint_code:
        print("\n⚠️  Constraint '-1 + x[6]' expects Y(FCC_A1,1,VA) = 1.0")
        print("    This means the second sublattice must be 100% VA")
        
except Exception as e:
    print(f"\nFailed to generate constraint code: {e}")
    import traceback
    traceback.print_exc()

# Now check a phase without VA in sublattices
print("\n" + "="*60)
print("Comparison with LIQUID phase (no VA in sublattices):")
liquid_model = Model(db, components, 'LIQUID')
print(f"LIQUID site fractions: {liquid_model.site_fractions}")
print(f"LIQUID constraints: {liquid_model.get_internal_constraints()}")