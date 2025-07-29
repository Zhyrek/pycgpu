#!/usr/bin/env python
"""Debug VA handling in GPU code generation."""

import numpy as np
from pycalphad import Database, Model
from pycalphad.core.workspace import Workspace
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
import pycalphad.variables as v

# Test with simple Al-Cu-Fe database
db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']

# Create models and examine VA handling
print("Examining VA handling in phase models:")
print("="*60)

for phase_name in ['FCC_A1', 'BCC_A2', 'LIQUID']:
    if phase_name not in db.phases:
        continue
        
    phase = db.phases[phase_name]
    print(f"\n{phase_name}:")
    print(f"  Sublattices: {phase.sublattices}")
    print(f"  Constituents: {phase.constituents}")
    
    # Create model
    model = Model(db, components, phase_name)
    print(f"  Site fractions: {model.site_fractions}")
    print(f"  Nonvacant elements: {model.nonvacant_elements}")
    
    # Check moles expressions for each component
    print(f"  Moles expressions:")
    for comp in components:
        try:
            moles_expr = model.moles(comp, per_formula_unit=True)
            print(f"    moles({comp}) = {moles_expr}")
        except Exception as e:
            print(f"    moles({comp}) = ERROR: {e}")
    
    # Check if model has internal constraints
    constraints = model.get_internal_constraints()
    print(f"  Internal constraints: {len(constraints)}")
    if constraints:
        for i, c in enumerate(constraints):
            print(f"    Constraint {i}: {c}")

# Now try to generate GPU code for FCC_A1
print("\n" + "="*60)
print("Attempting to generate GPU code for FCC_A1:")
print("="*60)

# Create workspace with just FCC_A1
conditions = {v.P: 101325, v.T: 1000, v.N: 1}
state_variables = sorted([v.P, v.T, v.N], key=str)

# Create models first
models = {'FCC_A1': Model(db, components, 'FCC_A1')}

# Create phase record factory
prf = PhaseRecordFactory(db, components, state_variables, models)

# Create a minimal workspace object
class MinimalWorkspace:
    def __init__(self):
        self.phases = ['FCC_A1']
        self.components = [v.Species(c) for c in components]
        self.models = models
        self.conditions = conditions
        self.verbose = True
        self.phase_record_factory = prf

wks = MinimalWorkspace()

# Try to generate code
try:
    from pycalphad.gpu.gpu_codegen import _generate_c_code_for_phase_models
    code, init_calls, unique_models, phase_map = _generate_c_code_for_phase_models(wks, include_hess=False, validate=True)
    print("GPU code generation successful!")
    print(f"Generated {len(code)} characters of C code")
    
    # Look for VA-related issues in the generated code
    if 'VA' in code:
        print("\nVA appears in generated code - checking context...")
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if 'VA' in line:
                print(f"  Line {i}: {line.strip()}")
                
except Exception as e:
    print(f"GPU code generation failed: {e}")
    import traceback
    traceback.print_exc()