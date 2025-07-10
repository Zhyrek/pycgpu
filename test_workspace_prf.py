#!/usr/bin/env python3
"""Test if workspace has phase_record_factory during code generation"""
import sys

# Monkey patch to check workspace
def patched_generate(wks_obj, include_hess=True, validate=True):
    print(f"\n=== CODE GENERATION WORKSPACE CHECK ===")
    print(f"Has phase_record_factory: {hasattr(wks_obj, 'phase_record_factory')}")
    
    if hasattr(wks_obj, 'phase_record_factory') and wks_obj.phase_record_factory:
        prf = wks_obj.phase_record_factory
        print(f"phase_record_factory.state_variables: {prf.state_variables}")
    else:
        print("No phase_record_factory available!")
    
    # Return empty to avoid actual code generation
    return "", "", [], {}

# Apply patch before importing equilibrium
from pycalphad.gpu import gpu_codegen
gpu_codegen._generate_c_code_for_phase_models = patched_generate

# Now test
from pycalphad import Database, equilibrium, variables as v

db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA'] 
phases = ['BCC_A2']
conditions = {v.N: 1, v.P: 101325, v.T: 1000, v.X('TI'): 0.4}

try:
    result = equilibrium(db, comps, phases, conditions, gpu=True, calc_opts={'pdens': 5})
except Exception as e:
    print(f"\nExpected error (since we didn't generate real code): {type(e).__name__}")