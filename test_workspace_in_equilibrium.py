#!/usr/bin/env python3
"""Test the workspace passed to code generation in equilibrium"""
from pycalphad import Database, equilibrium, variables as v
from pycalphad.gpu.gpu_codegen import notebook_get_all_sym_names_for_model
import sys

# Monkey patch to intercept the workspace
original_generate = None

def patched_generate(wks_obj, include_hess=True, validate=True):
    print(f"\n=== INTERCEPTED WORKSPACE ===")
    print(f"Workspace type: {type(wks_obj)}")
    print(f"Has phase_record_factory: {hasattr(wks_obj, 'phase_record_factory')}")
    
    if hasattr(wks_obj, 'phase_record_factory') and wks_obj.phase_record_factory is not None:
        print(f"phase_record_factory type: {type(wks_obj.phase_record_factory)}")
        print(f"phase_record_factory.state_variables: {wks_obj.phase_record_factory.state_variables}")
    
    # Check what variables would be used
    if hasattr(wks_obj, 'models'):
        for phase_name, model in wks_obj.models.items():
            names = notebook_get_all_sym_names_for_model(model, wks_obj)
            print(f"\nPhase {phase_name} variable names: {names}")
    
    # Call the original function
    return original_generate(wks_obj, include_hess, validate)

# Apply the monkey patch
from pycalphad.gpu import gpu_codegen
original_generate = gpu_codegen._generate_c_code_for_phase_models
gpu_codegen._generate_c_code_for_phase_models = patched_generate

# Now run equilibrium
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']
eq_conditions = {v.N: 1, v.P: 101325, v.T: 1000, v.X('TI'): 0.4}

print("Running equilibrium with GPU...")
try:
    result = equilibrium(db, comps, phases, eq_conditions, verbose=False, gpu=True, calc_opts={'pdens': 5})
    print(f"\nGPU result GM: {float(result.GM.values.flatten()[0])}")
except Exception as e:
    print(f"\nGPU calculation failed: {e}")
    import traceback
    traceback.print_exc()