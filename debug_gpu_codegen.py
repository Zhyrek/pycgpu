#!/usr/bin/env python
"""Debug GPU code generation to understand why BCC functions are missing."""

from pycalphad import Database, calculate, variables as v
from pycalphad.core.workspace import Workspace
from pycalphad.gpu.gpu_codegen import _generate_c_code_for_phase_models
import numpy as np

# Load database and set up system
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = list(db.phases.keys())

print("Phases in database:", phases)

# Set up conditions for test
T = 1300.0
conditions = {v.T: T, v.P: 101325, v.X('TI'): 0.5}

# Run calculate to create workspace
calc_result = calculate(db, comps, phases, T=T, P=101325, N=1, output='GM')

# Create workspace with verbose=True
wks = Workspace(db, comps, phases, conditions, calc_result, 
                parameters={}, phase_record_factory=None, verbose=True)

print("\nGenerating C code for phase models...")
try:
    model_funcs_c, pr_init_calls_c, unique_py_models, py_phase_name_to_unique_idx_map = \
        _generate_c_code_for_phase_models(wks, include_hess=True, validate=True)
    
    print(f"\nUnique models found: {len(unique_py_models)}")
    print(f"Phase name to index mapping: {py_phase_name_to_unique_idx_map}")
    
    print("\nPhase record init calls:")
    for i, call in enumerate(pr_init_calls_c):
        print(f"  {i}: {call}")
    
    # Check if BCC functions were generated
    if "pycgpu_model_1" in model_funcs_c:
        print("\nBCC functions (model_1) WERE generated in C code")
    else:
        print("\nBCC functions (model_1) were NOT generated in C code")
        
    # Count how many model functions were generated
    import re
    model_0_count = len(re.findall(r'pycgpu_model_0_\w+', model_funcs_c))
    model_1_count = len(re.findall(r'pycgpu_model_1_\w+', model_funcs_c))
    
    print(f"\nModel 0 functions found: {model_0_count}")
    print(f"Model 1 functions found: {model_1_count}")
    
except Exception as e:
    print(f"Error during code generation: {e}")
    import traceback
    traceback.print_exc()