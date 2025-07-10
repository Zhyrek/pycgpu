#!/usr/bin/env python3
"""Save the generated CUDA code after clearing cache"""
from pycalphad import Database, Model, Workspace
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory  
from pycalphad.variables import T, P, N
from pycalphad.gpu.gpu_equilibrium import clear_gpu_cache
from pycalphad.gpu.gpu_codegen import _generate_full_gpu_source

# Clear cache first
clear_gpu_cache()

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']

# Create phase record factory with proper conditions
conditions = {T: 1000, P: 101325, N: 1}
models = {phase: Model(db, comps, phase) for phase in phases}
prf = PhaseRecordFactory(db, comps, conditions, models)

# Create workspace with phase_record_factory
wks = Workspace(db, comps, phases, conditions, models=models, phase_record_factory=prf)

# Generate the CUDA code
print("Generating CUDA code...")
# Check the function signature
import inspect
print(f"_generate_full_gpu_source signature: {inspect.signature(_generate_full_gpu_source)}")

# Call with correct arguments
# Looks like it needs (wks_obj, starting_point, conditions, parameters, extra_code=None, include_hessian=True, code_validation=True)
# But let me use the code generation directly
from pycalphad.gpu.gpu_codegen import _generate_c_code_for_phase_models

model_funcs_c, pr_init_calls_c, unique_py_models, py_phase_name_to_unique_idx_map = \
    _generate_c_code_for_phase_models(wks, include_hess=True, validate=True)

# Save just the model functions
with open('generated_model_funcs.cu', 'w') as f:
    f.write(model_funcs_c)
    
print("Saved model functions to generated_model_funcs.cu")

# Search for formulamole_grad  
lines = model_funcs_c.split('\n')
for i, line in enumerate(lines):
    if 'pycgpu_model_0_formulamole_grad' in line:
        print(f"\nFound formulamole_grad at line {i+1}:")
        # Print the function and a few lines after
        for j in range(i, min(i+15, len(lines))):
            print(f"{j+1}: {lines[j]}")