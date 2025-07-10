#!/usr/bin/env python
"""Save the generated CUDA code for inspection."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
# from pycalphad.gpu.gpu_equilibrium import gpu_equilibrium_fixed_phases
from pycalphad.core.workspace import Workspace
from pycalphad.core.starting_point import starting_point
from pycalphad.gpu.gpu_codegen import _generate_full_gpu_source, _generate_c_code_for_phase_models
from pycalphad.core.utils import instantiate_models

# Set up minimal test case
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']
conds = {v.T: 1800, v.P: 101325, v.X('TI'): 0.3}

# Create workspace
wks = Workspace(database=dbf, components=comps, phases=phases, conditions=conds)
models = instantiate_models(dbf, comps, phases)

# Generate model functions C code first
print("Generating model functions C code...")
result = _generate_c_code_for_phase_models(
    wks,
    include_hess=True,
    validate=True
)

# Handle both 2 and 3 return value cases
print(f"Result type: {type(result)}, length: {len(result) if isinstance(result, tuple) else 'N/A'}")
if isinstance(result, tuple):
    if len(result) == 3:
        model_funcs_c, pr_init_calls_c, _ = result
    elif len(result) == 2:
        model_funcs_c, pr_init_calls_c = result
    else:
        # Maybe it returns 4 values?
        model_funcs_c = result[0]
        pr_init_calls_c = result[1]
        print(f"Warning: Got {len(result)} values, using first 2")
else:
    raise ValueError(f"Unexpected return type from _generate_c_code_for_phase_models: {type(result)}")

# Save just the model functions to file
with open("generated_equilibrium_kernel.cu", "w") as f:
    f.write("// Generated CUDA code for equilibrium kernel\n\n")
    f.write("// Phase functions:\n")
    f.write(model_funcs_c)
    f.write("\n// Phase initialization calls:\n")
    # pr_init_calls_c is a list of strings
    if isinstance(pr_init_calls_c, list):
        for call in pr_init_calls_c:
            f.write(call + "\n")
    else:
        f.write(pr_init_calls_c)

print("Generated code saved to generated_equilibrium_kernel.cu")
print("Look for the formulahess function to see if fix_hessian_spurious_terms was applied")