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
model_funcs_c, pr_init_calls_c, unique_models, phase_name_map = _generate_c_code_for_phase_models(
    wks,
    include_hess=True
)
num_unique_models = len(unique_models)

# Generate the CUDA source
print("Generating CUDA source code...")
cuda_source = _generate_full_gpu_source(wks, model_funcs_c, pr_init_calls_c, num_unique_models)

# Save to file
with open("generated_cuda_code.cu", "w") as f:
    f.write(cuda_source)

print(f"CUDA source saved to generated_cuda_code.cu ({len(cuda_source)} chars)")

# Find line 5513
lines = cuda_source.split('\n')
if len(lines) > 5513:
    print(f"\nLine 5513: {lines[5512]}")  # 0-indexed
    print("Context:")
    for i in range(max(0, 5510), min(len(lines), 5520)):
        print(f"{i+1}: {lines[i]}")
else:
    print(f"Code only has {len(lines)} lines")