#!/usr/bin/env python
"""Debug script to identify exact location of GPU memory access error."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.gpu.gpu_equilibrium import gpu_equilibrium_fixed_phases
from pycalphad.core.workspace import Workspace
from pycalphad.core.utils import instantiate_models
from pycalphad.core.starting_point import starting_point
import os

# Enable maximum debug output
os.environ['PYCALPHAD_DEBUG'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Force synchronous execution for better error detection

print("Debugging GPU memory access issue...")
print("=" * 80)

# Set up minimal test case
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']
conds = {v.T: 1800, v.P: 101325, v.X('TI'): 0.3}

# Create workspace
wks = Workspace(database=dbf, components=comps, phases=phases, conditions=conds)

# Get starting point
properties = starting_point(dbf, comps, phases, conds, wks)

print("Starting point properties obtained")
print(f"Phase amounts shape: {properties.NP.shape}")
print(f"Phase amounts: {properties.NP.values}")

# Try to call GPU equilibrium directly to get more detailed error info
try:
    print("\nCalling gpu_equilibrium_fixed_phases directly...")
    result = gpu_equilibrium_fixed_phases(wks, properties)
    print("GPU succeeded!")
except Exception as e:
    print(f"GPU failed with: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    
print("\nAnalyzing potential causes:")
print("1. Array size mismatches between CPU and GPU")
print("2. Incorrect indexing in global memory arrays")
print("3. Struct alignment issues between host and device")
print("4. Function pointer corruption")

# Let's check the sizes being used
from pycalphad.gpu.gpu_codegen import compute_dynamic_kernel_sizes, _get_c_define

models = instantiate_models(dbf, comps, phases)
dynamic_sizes = compute_dynamic_kernel_sizes(wks)

print("\nDynamic kernel sizes:")
for key, value in dynamic_sizes.items():
    print(f"  {key}: {value}")

print("\nStatic C defines:")
for define in ["MAX_COMPONENTS", "MAX_PHASES", "MAX_STATEVARS", "MAX_DOF_PER_PHASE", 
               "MAX_INTERNAL_CONSTRAINTS", "MAX_FIXED_MOLE_FRACTION_CONDITIONS"]:
    print(f"  {define}: {_get_c_define(define)}")

print("\nChecking array allocation sizes...")
# The GPU allocates these arrays per thread
max_components = _get_c_define("MAX_COMPONENTS")
max_phases = _get_c_define("MAX_PHASES") 
max_statevars = _get_c_define("MAX_STATEVARS")
max_dof = _get_c_define("MAX_DOF_PER_PHASE")

print(f"\nPer-thread array sizes:")
print(f"  equilibrium_matrix: {(max_phases + max_components + 1) * (max_components + max_phases + max_statevars)}")
print(f"  hess: {(max_statevars + max_dof) * (max_statevars + max_dof)}")
print(f"  phase_matrix: {(max_dof + max_components) * (max_dof + max_components)}")

print("\nThe issue likely occurs when:")
print("- Actual data exceeds MAX_* limits")
print("- Array indexing uses wrong dimensions")
print("- Struct members are accessed with wrong offsets")