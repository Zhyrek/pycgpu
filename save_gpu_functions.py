#!/usr/bin/env python3
"""Save generated GPU functions to file for inspection"""

import sys
sys.path.insert(0, '.')

from pycalphad import Database, equilibrium, variables as v
import os

# Monkey-patch gpu_equilibrium to save the generated code
original_equilibrium_gpu = None
generated_code = None

def patched_equilibrium_gpu(*args, **kwargs):
    global generated_code
    # Import here to avoid circular import
    from pycalphad.gpu.gpu_equilibrium import _generate_equilibrium_cuda_code
    
    # Get the workspace object
    wks_obj = args[0]
    
    # Generate the code
    code, _ = _generate_equilibrium_cuda_code(wks_obj)
    generated_code = code
    
    # Save to file
    with open('gpu_generated_functions.cu', 'w') as f:
        f.write(code)
    
    print("Generated GPU code saved to gpu_generated_functions.cu")
    
    # Call original function
    return original_equilibrium_gpu(*args, **kwargs)

# Apply monkey patch
from pycalphad.gpu import gpu_equilibrium
original_equilibrium_gpu = gpu_equilibrium.equilibrium_gpu
gpu_equilibrium.equilibrium_gpu = patched_equilibrium_gpu

# Run equilibrium calculation
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']
eq_conditions = {v.N: 1, v.P: 101325, v.T: 1000, v.X('TI'): 0.4}

try:
    result = equilibrium(db, comps, phases, eq_conditions, verbose=False, gpu=True, to='GM', calc_opts={'pdens': 50})
except Exception as e:
    print(f"Error during equilibrium: {e}")
    if generated_code:
        print("But code was generated and saved.")