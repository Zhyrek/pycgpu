#!/usr/bin/env python
"""Save the generated CUDA file for debugging."""

import os
import sys
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import numpy as np

# Monkey patch to save the generated code
original_rawmodule = None

def patched_rawmodule(code, *args, **kwargs):
    # Save the code to file
    with open('generated_equilibrium_kernel.cu', 'w') as f:
        f.write(code)
    print(f"✓ Saved generated CUDA code to generated_equilibrium_kernel.cu ({len(code)} bytes)")
    
    # Call original
    return original_rawmodule(code, *args, **kwargs)

# Apply monkey patch
import cupy as cp
original_rawmodule = cp.RawModule
cp.RawModule = patched_rawmodule

# Create test case
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

print("Generating GPU code...")

# Test conditions
conditions = {v.X('TI'): 0.01, v.T: 1000, v.P: 101325}

# Run GPU calculation - this will save the generated code
try:
    gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True)
    print("✓ GPU code generated and compiled successfully")
except Exception as e:
    print(f"GPU generation/compilation failed: {e}")