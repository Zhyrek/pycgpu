#!/usr/bin/env python
"""Regenerate GPU code without debug statements."""

import os
import sys
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import numpy as np

# Remove any existing generated file
if os.path.exists('generated_equilibrium_kernel.cu'):
    os.remove('generated_equilibrium_kernel.cu')
    print("Removed existing generated_equilibrium_kernel.cu")

# Create test case
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

print("Regenerating GPU code...")

# Test conditions
conditions = {v.X('TI'): 0.01, v.T: 1000, v.P: 101325}

# Run GPU calculation - this will regenerate the code
try:
    gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True)
    print("✓ GPU code generated successfully")
except Exception as e:
    print(f"GPU generation/compilation failed: {e}")
    # The code generation might have succeeded even if compilation failed
    
# Check if the file was generated
if os.path.exists('generated_equilibrium_kernel.cu'):
    print("✓ generated_equilibrium_kernel.cu was created")
    # Count lines
    with open('generated_equilibrium_kernel.cu', 'r') as f:
        lines = f.readlines()
    print(f"  File has {len(lines)} lines")
    
    # Check for debug macros
    debug_macros = ['GPU_DEBUG_PRINT', 'GPU_DEBUG_ACTIVE', 'INITIALIZATION', 'HESSIAN']
    found_macros = []
    for macro in debug_macros:
        for i, line in enumerate(lines):
            if macro in line:
                found_macros.append((macro, i+1, line.strip()))
    
    if found_macros:
        print("\n⚠️  Found debug macros that should not be present:")
        for macro, line_num, line in found_macros[:5]:  # Show first 5
            print(f"  Line {line_num}: {macro} in: {line[:80]}...")
    else:
        print("✓ No debug macros found in generated code")
else:
    print("✗ generated_equilibrium_kernel.cu was not created")