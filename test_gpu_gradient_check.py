#!/usr/bin/env python
"""Check if GPU gradient function is working correctly."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("GPU GRADIENT CHECK")
print("=" * 60)

# Clear cache
cache_dir = os.path.expanduser('~/.cache/pycalphad_gpu')
if os.path.exists(cache_dir):
    import shutil
    shutil.rmtree(cache_dir)

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Run a simple test to generate the kernel
conditions = {v.X('TI'): 0.5, v.T: 1000, v.P: 101325}
result = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)

# Extract the generated kernel
with open('generated_equilibrium_kernel.cu', 'r') as f:
    kernel_content = f.read()

# Check the formulamole_grad function
import re
grad_pattern = r'__device__ void pycgpu_model_0_formulamole_grad\(double\* out, const double\* x\) \{([^}]+)\}'
match = re.search(grad_pattern, kernel_content)

if match:
    print("\nFound formulamole_grad function:")
    print("__device__ void pycgpu_model_0_formulamole_grad(double* out, const double* x) {")
    print(match.group(1))
    print("}")
    
    # Extract the gradient values
    grad_content = match.group(1)
    print("\nGradient analysis:")
    print("- d(moles(NB))/dY(NB) = out[3] =", "1.0" if "out[3] = 1.0" in grad_content else "NOT 1.0")
    print("- d(moles(NB))/dY(TI) = out[4] =", "0" if "out[4] = 0" in grad_content else "NOT 0") 
    print("- d(moles(TI))/dY(NB) = out[8] =", "0" if "out[8] = 0" in grad_content else "NOT 0")
    print("- d(moles(TI))/dY(TI) = out[9] =", "1.0" if "out[9] = 1.0" in grad_content else "NOT 1.0")
    
    print("\nExpected behavior (treating all site fractions as independent):")
    print("- d(moles(i))/dY(j) = 1 if i==j, 0 otherwise")
    
    if "out[3] = 1.0" in grad_content and "out[9] = 1.0" in grad_content:
        print("\n✓ GRADIENT IS CORRECT - All site fractions are independent")
    else:
        print("\n✗ GRADIENT IS WRONG - Dependent substitution is being applied")
else:
    print("\nERROR: Could not find formulamole_grad function")