#!/usr/bin/env python3
"""Verify GPU path is being taken"""

import sys
import pycalphad as pyc
from pycalphad import Database, equilibrium, variables as v

# Monkey patch to trace imports
original_import = __builtins__.__import__

def traced_import(name, *args, **kwargs):
    if 'gpu' in name.lower():
        print(f">>> IMPORTING: {name}")
    return original_import(name, *args, **kwargs)

__builtins__.__import__ = traced_import

# Test GPU call
print("Setting up test...")
tdb = pyc.Database("NbTi.tdb")
phases = ["LIQUID", "BCC_A2"]
comps = ["NB", "TI", "VA"]
conditions = {v.X("TI"): 0.5, v.T: 500}

print("\nCalling equilibrium with gpu=True...")
eq_gpu = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=True)

print("\nGPU path was taken if you see GPU imports above.")

# Also check if the GPU module is loaded
print("\nLoaded GPU modules:")
for module in sys.modules:
    if 'gpu' in module.lower():
        print(f"  {module}")

# Restore original import
__builtins__.__import__ = original_import