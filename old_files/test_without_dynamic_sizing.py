#!/usr/bin/env python3
"""
Test GPU without dynamic sizing to see if it works
"""

import os
import glob

def clear_cupy_kernel_cache():
    cache_dir = os.path.expanduser("~/.cupy/kernel_cache")
    if os.path.exists(cache_dir):
        cubin_files = glob.glob(os.path.join(cache_dir, "*.cubin"))
        for cubin_file in cubin_files:
            try:
                os.remove(cubin_file)
            except OSError:
                pass

clear_cupy_kernel_cache()

# Temporarily disable dynamic sizing by modifying the function
import sys
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')

# Monkey patch to disable dynamic sizing temporarily
import pycalphad.gpu.gpu_equilibrium as gpu_eq

original_compute_func = gpu_eq.compute_dynamic_kernel_sizes

def disabled_dynamic_sizing(wks_obj):
    """Return the old hard-coded values to test if that works"""
    return {
        "MAX_COMPONENTS": 32,
        "MAX_PHASES": 64, 
        "MAX_STATEVARS": 8,
        "MAX_DOF_PER_PHASE": 64,
        "MAX_INTERNAL_CONSTRAINTS": 32,
        "MAX_FIXED_MOLE_FRACTION_CONDITIONS": 32,
        "MAX_GRID_POINTS": 10000,
        "MIN_PHASE_FRACTION": 1e-6,
    }

# Temporarily replace the function
gpu_eq.compute_dynamic_kernel_sizes = disabled_dynamic_sizing

import numpy as np
import pycalphad as pyc
from pycalphad import Database, equilibrium, variables as v

print("🔍 TESTING GPU WITHOUT DYNAMIC SIZING")
print("=" * 50)

# Test conditions
tdb = Database("NbTi.tdb")
phases = ["BCC_A2"]
comps = ["NB", "TI", "VA"]
conditions = {
    v.X("TI"): 0.1,
    v.T: 800,
    v.P: 101325
}

print(f"Using hard-coded MAX_* values (old behavior)")
print(f"MAX_COMPONENTS=32, MAX_PHASES=64, etc.")

try:
    print("\n🚀 Attempting GPU calculation with hard-coded sizing...")
    result = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=True)
    print("✅ GPU calculation with hard-coded sizing succeeded!")
    
    # Check if we actually used GPU or fell back to CPU
    if "[GPU] ERROR:" in str(result) or "[GPU] Falling back" in str(result):
        print("❌ Actually fell back to CPU")
    else:
        print("✅ Possibly used GPU successfully")
    
except Exception as e:
    print(f"\n❌ GPU FAILURE: {e}")
    import traceback
    traceback.print_exc()

# Restore the original function
gpu_eq.compute_dynamic_kernel_sizes = original_compute_func