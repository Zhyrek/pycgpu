#!/usr/bin/env python3
"""
Debug the GPU failure during PhaseRecord initialization
"""

import os
import glob
import traceback

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

import numpy as np
import pycalphad as pyc
from pycalphad import Database, equilibrium, variables as v

print("🔍 DEBUGGING GPU FAILURE")
print("=" * 50)

# Test conditions - simple case
tdb = Database("NbTi.tdb")
phases = ["BCC_A2"]
comps = ["NB", "TI", "VA"]
conditions = {
    v.X("TI"): 0.1,
    v.T: 800,
    v.P: 101325
}

print(f"Test conditions: {conditions}")

try:
    print("\n🚀 Attempting GPU calculation with full error tracing...")
    result = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=True)
    print("✅ GPU calculation succeeded!")
    
except Exception as e:
    print(f"\n❌ GPU FAILURE CAUGHT: {e}")
    print("\n🔍 FULL TRACEBACK:")
    traceback.print_exc()
    
    print(f"\n🔍 ERROR TYPE: {type(e)}")
    print(f"🔍 ERROR ARGS: {e.args}")
    
    # Try to identify the specific issue
    error_str = str(e)
    if "'NoneType' object has no attribute 'split'" in error_str:
        print("\n🎯 IDENTIFIED ISSUE: String processing error")
        print("   - Something that should be a string is None")
        print("   - Likely in PhaseRecord initialization")
        print("   - May be related to dynamic sizing changing expected values")
    
    print(f"\n📝 NEXT STEPS:")
    print("1. Check if dynamic sizing breaks string operations in kernel")
    print("2. Examine PhaseRecord initialization code")
    print("3. Compare generated kernel with/without dynamic sizing")
    print("4. Check if any constants are used as strings in the C code")