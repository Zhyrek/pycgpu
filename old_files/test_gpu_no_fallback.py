#!/usr/bin/env python3
"""
Test GPU with fallback disabled to see actual GPU errors
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

import numpy as np
import pycalphad as pyc
from pycalphad import Database, variables as v

print("🚀 **GPU-ONLY TEST (NO FALLBACK)**")
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

print(f"Test conditions: {conditions}")
print(f"CRITICAL: fallback_on_error=False - GPU must work or fail!")

try:
    print("\n🚀 Attempting GPU calculation with NO FALLBACK...")
    from pycalphad.gpu.gpu_equilibrium import equilibrium_gpu
    
    # Call GPU equilibrium directly with fallback disabled
    result = equilibrium_gpu(tdb, comps, phases, conditions, 
                            gpu=True, verbose=True, fallback_on_error=False)
    
    print("✅ **GPU SUCCESS!** No fallback occurred!")
    print(f"GM: {result.GM.values.flatten()[0]:.6f} J/mol")
    
except Exception as e:
    print(f"\n❌ **GPU FAILURE** (no fallback):")
    print(f"Error: {e}")
    print(f"Type: {type(e)}")
    
    # Print full traceback to understand the issue
    import traceback
    print(f"\n🔍 **FULL TRACEBACK:**")
    traceback.print_exc()
    
    print(f"\n🎯 **ANALYSIS:**")
    if "'NoneType' object has no attribute 'split'" in str(e):
        print("- String processing error in CuPy/GPU code")
        print("- Likely in module.get_function() call") 
        print("- Function name might be None or malformed")
    
    print(f"\n📝 **NEXT FIX:** Investigate the exact 'split' error location")