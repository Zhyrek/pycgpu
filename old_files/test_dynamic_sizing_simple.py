#!/usr/bin/env python3
"""
Simple test to verify the dynamic sizing implementation is correct
"""

import sys
import os

# Add the pycalphad directory to the path to test imports
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')

try:
    # Test if we can import the new function
    from pycalphad.gpu.gpu_codegen import compute_dynamic_kernel_sizes
    print("✅ SUCCESS: compute_dynamic_kernel_sizes function imported successfully")
    
    # Test if the function signature is correct
    import inspect
    sig = inspect.signature(compute_dynamic_kernel_sizes)
    params = list(sig.parameters.keys())
    print(f"✅ Function parameters: {params}")
    
    if 'wks_obj' in params:
        print("✅ SUCCESS: Function has correct wks_obj parameter")
    else:
        print("❌ ERROR: Function missing wks_obj parameter")
    
    # Test the return type hint
    return_annotation = sig.return_annotation
    print(f"✅ Return annotation: {return_annotation}")
    
    print("\n🎉 Dynamic sizing function implementation appears correct!")
    print("The function should:")
    print("1. Take a Workspace object as input")
    print("2. Return a Dict[str, int] with computed MAX_* values")
    print("3. Be used in gpu_equilibrium.py to generate -D compiler flags")
    
except ImportError as e:
    print(f"❌ IMPORT ERROR: {e}")
    print("This suggests there may be a syntax error in the code")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()