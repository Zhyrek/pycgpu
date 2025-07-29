#!/usr/bin/env python
"""Diagnose why GPU fails with phases having VA in sublattices."""

import numpy as np
from pycalphad import Database, equilibrium
import pycalphad.variables as v
import os

# Suppress CPU debug output
os.environ['PYCALPHAD_DEBUG'] = '0'

# Test with simplified conditions
db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']

conditions = {
    v.T: 1000,
    v.P: 101325,
    v.X('AL'): 0.5,
    v.X('CU'): 0.3,
    v.N: 1
}

print("Testing why GPU fails with VA in sublattices")
print("="*60)

# First test CPU to see if it works
print("\n1. Testing CPU equilibrium with FCC_A1:")
try:
    cpu_result = equilibrium(db, components, ['FCC_A1'], conditions, calc_opts={'pdens': 10})
    print(f"   CPU Success! GM = {cpu_result.GM.values[0,0,0,0]:.1f} J/mol")
    
    # Extract phase information
    if hasattr(cpu_result, 'Phase'):
        phases = cpu_result.Phase.values[0,0,0,:]
        phase_amounts = cpu_result.NP.values[0,0,0,:]
        active_phases = [(p, amt) for p, amt in zip(phases, phase_amounts) if amt > 1e-10]
        print(f"   Active phases: {active_phases}")
        
except Exception as e:
    print(f"   CPU Failed: {e}")

# Now test GPU with verbose output disabled to see the actual error
print("\n2. Testing GPU equilibrium with FCC_A1:")
try:
    # Clear any GPU cache
    from pycalphad.gpu.gpu_equilibrium import _gpu_module_cache
    _gpu_module_cache.clear()
    print("   GPU cache cleared")
    
    gpu_result = equilibrium(db, components, ['FCC_A1'], conditions, 
                           calc_opts={'pdens': 10}, gpu=True, verbose=False)
    print(f"   GPU Success! GM = {gpu_result.GM.values[0,0,0,0]:.1f} J/mol")
except Exception as e:
    print(f"   GPU Failed with error: {type(e).__name__}: {e}")
    
    # Try to get more specific error information
    error_msg = str(e)
    if "compilation" in error_msg.lower():
        print("   → Compilation error detected")
    elif "cuda" in error_msg.lower():
        print("   → CUDA error detected")
    elif "validation" in error_msg.lower():
        print("   → Code validation error detected")

# Test with LIQUID phase (no VA in sublattices) as control
print("\n3. Testing GPU with LIQUID phase (no VA in sublattices):")
try:
    _gpu_module_cache.clear()
    gpu_liquid = equilibrium(db, components, ['LIQUID'], conditions,
                           calc_opts={'pdens': 10}, gpu=True, verbose=False)
    print(f"   GPU Success! GM = {gpu_liquid.GM.values[0,0,0,0]:.1f} J/mol")
except Exception as e:
    print(f"   GPU Failed: {e}")

# Now test with verbose=True to see where it fails
print("\n4. Testing GPU with FCC_A1 with verbose output:")
print("-"*60)
try:
    _gpu_module_cache.clear()
    gpu_verbose = equilibrium(db, components, ['FCC_A1'], conditions,
                            calc_opts={'pdens': 10}, gpu=True, verbose=True)
    print(f"GPU Success! GM = {gpu_verbose.GM.values[0,0,0,0]:.1f} J/mol")
except Exception as e:
    print(f"\nGPU Failed with final error: {e}")
    
print("\n" + "="*60)
print("Summary:")
print("- FCC_A1 has structure (AL,CU,FE):(VA) with Y(FCC_A1,1,VA) constrained to 1.0")
print("- The GPU code generation succeeds but something fails during execution")
print("- The issue is specific to phases with VA-only sublattices")