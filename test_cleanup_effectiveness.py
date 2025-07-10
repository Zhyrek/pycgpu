#!/usr/bin/env python3
"""Test if the cleanup is effectively removing spurious terms"""

from pycalphad import Database, equilibrium
import pycalphad.variables as v

# First, disable cleanup to see the original code
import pycalphad.gpu.gpu_codegen as gpu_codegen

# Save the original function
original_final_cleanup = gpu_codegen._final_hessian_cleanup

# Create a no-op replacement
def no_cleanup(code):
    print("[DISABLED] Final cleanup disabled for testing")
    return code

# Temporarily disable cleanup
gpu_codegen._final_hessian_cleanup = no_cleanup

# Set up the calculation
dbf = Database("NbTi.tdb")
comps = ["NB", "TI", "VA"]
phases = ["BCC_A2"]
conditions = {
    v.X("TI"): 0.4,
    v.T: 1000,
    v.P: 101325,
    v.N: 1,
}

print("Running GPU calculation WITHOUT cleanup...")
try:
    gpu_eq = equilibrium(dbf, comps, phases, conditions, verbose=True, gpu=True)
except Exception as e:
    print(f"Error: {e}")

# Re-enable cleanup
gpu_codegen._final_hessian_cleanup = original_final_cleanup

print("\n\nRunning GPU calculation WITH cleanup...")
gpu_eq = equilibrium(dbf, comps, phases, conditions, verbose=True, gpu=True)