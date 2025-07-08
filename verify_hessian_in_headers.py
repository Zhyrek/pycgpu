#!/usr/bin/env python
"""Verify that the Hessian fix is present in the GPU header files."""

import os

# Check minimizer.h directly
minimizer_h_path = "/mnt/c/users/scott/Documents/pycalphad/pycalphad/gpu/minimizer.h"

print("Checking minimizer.h for Hessian fix...")
print("=" * 80)

with open(minimizer_h_path, 'r') as f:
    content = f.read()
    
# Look for the critical fix
if "pr->formulahess(csst->hess, compset->dof);" in content:
    print("✓ FOUND: Hessian correctly uses compset->dof (workspace DOF)")
    # Find the line number
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if "pr->formulahess(csst->hess, compset->dof);" in line:
            print(f"  Found at line {i+1}")
            # Print surrounding context
            print("\n  Context:")
            for j in range(max(0, i-2), min(len(lines), i+3)):
                print(f"  {j+1}: {lines[j]}")
            break
elif "pr->formulahess(csst->hess, model_dof_for_calcs);" in content:
    print("✗ ERROR: Hessian still uses model_dof_for_calcs (OLD BROKEN CODE)")
    # Find the line number
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if "pr->formulahess(csst->hess, model_dof_for_calcs);" in line:
            print(f"  Found at line {i+1}")
            print("\n  THIS NEEDS TO BE FIXED!")
            break
else:
    print("? WARNING: Could not find formulahess call")

# Also check for the comment that explains the fix
if "MUST pass full workspace DOF like CPU does" in content:
    print("\n✓ Found comment explaining the fix")
else:
    print("\n? No explanatory comment found")

# Check if there's a GPU-specific version that might be overriding this
print("\n" + "=" * 80)
print("Checking for any GPU-specific minimizer implementations...")

gpu_codegen_path = "/mnt/c/users/scott/Documents/pycalphad/pycalphad/gpu/gpu_codegen.py"
with open(gpu_codegen_path, 'r') as f:
    codegen_content = f.read()
    
# Check if there's a gpu_phase_minimize function that might override the fix
if "gpu_phase_minimize" in codegen_content:
    print("! Found gpu_phase_minimize function in gpu_codegen.py")
    lines = codegen_content.split('\n')
    for i, line in enumerate(lines):
        if "formulahess" in line and "model_dof" in line:
            print(f"  Line {i+1}: {line.strip()}")
            if "model_dof_for_calcs" in line:
                print("  ⚠️  This line may be overriding the fix!")

# Check for phase_minimize in the global mem solver
if "phase_minimize" in codegen_content and "global_mem" in codegen_content:
    print("\nFound phase_minimize references in global memory solver")
    
print("\n" + "=" * 80)
print("Summary:")
print("The fix should be in minimizer.h where formulahess is called with compset->dof")
print("This ensures the GPU Hessian receives the full workspace DOF format [N, P, T, Y1, Y2...]")
print("instead of the model DOF format [T, Y1, Y2...]")