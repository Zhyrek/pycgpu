#!/usr/bin/env python
"""Check if the GPU Hessian fix is properly implemented in the generated code."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.gpu.gpu_equilibrium import _generate_kernel_code
from pycalphad.core.workspace import Workspace
from pycalphad.core.utils import instantiate_models

# Set up the problem
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']  
phases = ['BCC_A2']
conds = {v.T: 1800, v.P: 101325, v.X('TI'): 0.3}

# Create workspace and models
wks = Workspace(database=dbf, components=comps, phases=phases, conditions=conds)
models = instantiate_models(dbf, comps, phases)

# Generate kernel code
kernel_code = _generate_kernel_code(wks, models, debug=True)

# Check for the Hessian fix in minimizer.h
print("Checking for Hessian DOF fix in generated code...")
print("=" * 80)

# Look for the critical line where formulahess is called
if "pr->formulahess(csst->hess, compset->dof);" in kernel_code:
    print("✓ FOUND: Hessian is correctly called with compset->dof (workspace DOF)")
    print("  This is the correct implementation that passes full workspace DOF")
elif "pr->formulahess(csst->hess, model_dof_for_calcs);" in kernel_code:
    print("✗ ERROR: Hessian is still called with model_dof_for_calcs")
    print("  This is the OLD broken implementation")
else:
    print("? WARNING: Could not find formulahess call in generated code")

# Also check for the gradient and other function calls
print("\n" + "=" * 80)
print("Checking other function calls...")

if "pr->formulagrad(csst->grad, compset->dof);" in kernel_code:
    print("✓ Gradient correctly uses compset->dof")
else:
    print("✗ Gradient not using compset->dof")
    
if "pr->internal_cons_func(csst->internal_cons, compset->dof);" in kernel_code:
    print("✓ Internal constraints correctly use compset->dof")
else:
    print("✗ Internal constraints not using compset->dof")

# Save the generated code for inspection
with open("generated_equilibrium_kernel.cu", "w") as f:
    f.write(kernel_code)
print("\nGenerated kernel code saved to: generated_equilibrium_kernel.cu")

# Extract and display the relevant section
print("\n" + "=" * 80)
print("Relevant code section around Hessian calculation:")
print("=" * 80)

# Find the section with formulahess
start_idx = kernel_code.find("if (pr->formulahess != nullptr)")
if start_idx != -1:
    # Find the end of this block
    end_idx = kernel_code.find("} else {", start_idx)
    if end_idx != -1:
        end_idx = kernel_code.find("}", end_idx + 1) + 1
        relevant_section = kernel_code[start_idx:end_idx]
        print(relevant_section[:1000])  # Print first 1000 chars
else:
    print("Could not find formulahess section in code")