#!/usr/bin/env python3
"""Verify the GPU variable index mismatch issue"""

import numpy as np
import sys
sys.path.insert(0, '.')

from pycalphad import Database, Model

# Load database and create model
db = Database('NbTi.tdb')
model = Model(db, ['NB', 'TI', 'VA'], 'BCC_A2')

print("=== Model Analysis ===")
print(f"Model state variables: {model.state_variables}")
print(f"Model site fractions: {model.site_fractions}")

# Get the G expression to see what variables it uses
print("\n=== G Expression (first 500 chars) ===")
print(str(model.G)[:500])

# Now let's simulate what happens when GPU gets workspace DOF
print("\n=== Variable Mismatch Simulation ===")

# Workspace DOF passed to GPU: [N, P, T, Y_NB, Y_TI]
workspace_dof = np.array([1.0, 101325.0, 1000.0, 0.612245, 0.387755])
print(f"Workspace DOF passed: {workspace_dof}")

# GPU expects model DOF: [T, Y_NB, Y_TI]
print("\nGPU reads:")
print(f"  x[0] as T = {workspace_dof[0]} (should be 1000.0)")
print(f"  x[1] as Y_NB = {workspace_dof[1]} (should be 0.612245)")
print(f"  x[2] as Y_TI = {workspace_dof[2]} (should be 0.387755)")

print("\nSite fraction sum:")
print(f"  x[1] + x[2] = {workspace_dof[1]} + {workspace_dof[2]} = {workspace_dof[1] + workspace_dof[2]}")
print(f"  Should be ~1.0, but got {workspace_dof[1] + workspace_dof[2]}")

print("\n=== Impact on Hessian ===")
print("Terms with (Y_NB + Y_TI) in denominator would use 102325 instead of 1.0")
print("This would make those terms ~102325x smaller")
print("But terms without this denominator would be evaluated at wrong T")
print(f"T=1.0K instead of T=1000K could cause massive differences in temperature-dependent terms")

# Check if there are temperature-dependent terms
if 'T*' in str(model.G) or 'log(T)' in str(model.G):
    print("\nG expression contains temperature-dependent terms!")
    print("Evaluating at T=1.0 instead of T=1000 would cause huge errors")