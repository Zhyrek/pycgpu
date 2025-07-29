#!/usr/bin/env python
"""Test the order of outputs from CSE gradient functions."""

import numpy as np
from pycalphad import Database, variables as v
from pycalphad.core.workspace import Workspace

# Load database
db = Database('NbTi.tdb')
components = ['NB', 'TI', 'VA']
phase_name = 'BCC_A2'

conditions = {
    v.T: 500,
    v.P: 101325,
    v.X('TI'): 0.1,
    v.N: 1
}

print("Testing CSE gradient output order...")
print("="*80)

# Create workspace
wks = Workspace(db, components, [phase_name], conditions, verbose=False)

# Get phase record
phase_record = wks.phase_record_factory[phase_name]

# Test DOF (from CPU trace)
dof = np.array([1.0, 101325.0, 500.0, 0.93877551, 0.06122449])
print(f"Test DOF: N={dof[0]}, P={dof[1]}, T={dof[2]}, Y(NB)={dof[3]:.6f}, Y(TI)={dof[4]:.6f}")

# Get gradient
grad = np.zeros(5)
phase_record.formulagrad(grad, dof)

print("\nCPU gradient array (full format):")
for i, val in enumerate(grad):
    names = ['dG/dN', 'dG/dP', 'dG/dT', 'dG/dY(NB)', 'dG/dY(TI)']
    print(f"  grad[{i}] = {val:.6f} ({names[i]})")

# The CPU trace shows:
# gradient values: [-14863.1786191   -8784.46966485]
# This appears to be [dG/dY(NB), dG/dY(TI)]

print("\nFrom CPU trace, 'gradient values' shows:")
print("  [-14863.1786191, -8784.46966485]")
print("  This matches grad[3] and grad[4] (site fraction derivatives)")

print("\nThe issue is:")
print("  CSE gradient outputs: [dG/dT, dG/dY(NB), dG/dY(TI)]")
print("  Expected order: [-51.36..., -14863.17..., -8784.46...]")
print("  But GPU is seeing: [-14863.17..., -51.36..., -8784.46...]")
print("  The T derivative and Y(NB) derivative are swapped!")

print("\n" + "="*80)
print("The CSE gradient function must be outputting:")
print("  out[0] = dG/dY(NB) = -14863.17...")
print("  out[1] = dG/dT = -51.36...")  
print("  out[2] = dG/dY(TI) = -8784.46...")
print("This is NOT the expected [T, Y(NB), Y(TI)] order!")
print("="*80)