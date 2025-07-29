#!/usr/bin/env python
"""Test if CSE energy and gradient functions are correct for both phases in miscibility gap."""

import numpy as np
from pycalphad import Database, variables as v
from pycalphad.core.workspace import Workspace
from pycalphad.gpu.gpu_codegen import _generate_c_code_for_phase_models
import cupy as cp

# Load database
db = Database('NbTi.tdb')
components = ['NB', 'TI', 'VA']
phase_name = 'BCC_A2'

# Test conditions
conditions = {
    v.T: 500,
    v.P: 101325,
    v.X('TI'): 0.1,
    v.N: 1
}

print("Testing CSE energy/gradient for miscibility gap compositions...")
print("="*80)

# Create workspace
wks = Workspace(db, components, [phase_name], conditions, verbose=False)

# Get phase record
phase_record = wks.phase_record_factory[phase_name]

# Test two compositions from the miscibility gap
test_cases = [
    ("Ti-poor phase", np.array([1.0, 101325.0, 500.0, 0.94085304, 0.05914696])),
    ("Ti-rich phase", np.array([1.0, 101325.0, 500.0, 0.05914696, 0.94085304]))
]

for name, dof in test_cases:
    print(f"\n{name}:")
    print(f"  DOF: N={dof[0]}, P={dof[1]}, T={dof[2]}, Y(NB)={dof[3]:.6f}, Y(TI)={dof[4]:.6f}")
    
    # CPU calculation
    cpu_energy = np.zeros(1)
    phase_record.obj(cpu_energy, dof)
    
    cpu_grad = np.zeros(5)
    phase_record.formulagrad(cpu_grad, dof)
    
    print(f"  CPU energy: {cpu_energy[0]:.6f}")
    print(f"  CPU gradient shape: {cpu_grad.shape}")
    print(f"  CPU gradient values: {cpu_grad}")
    
    # Check if gradient is all zeros (which would be a problem)
    if np.all(np.abs(cpu_grad) < 1e-10):
        print("  ✗ WARNING: Gradient is all zeros!")
    else:
        # Check gradient values
        print(f"  CPU gradient components:")
        for i, name in enumerate(['dG/dN', 'dG/dP', 'dG/dT', 'dG/dY(NB)', 'dG/dY(TI)']):
            print(f"    {name} = {cpu_grad[i]:.6f}")

# Check if the two phases have different energies
print("\n" + "="*80)
print("Phase energy comparison:")
dof1 = np.array([1.0, 101325.0, 500.0, 0.94085304, 0.05914696])
dof2 = np.array([1.0, 101325.0, 500.0, 0.05914696, 0.94085304])

energy1 = np.zeros(1)
energy2 = np.zeros(1)
phase_record.obj(energy1, dof1)
phase_record.obj(energy2, dof2)

print(f"Ti-poor phase energy: {energy1[0]:.6f}")
print(f"Ti-rich phase energy: {energy2[0]:.6f}")
print(f"Energy difference: {abs(energy1[0] - energy2[0]):.6f}")

if abs(energy1[0] - energy2[0]) > 1000:
    print("✓ Phases have significantly different energies (expected for miscibility gap)")
else:
    print("✗ Phases have similar energies (unexpected)")

print("="*80)