#!/usr/bin/env python3
"""Test CPU hessian calculation directly"""

import sys
sys.path.insert(0, '.')

from pycalphad import Database, Model
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
import numpy as np

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = list(db.phases.keys())

# Build models for all phases
models = {phase_name: Model(db, comps, phase_name) for phase_name in phases}

# Create phase record factory
prf = PhaseRecordFactory(db, comps, {'N': 1.0, 'P': 101325.0, 'T': 1000.0}, models)

# Get phase record for BCC_A2
bcc_record = prf.get_phase_property('BCC_A2', 'G', include_grad=True, include_hess=True)

# Test with workspace DOF
workspace_dof = np.array([1.0, 101325.0, 1000.0, 0.612245, 0.387755])

# Allocate output array for hessian (5x5 for 3 state vars + 2 site fractions)
hess_out = np.zeros((5, 5), order='C')

# Call the hessian function
# Note: phase_rec.pyx expects formulahess which is the compiled function
print("Testing CPU hessian calculation...")

# bcc_record is a BuildFunctionsResult with func, grad, hess attributes
if bcc_record.hess is not None:
    # This is the compiled hessian function
    hess_flat = np.zeros(25)  # Flattened hessian
    bcc_record.hess(hess_flat, workspace_dof)
    
    # Reshape to 5x5
    hess_out = hess_flat.reshape((5, 5))
    
    print(f"Hessian at workspace DOF {workspace_dof}:")
    print(f"Full Hessian matrix:")
    for i in range(5):
        for j in range(5):
            print(f"  H[{i},{j}] = {hess_out[i,j]:.6e}", end='')
        print()
    
    print(f"\nSite fraction block (lower right 2x2):")
    print(f"H[3,3] = {hess_out[3,3]:.6e}")
    print(f"H[3,4] = {hess_out[3,4]:.6e}")
    print(f"H[4,3] = {hess_out[4,3]:.6e}")
    print(f"H[4,4] = {hess_out[4,4]:.6e}")
else:
    print("Hessian function not found in bcc_record")