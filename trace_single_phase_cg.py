#!/usr/bin/env python
"""Trace c_G calculation for single phase."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory

# Directly calculate what c_G should be for single phase
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']
conditions = {v.T: 600, v.P: 101325, v.N: 1}

# Create phase record
prf = PhaseRecordFactory(dbf, comps, {v.T, v.P, v.N}, parameters={})
bcc_rec = prf['BCC_A2']

# Set up single phase with Y(TI) = 0.90314714 (from CPU trace)
state_vars = np.array([1., 101325., 600.])
y_ti = 0.90314714
site_fracs = np.array([1-y_ti, y_ti])  # Y(NB), Y(TI) for single sublattice

print("SINGLE PHASE c_G CALCULATION")
print("=" * 60)
print(f"Site fractions: Y(NB)={1-y_ti:.8f}, Y(TI)={y_ti:.8f}")
print(f"Mole fractions: X(NB)={1-y_ti:.8f}, X(TI)={y_ti:.8f}")

# Calculate energy and derivatives
energy = float(bcc_rec.obj(site_fracs, state_vars))
grad = np.zeros(5)
bcc_rec.grad(grad, site_fracs, state_vars)

print(f"\nEnergy: {energy:.6f}")
print(f"Gradient: {grad}")

# Calculate Hessian
hess = np.zeros((5, 5))
bcc_rec.formulahess(hess, site_fracs, state_vars)
print(f"\nHessian:")
print(f"  H[3,3] = {hess[3,3]:.6e}")
print(f"  H[3,4] = {hess[3,4]:.6e}")  
print(f"  H[4,3] = {hess[4,3]:.6e}")
print(f"  H[4,4] = {hess[4,4]:.6e}")

# Calculate c_component (dc/dy)
c_component = np.zeros((2, 2))
bcc_rec.formulamole_obj(c_component, site_fracs, state_vars, bcc_rec._obj_parameters)
print(f"\nc_component (dc/dy):")
print(f"  c[0,0] = {c_component[0,0]:.6e}")
print(f"  c[0,1] = {c_component[0,1]:.6e}")
print(f"  c[1,0] = {c_component[1,0]:.6e}")
print(f"  c[1,1] = {c_component[1,1]:.6e}")

# Calculate mass_jac = c_component @ dof_2d_to_1d
# For single sublattice: dof_2d_to_1d = [[1, 0], [0, 1]]
mass_jac = c_component
print(f"\nmass_jac:")
print(f"  [0,:] = {mass_jac[0,:]}")
print(f"  [1,:] = {mass_jac[1,:]}")

# Calculate e_matrix (phase amount derivatives)
# e_ij = sum_k (mass_jac[i,k] * inv_hess[k,j])
# For 2 components, 2 DOF, the phase Hessian is 2x2 at indices [3:5, 3:5]
phase_hess = hess[3:5, 3:5]
print(f"\nPhase Hessian (2x2):")
print(phase_hess)

# Invert it
try:
    inv_hess = np.linalg.inv(phase_hess)
    print(f"\nInverse Hessian:")
    print(inv_hess)
    
    # Calculate e_matrix
    e_matrix = mass_jac @ inv_hess
    print(f"\ne_matrix:")
    print(e_matrix)
    
    # c_G = -e_matrix @ grad[3:5]
    phase_grad = grad[3:5]
    c_G = -e_matrix @ phase_grad
    print(f"\nc_G calculation:")
    print(f"  phase_grad = {phase_grad}")
    print(f"  c_G = -e_matrix @ phase_grad = {c_G}")
    
except np.linalg.LinAlgError:
    print("ERROR: Phase Hessian is singular!")
    
print(f"\nExpected from CPU trace: c_G ≈ [0.20879587, -0.20879587]")
print("Note: CPU shows gradient [-19614.94249925 -13154.5657734], which are the chemical potentials")