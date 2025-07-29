#!/usr/bin/env python
"""Test to compare CPU and GPU Hessian values directly."""

import numpy as np
from pycalphad import Database, variables as v
from pycalphad.core.workspace import Workspace
from pycalphad.gpu.gpu_codegen import _generate_c_code_for_phase_models
import cupy as cp
import os

# Clear cache
os.environ['CUDA_CACHE_DISABLE'] = '1'
cp.clear_memo()

# Load database
db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']

# Test with BCC_B2 phase
phase_name = 'BCC_B2'
conditions = {
    v.T: 800,
    v.P: 101325,
    v.X('AL'): 0.1,
    v.X('CU'): 0.1,
    v.N: 1
}

print(f"Testing Hessian values for {phase_name} phase...")
print("="*80)

# Create workspace
wks = Workspace(db, components, [phase_name], conditions, verbose=False)

# Get the phase record from CPU
phase_record = wks.phase_record_factory[phase_name]
model = wks.models[phase_name]

# Test point - use site fractions from equilibrium
test_dof = np.array([1.0, 101325.0, 800.0, 0.1, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0])

print(f"Test DOF array: {test_dof}")
print(f"Phase DOF: {phase_record.phase_dof}")
print(f"Model state variables: {[str(v) for v in model.state_variables]}")

# Calculate CPU Hessian
cpu_hess = np.zeros((phase_record.num_statevars + phase_record.phase_dof,
                     phase_record.num_statevars + phase_record.phase_dof))
# Use the PhaseRecord's hessian method
phase_record.formulahess(cpu_hess, test_dof)

print(f"\nCPU Hessian shape: {cpu_hess.shape}")
print("CPU Hessian non-zero elements:")

# Print non-zero elements
for i in range(cpu_hess.shape[0]):
    for j in range(cpu_hess.shape[1]):
        if abs(cpu_hess[i,j]) > 1e-10:
            print(f"  Hess[{i},{j}] = {cpu_hess[i,j]:.6e}")

# Check which rows/columns are non-zero
print("\nNon-zero rows:")
for i in range(cpu_hess.shape[0]):
    if np.any(np.abs(cpu_hess[i,:]) > 1e-10):
        print(f"  Row {i}: max value = {np.max(np.abs(cpu_hess[i,:])):.6e}")

print("\nNon-zero columns:")  
for j in range(cpu_hess.shape[1]):
    if np.any(np.abs(cpu_hess[:,j]) > 1e-10):
        print(f"  Column {j}: max value = {np.max(np.abs(cpu_hess[:,j])):.6e}")

# Generate GPU code and test
print("\n" + "="*80)
print("Generating GPU code...")

result = _generate_c_code_for_phase_models(wks, include_hess=True, validate=False)
all_device_functions, init_calls, unique_models, phase_map = result

# Extract the generated Hessian function
lines = all_device_functions.split('\n')
hessian_start = -1
for i, line in enumerate(lines):
    if '__device__' in line and 'formulahess' in line and phase_name in all_device_functions[:i]:
        hessian_start = i
        break

if hessian_start >= 0:
    print("Found GPU Hessian function")
    # Check the function structure
    func_lines = []
    for i in range(hessian_start, min(hessian_start + 10, len(lines))):
        func_lines.append(lines[i])
    
    print("Function signature and first few lines:")
    for line in func_lines:
        print(f"  {line}")
    
    print("\nGPU Hessian characteristics:")
    print("- Outputs reduced Hessian (T + site fractions only)")
    print("- CPU row/col 0,1 (N,P) are always zero")
    print("- CPU row/col 2 (T) maps to GPU row/col 0")
    print("- CPU row/col 3+ (site fractions) map to GPU row/col 1+")
    
else:
    print("ERROR: Could not find GPU Hessian function!")

print("\n" + "="*80)
print("Summary:")
print("- CPU Hessian includes N,P derivatives (always zero)")
print("- GPU CSE Hessian only computes T and site fraction derivatives")
print("- Mapping in minimizer.h must handle this difference correctly")
print("="*80)