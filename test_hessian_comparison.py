#!/usr/bin/env python
"""Compare CPU and GPU Hessian calculations for BCC_A2 phase."""

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
db = Database('NbTi.tdb')
components = ['NB', 'TI', 'VA']

# Test with BCC_A2 phase at the condition
phase_name = 'BCC_A2'
conditions = {
    v.T: 500,
    v.P: 101325,
    v.X('TI'): 0.1,
    v.N: 1
}

print(f"Comparing Hessian calculations for {phase_name} phase")
print(f"Conditions: X(TI)={conditions[v.X('TI')]}, T={conditions[v.T]}K")
print("="*80)

# Create workspace
wks = Workspace(db, components, [phase_name], conditions, verbose=False)

# Get the phase record from CPU
phase_record = wks.phase_record_factory[phase_name]
model = wks.models[phase_name]

# Use the site fractions from the equilibrium calculation
# For X(TI)=0.1 at 500K, we have two compositions in miscibility gap:
# Phase 0: Y(BCC_A2,TI) = 0.05879468
# Phase 1: Y(BCC_A2,TI) = 0.94180762

test_cases = [
    (0.05879468, "Ti-poor phase"),
    (0.94180762, "Ti-rich phase")
]

for y_ti, description in test_cases:
    print(f"\nTesting {description}: Y(TI)={y_ti}")
    print("-"*40)
    
    # Set up DOF array: [N, P, T, Y(NB), Y(TI)]
    test_dof = np.array([1.0, 101325.0, 500.0, 1.0 - y_ti, y_ti])
    
    # Calculate CPU Hessian
    cpu_hess = np.zeros((phase_record.num_statevars + phase_record.phase_dof,
                         phase_record.num_statevars + phase_record.phase_dof))
    phase_record.formulahess(cpu_hess, test_dof)
    
    # Extract the site fraction block (rows/cols 3,4)
    print("CPU Hessian site fraction block:")
    print(f"  H[3,3] = {cpu_hess[3,3]:.6e}")
    print(f"  H[3,4] = {cpu_hess[3,4]:.6e}")
    print(f"  H[4,3] = {cpu_hess[4,3]:.6e}")
    print(f"  H[4,4] = {cpu_hess[4,4]:.6e}")
    
    # Also check temperature derivatives
    if abs(cpu_hess[2,2]) > 1e-10:
        print(f"  H[2,2] (T,T) = {cpu_hess[2,2]:.6e}")
    if abs(cpu_hess[2,3]) > 1e-10:
        print(f"  H[2,3] (T,Y_NB) = {cpu_hess[2,3]:.6e}")
    if abs(cpu_hess[2,4]) > 1e-10:
        print(f"  H[2,4] (T,Y_TI) = {cpu_hess[2,4]:.6e}")

# Now check what the GPU code generates
print("\n" + "="*80)
print("GPU code generation check:")
print("="*80)

# Generate GPU code
result = _generate_c_code_for_phase_models(wks, include_hess=True, validate=False)
all_device_functions, init_calls, unique_models, phase_map = result

# Find the Hessian function
lines = all_device_functions.split('\n')
hessian_found = False
for i, line in enumerate(lines):
    if '__device__' in line and 'formulahess' in line:
        hessian_found = True
        print(f"\nFound GPU Hessian function at line {i}")
        # Check how the output array is filled
        print("Checking output array indexing in GPU function...")
        
        # Look for out[0] = ... pattern (should be T,T)
        # Look for out[1] = ... pattern (should be T,Y_NB)
        # Look for out[2] = ... pattern (should be T,Y_TI or Y_NB,Y_NB depending on layout)
        
        for j in range(i, min(i + 100, len(lines))):
            if 'out[0]' in lines[j]:
                print(f"  Line {j}: {lines[j].strip()}")
                break
        
        # Count total output assignments
        out_count = 0
        for j in range(i, len(lines)):
            if lines[j].strip() == '}':
                break
            if 'out[' in lines[j] and '=' in lines[j]:
                out_count += 1
        
        print(f"  Total output assignments: {out_count}")
        
        # For BCC_A2 with 2 site fractions, reduced Hessian should be 3x3:
        # out[0] = H[0,0] (T,T)
        # out[1] = H[0,1] (T,Y_NB) 
        # out[2] = H[0,2] (T,Y_TI)
        # out[3] = H[1,0] (Y_NB,T)
        # out[4] = H[1,1] (Y_NB,Y_NB)
        # out[5] = H[1,2] (Y_NB,Y_TI)
        # out[6] = H[2,0] (Y_TI,T)
        # out[7] = H[2,1] (Y_TI,Y_NB)
        # out[8] = H[2,2] (Y_TI,Y_TI)
        
        expected_size = 3 * 3  # (1 + phase_dof) squared
        print(f"  Expected output size for reduced Hessian: {expected_size}")
        
        break

if not hessian_found:
    print("ERROR: Could not find GPU Hessian function!")

print("\n" + "="*80)
print("Key mapping:")
print("- CPU H[2,2] (T,T) -> GPU out[0]")
print("- CPU H[2,3] (T,Y_NB) -> GPU out[1]") 
print("- CPU H[2,4] (T,Y_TI) -> GPU out[2]")
print("- CPU H[3,3] (Y_NB,Y_NB) -> GPU out[4]")
print("- CPU H[3,4] (Y_NB,Y_TI) -> GPU out[5]")
print("- CPU H[4,4] (Y_TI,Y_TI) -> GPU out[8]")
print("="*80)