#!/usr/bin/env python
"""Simple test to verify padding is applied."""

import pycalphad.gpu.gpu_equilibrium as gpu_eq

# Test the struct creation function directly
import numpy as np

# Set up test data
num_conditions = 1
dynamic_sizes = {
    'MAX_PHASES': 6,
    'MAX_COMPONENTS': 4,
    'MAX_DOF_PER_PHASE': 4
}

# Create dummy arrays
initial_phase_data_arrays = {
    'phase_indices': np.zeros((num_conditions, 6), dtype=np.int32),
    'phase_amounts': np.zeros((num_conditions, 6), dtype=np.float64),
    'site_fractions': np.zeros((num_conditions, 6, 4), dtype=np.float64),
    'compositions': np.zeros((num_conditions, 6, 4), dtype=np.float64),
    'chemical_potentials': np.zeros((num_conditions, 4), dtype=np.float64),
    'num_phases': np.zeros(num_conditions, dtype=np.int32)
}

print("Testing padding in _create_initial_phase_data_struct_array...")
print("="*60)

# Call the function
result = gpu_eq._create_initial_phase_data_struct_array(
    initial_phase_data_arrays, 
    num_conditions, 
    dynamic_sizes, 
    verbose=True
)

print(f"\nResult shape: {result.shape}")
print(f"Expected without padding: (1, 65)")
print(f"Expected with padding: (1, 80)")

if result.shape[1] == 80:
    print("\n✓ SUCCESS: Padding is correctly applied!")
else:
    print(f"\n✗ FAIL: Padding not applied, got {result.shape[1]} doubles per struct")

# Also calculate what the size should be manually
MAX_PHASES = 6
MAX_COMPONENTS = 4
MAX_DOF_PER_PHASE = 4
expected = MAX_PHASES + MAX_PHASES + (MAX_PHASES * MAX_DOF_PER_PHASE) + (MAX_PHASES * MAX_COMPONENTS) + MAX_COMPONENTS + 1
print(f"\nManual calculation: {MAX_PHASES} + {MAX_PHASES} + {MAX_PHASES*MAX_DOF_PER_PHASE} + {MAX_PHASES*MAX_COMPONENTS} + {MAX_COMPONENTS} + 1 = {expected}")