#!/usr/bin/env python
"""Verify that padding is actually applied in GPU equilibrium calculation."""

import numpy as np
from pycalphad import equilibrium, Database
import os

# Find the database file
for root, dirs, files in os.walk('.'):
    if 'aubi.tdb' in files:
        db_path = os.path.join(root, 'aubi.tdb')
        print(f"Found database at: {db_path}")
        break
else:
    print("Warning: Could not find aubi.tdb, using inline database")
    from pycalphad.tests.fixtures import Database as TestDB
    from pycalphad.tests.fixtures import AUBI_TDB
    dbf = TestDB(AUBI_TDB)
    db_path = None

if db_path:
    dbf = Database(db_path)

# Test with 6 phases (problematic case)
phases = ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'HCP_A3', 'RHOMBOHEDRAL_A7', 'BCC_A2']
comps = ['AU', 'BI', 'VA']

# Single test condition
conditions = {
    'T': 500,
    'P': 101325,
    'X(BI)': 0.3
}

print("Testing padding verification...")
print("="*60)

# Run with GPU and enable verbose mode
# Monkey-patch the function to check the stride
import pycalphad.gpu.gpu_equilibrium as gpu_eq

original_func = gpu_eq._create_initial_phase_data_struct_array

def patched_func(*args, **kwargs):
    result = original_func(*args, **kwargs)
    print(f"\n*** PADDING VERIFICATION ***")
    print(f"Result shape: {result.shape}")
    print(f"Expected shape without padding: (1, 65)")
    print(f"Expected shape with padding: (1, 80)")
    if result.shape[1] == 80:
        print("✓ PADDING IS APPLIED CORRECTLY!")
    else:
        print("✗ PADDING NOT APPLIED - stride is", result.shape[1])
    print("***************************\n")
    return result

gpu_eq._create_initial_phase_data_struct_array = patched_func

try:
    # Run equilibrium calculation with GPU
    eq_result = equilibrium(dbf, comps, phases, conditions, 
                           calc_opts={'pdens': 50},
                           verbose=True, gpu=True)
    print("\nEquilibrium calculation completed successfully")
except Exception as e:
    print(f"\nError during calculation: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Restore original function
    gpu_eq._create_initial_phase_data_struct_array = original_func