#!/usr/bin/env python
"""Test kernel caching between runs"""

import sys
import os
import time
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')

from pycalphad import Database, equilibrium, variables as v

def main():
    dbf = Database('/mnt/c/users/scott/Documents/pycalphad/Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    all_phases = list(dbf.phases.keys())

    # Fixed condition for testing
    conditions = {v.X('AL'): 0.2, v.X('CU'): 0.1, v.T: 700, v.P: 101325}

    print("=" * 80)
    print("KERNEL CACHING TEST")
    print("=" * 80)
    print(f"Testing with {len(all_phases)} phases from Al-Cu-Fe database")
    print(f"Condition: X(AL)={conditions[v.X('AL')]:.2f}, X(CU)={conditions[v.X('CU')]:.2f}, T={conditions[v.T]:.0f}K")
    print()

    # Check for existing kernel files
    import glob
    existing_kernels = glob.glob("generated_equilibrium_kernel*.cu")
    if existing_kernels:
        print(f"Found {len(existing_kernels)} existing kernel files:")
        for kf in existing_kernels[:3]:
            print(f"  - {kf}")
    else:
        print("No existing kernel files found")
    print()

    # Run 3 times
    for run in range(1, 4):
        print(f"Run {run}:")
        print("-" * 40)

        # Clear any in-memory cache by restarting (if needed)
        if run == 2:
            # Force a new process state
            import importlib
            import pycalphad.gpu.gpu_equilibrium as gpu_eq
            if hasattr(gpu_eq, '_gpu_module_cache'):
                print(f"  In-memory cache has {len(gpu_eq._gpu_module_cache)} entries")
                # Don't clear it - we want to test in-session caching

        start = time.time()
        result = equilibrium(dbf, comps, all_phases, conditions,
                           calc_opts={'pdens': 50},
                           gpu=True,
                           verbose=True)
        elapsed = time.time() - start

        print(f"  Time: {elapsed:.2f}s")
        print(f"  GM: {float(result.GM.values):.2f} J/mol")

        # Check for new kernel files
        new_kernels = glob.glob("generated_equilibrium_kernel*.cu")
        if len(new_kernels) > len(existing_kernels):
            print(f"  NEW kernel file created!")
            existing_kernels = new_kernels
        else:
            print(f"  No new kernel files (reused existing)")
        print()

if __name__ == '__main__':
    main()