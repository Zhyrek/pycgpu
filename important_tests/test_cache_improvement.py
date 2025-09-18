#!/usr/bin/env python
"""Test that kernel caching works based on phases/components only"""

import sys
import os
import time
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')

from pycalphad import Database, equilibrium, variables as v

def main():
    dbf = Database('/mnt/c/users/scott/Documents/pycalphad/Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    all_phases = list(dbf.phases.keys())

    print("=" * 80)
    print("TESTING CACHE KEY IMPROVEMENT")
    print("=" * 80)
    print(f"Using {len(all_phases)} phases from Al-Cu-Fe database")
    print(f"Components: {', '.join(comps)}")
    print()

    # Test different conditions but same phases/components
    conditions = [
        {v.X('AL'): 0.2, v.X('CU'): 0.1, v.T: 700, v.P: 101325},
        {v.X('AL'): 0.3, v.X('CU'): 0.2, v.T: 800, v.P: 101325},  # Different composition and T
        {v.X('AL'): 0.1, v.X('CU'): 0.3, v.T: 900, v.P: 101325},  # Very different values
    ]

    for i, cond in enumerate(conditions, 1):
        print(f"\nRun {i}: X(AL)={cond[v.X('AL')]:.1f}, X(CU)={cond[v.X('CU')]:.1f}, T={cond[v.T]:.0f}K")
        print("-" * 40)

        start = time.time()
        result = equilibrium(dbf, comps, all_phases, cond,
                           calc_opts={'pdens': 50},
                           gpu=True,
                           verbose=True)
        elapsed = time.time() - start

        print(f"Time: {elapsed:.2f}s")
        print(f"GM: {float(result.GM.values):.2f} J/mol")

        if i == 1:
            print("(First run - should compile kernel)")
        else:
            if elapsed < 5:
                print("✓ FAST - kernel was reused!")
            else:
                print("✗ SLOW - kernel was recompiled")

if __name__ == '__main__':
    main()