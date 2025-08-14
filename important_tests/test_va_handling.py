#!/usr/bin/env python
"""Test how VA is handled in constraint matrix."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def main():
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['BCC_B2', 'AL5FE2']
    
    conditions = {
        v.X('AL'): 0.60,
        v.X('CU'): 0.10,
        v.T: 600,
        v.P: 101325
    }
    
    print("=" * 80)
    print("VA HANDLING TEST")
    print("=" * 80)
    
    print("\nComponents:", comps)
    print("Component indices:")
    for i, comp in enumerate(comps):
        print(f"  {i}: {comp}")
    
    print("\nConstraint setup:")
    print("  X(AL) = 0.60 -> coefficient[0][0] = 1.0, coefficient[0][3] = 0.0 (VA)")
    print("  X(CU) = 0.10 -> coefficient[1][1] = 1.0, coefficient[1][3] = 0.0 (VA)")
    
    print("\nThe issue:")
    print("  When write_row_fixed_mole_fraction is called with component_idx=3 (VA),")
    print("  it should have prefactor=0.0 for both constraints.")
    print("  But if the masses include VA, this could affect the calculation.")
    
    print("\nRunning CPU calculation...")
    cpu_result = equilibrium(dbf, comps, phases, conditions, gpu=False)
    
    # Check phase compositions
    print("\nCPU Phase compositions:")
    cpu_phases = cpu_result.Phase.values.flatten()
    cpu_np = cpu_result.NP.values.flatten()
    x_vals = cpu_result.X.values.reshape(-1, 3)  # AL, CU, FE
    
    for i, (phase, amount) in enumerate(zip(cpu_phases, cpu_np)):
        if not np.isnan(amount) and amount > 0.001:
            print(f"  {phase}: {amount:.4f}")
            if i < len(x_vals):
                print(f"    X(AL)={x_vals[i][0]:.4f}, X(CU)={x_vals[i][1]:.4f}, X(FE)={x_vals[i][2]:.4f}")
                print(f"    Sum = {x_vals[i].sum():.4f} (should be 1.0)")
    
    print("\nNote: VA doesn't appear in the X values because it's not a component in the mass balance.")
    print("The constraint matrix should only involve AL, CU, FE masses, not VA.")

if __name__ == "__main__":
    main()