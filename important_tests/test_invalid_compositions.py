#!/usr/bin/env python
"""Test how GPU handles invalid compositions in a grid."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def main():
    """Test with a grid that includes invalid compositions."""
    
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['LIQUID']
    
    print("=" * 80)
    print("TESTING INVALID COMPOSITIONS IN GRID")
    print("=" * 80)
    
    # Create a 3x3 grid at T=900K
    # Some compositions will be invalid (X_AL + X_CU >= 1.0)
    batch_cond = {
        v.X('AL'): [0.2, 0.5, 0.7],
        v.X('CU'): [0.2, 0.4, 0.6],
        v.T: 900,
        v.P: 101325
    }
    
    print("\nGrid compositions (3x3):")
    print("X_AL \\ X_CU |  0.2  |  0.4  |  0.6")
    print("------------|-------|-------|-------")
    
    valid_count = 0
    invalid_count = 0
    for x_al in [0.2, 0.5, 0.7]:
        row = f"    {x_al:.1f}    |"
        for x_cu in [0.2, 0.4, 0.6]:
            x_fe = 1.0 - x_al - x_cu
            if x_fe >= 0 and x_fe <= 1.0:
                row += f" VALID |"
                valid_count += 1
            else:
                row += f" INVAL |"
                invalid_count += 1
        print(row)
    
    print(f"\nValid compositions: {valid_count}")
    print(f"Invalid compositions: {invalid_count}")
    
    # Run batch calculation
    print("\nRunning batch calculation...")
    result_batch = equilibrium(dbf, comps, phases, batch_cond, calc_opts={'pdens': 50}, gpu=True, verbose=False)
    
    print(f"Result shape: {result_batch.GM.shape}")
    print(f"Number of results: {result_batch.GM.size}")
    
    gm_batch = result_batch.GM.values.flatten()
    
    # Check results
    print("\nResults:")
    i = 0
    for x_al in [0.2, 0.5, 0.7]:
        for x_cu in [0.2, 0.4, 0.6]:
            x_fe = 1.0 - x_al - x_cu
            if i < len(gm_batch):
                gm = gm_batch[i]
                validity = "VALID" if (x_fe >= 0 and x_fe <= 1.0) else "INVALID"
                if np.isnan(gm):
                    print(f"  [{i}] X_AL={x_al:.1f}, X_CU={x_cu:.1f}, X_FE={x_fe:.1f} ({validity}): GM = NaN")
                else:
                    print(f"  [{i}] X_AL={x_al:.1f}, X_CU={x_cu:.1f}, X_FE={x_fe:.1f} ({validity}): GM = {gm:.2f}")
            i += 1
    
    # Test individual valid points
    print("\nIndividual calculation of valid points:")
    i = 0
    for x_al in [0.2, 0.5, 0.7]:
        for x_cu in [0.2, 0.4, 0.6]:
            x_fe = 1.0 - x_al - x_cu
            if x_fe >= 0 and x_fe <= 1.0:
                cond = {v.X('AL'): x_al, v.X('CU'): x_cu, v.T: 900, v.P: 101325}
                result = equilibrium(dbf, comps, phases, cond, calc_opts={'pdens': 50}, gpu=True, verbose=False)
                gm_ind = result.GM.values.item()
                print(f"  X_AL={x_al:.1f}, X_CU={x_cu:.1f}, X_FE={x_fe:.1f}: GM = {gm_ind:.2f}")
                
                # Compare with batch
                if i < len(gm_batch):
                    diff = abs(gm_batch[i] - gm_ind) if not np.isnan(gm_batch[i]) else float('inf')
                    if diff < 1.0:
                        print(f"    ✓ Matches batch result")
                    else:
                        print(f"    ✗ Differs from batch by {diff:.2f}")
            i += 1

if __name__ == "__main__":
    main()