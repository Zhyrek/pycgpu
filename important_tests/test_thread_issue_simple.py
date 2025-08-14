#!/usr/bin/env python
"""Simple test to demonstrate the thread indexing issue."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def main():
    """Test thread indexing with a simple 2-condition batch."""
    
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['LIQUID']
    
    print("=" * 80)
    print("SIMPLE THREAD INDEXING TEST")
    print("=" * 80)
    
    # Test 1: Two individual conditions
    print("\n1. Individual conditions:")
    cond1 = {v.X('AL'): 0.2, v.X('CU'): 0.2, v.T: 800, v.P: 101325}
    cond2 = {v.X('AL'): 0.4, v.X('CU'): 0.3, v.T: 1000, v.P: 101325}
    
    result1 = equilibrium(dbf, comps, phases, cond1, calc_opts={'pdens': 50}, gpu=True, verbose=False)
    result2 = equilibrium(dbf, comps, phases, cond2, calc_opts={'pdens': 50}, gpu=True, verbose=False)
    
    gm1 = result1.GM.values.item()
    gm2 = result2.GM.values.item()
    
    print(f"  Condition 1 (X_AL=0.2, X_CU=0.2, T=800): GM = {gm1:.2f}")
    print(f"  Condition 2 (X_AL=0.4, X_CU=0.3, T=1000): GM = {gm2:.2f}")
    
    # Test 2: Batch with fixed temperature (creates 2x2=4 grid)
    print("\n2. Batch with lists (creates 2x2 grid):")
    batch_cond = {
        v.X('AL'): [0.2, 0.4],
        v.X('CU'): [0.2, 0.3],
        v.T: 900,  # Fixed
        v.P: 101325
    }
    
    result_batch = equilibrium(dbf, comps, phases, batch_cond, calc_opts={'pdens': 50}, gpu=True, verbose=False)
    
    print(f"  Result shape: {result_batch.GM.shape}")
    print(f"  Number of results: {result_batch.GM.size}")
    
    gm_batch = result_batch.GM.values.flatten()
    
    # The grid will be:
    # [0]: X_AL=0.2, X_CU=0.2, T=900
    # [1]: X_AL=0.2, X_CU=0.3, T=900
    # [2]: X_AL=0.4, X_CU=0.2, T=900
    # [3]: X_AL=0.4, X_CU=0.3, T=900
    
    print("\n  Grid points created:")
    grid_points = [
        (0.2, 0.2, 900),
        (0.2, 0.3, 900),
        (0.4, 0.2, 900),
        (0.4, 0.3, 900)
    ]
    
    for i, (x_al, x_cu, t) in enumerate(grid_points):
        if i < len(gm_batch):
            print(f"    [{i}] X_AL={x_al}, X_CU={x_cu}, T={t}: GM = {gm_batch[i]:.2f}")
    
    # Test 3: Run the grid points individually for comparison
    print("\n3. Individual runs of grid points:")
    for i, (x_al, x_cu, t) in enumerate(grid_points):
        cond = {v.X('AL'): x_al, v.X('CU'): x_cu, v.T: t, v.P: 101325}
        result = equilibrium(dbf, comps, phases, cond, calc_opts={'pdens': 50}, gpu=True, verbose=False)
        gm = result.GM.values.item()
        print(f"    [{i}] X_AL={x_al}, X_CU={x_cu}, T={t}: GM = {gm:.2f}")
        
        # Check if batch matches individual
        if i < len(gm_batch):
            diff = abs(gm_batch[i] - gm)
            status = "✓" if diff < 1.0 else "✗"
            if diff >= 1.0:
                print(f"        Difference: {diff:.2f} {status}")
    
    print("\n" + "=" * 80)
    print("If batch results don't match individual results, there's a thread indexing issue.")
    print("=" * 80)

if __name__ == "__main__":
    main()