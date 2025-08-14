#!/usr/bin/env python
"""Test to debug thread indexing issues in GPU kernel."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def test_simple_conditions():
    """Test with very simple distinct conditions to trace the issue."""
    
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['LIQUID']  # Just one phase to simplify
    
    print("=" * 80)
    print("TESTING THREAD INDEXING WITH SIMPLE CONDITIONS")
    print("=" * 80)
    
    # Create 3 very distinct conditions
    test_conditions = [
        (0.1, 0.1, 600),   # Low Al, Low Cu, Low T
        (0.5, 0.2, 900),   # Mid Al, Low Cu, Mid T
        (0.3, 0.4, 1200),  # Low Al, Mid Cu, High T
    ]
    
    # Test individually
    print("\n1. Individual results:")
    individual_results = []
    for idx, (x_al, x_cu, temp) in enumerate(test_conditions):
        conditions = {
            v.X('AL'): x_al,
            v.X('CU'): x_cu,
            v.T: temp,
            v.P: 101325
        }
        
        result = equilibrium(dbf, comps, phases, conditions,
                           calc_opts={'pdens': 50},
                           gpu=True, verbose=False)
        gm = result.GM.values.item()
        mu_al = result.MU.sel(component='AL').values.item()
        mu_cu = result.MU.sel(component='CU').values.item()
        individual_results.append((gm, mu_al, mu_cu))
        print(f"  Condition {idx}: X(AL)={x_al:.1f}, X(CU)={x_cu:.1f}, T={temp}K")
        print(f"    GM={gm:.2f}, MU(AL)={mu_al:.2f}, MU(CU)={mu_cu:.2f}")
    
    # Test in batch - use numpy arrays to create exact point conditions
    # For exact points (not a grid), we need to pass numpy arrays
    print("\n2. Batch results:")
    # Create conditions that will be processed as 3 specific points
    import numpy as np
    conditions_batch = {
        v.X('AL'): np.array([0.1, 0.5, 0.3]),  # numpy array for exact points
        v.X('CU'): np.array([0.1, 0.2, 0.4]),  # numpy array for exact points
        v.T: np.array([600, 900, 1200]),       # numpy array for exact points
        v.P: 101325
    }
    
    result_batch = equilibrium(dbf, comps, phases, conditions_batch,
                             calc_opts={'pdens': 50},
                             gpu=True, verbose=False)  # Disable verbose to avoid broken pipe
    
    batch_gm = result_batch.GM.values.flatten()
    batch_mu_al = result_batch.MU.sel(component='AL').values.flatten()
    batch_mu_cu = result_batch.MU.sel(component='CU').values.flatten()
    
    # Batch should have the same conditions as individual tests
    for idx, (x_al, x_cu, temp) in enumerate(test_conditions):
        print(f"  Condition {idx}: X(AL)={x_al:.1f}, X(CU)={x_cu:.1f}, T={temp}K")
        if idx < len(batch_gm):
            print(f"    GM={batch_gm[idx]:.2f}, MU(AL)={batch_mu_al[idx]:.2f}, MU(CU)={batch_mu_cu[idx]:.2f}")
        else:
            print(f"    ERROR: No result for condition {idx}")
    
    # Compare
    print("\n3. Comparison:")
    print("-" * 60)
    all_pass = True
    for idx, (x_al, x_cu, temp) in enumerate(test_conditions):
        ind_gm, ind_mu_al, ind_mu_cu = individual_results[idx]
        batch_gm_val = batch_gm[idx]
        batch_mu_al_val = batch_mu_al[idx]
        batch_mu_cu_val = batch_mu_cu[idx]
        
        gm_diff = abs(batch_gm_val - ind_gm)
        mu_al_diff = abs(batch_mu_al_val - ind_mu_al)
        mu_cu_diff = abs(batch_mu_cu_val - ind_mu_cu)
        
        status = "✓" if gm_diff < 1.0 else "✗"
        if gm_diff >= 1.0:
            all_pass = False
            
        print(f"Condition {idx}:")
        print(f"  GM: Individual={ind_gm:.2f}, Batch={batch_gm_val:.2f}, Diff={gm_diff:.2f} {status}")
        print(f"  MU(AL): Individual={ind_mu_al:.2f}, Batch={batch_mu_al_val:.2f}, Diff={mu_al_diff:.2f}")
        print(f"  MU(CU): Individual={ind_mu_cu:.2f}, Batch={batch_mu_cu_val:.2f}, Diff={mu_cu_diff:.2f}")
        
        # Check if this batch result matches a different individual result
        for check_idx, (check_gm, check_mu_al, check_mu_cu) in enumerate(individual_900k_results):
            if check_idx != idx and abs(batch_gm_val - check_gm) < 1.0:
                print(f"  *** WARNING: Batch result matches condition {check_idx} instead!")
    
    print("\n" + "=" * 80)
    if all_pass:
        print("✓ ALL CONDITIONS MATCH")
    else:
        print("✗ THREAD INDEXING ISSUE DETECTED")
    print("=" * 80)

if __name__ == "__main__":
    test_simple_conditions()