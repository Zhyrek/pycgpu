#!/usr/bin/env python
"""
Simple multi-condition test of Al-Cu-Fe system with GPU.
"""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
import time

warnings.filterwarnings('ignore')

def test_alcufe_multi():
    """Test Al-Cu-Fe system with multiple conditions."""
    
    print("=" * 60)
    print("Al-Cu-Fe MULTI-CONDITION GPU TEST")
    print("=" * 60)
    
    # Load database
    db = Database('Al-Cu-Fe.tdb')
    all_phases = list(db.phases.keys())
    print(f"\nNumber of phases: {len(all_phases)}")
    
    # Test 1: Multiple conditions at once
    print("\n" + "-" * 40)
    print("TEST: 10 CONDITIONS SIMULTANEOUSLY")
    print("-" * 40)
    
    n_points = 10
    cu_fractions = np.linspace(0.1, 0.4, n_points)
    fe_fractions = np.linspace(0.1, 0.4, n_points)
    
    conditions_multi = {
        v.T: 900 * np.ones(n_points),
        v.P: 101325 * np.ones(n_points),
        v.X('CU'): cu_fractions,
        v.X('FE'): fe_fractions
    }
    
    print(f"Testing {n_points} conditions:")
    for i in range(min(3, n_points)):
        print(f"  Point {i+1}: X(CU)={cu_fractions[i]:.2f}, X(FE)={fe_fractions[i]:.2f}, X(AL)={1-cu_fractions[i]-fe_fractions[i]:.2f}")
    if n_points > 3:
        print(f"  ... and {n_points-3} more points")
    
    try:
        start = time.time()
        result = equilibrium(db, ['AL', 'CU', 'FE'], all_phases,
                           conditions_multi, verbose=False, gpu=True,
                           calc_opts={'pdens': 60})
        gpu_time = time.time() - start
        
        print(f"\nResults:")
        print(f"  Total GPU time: {gpu_time:.3f}s")
        print(f"  Time per condition: {gpu_time/n_points:.4f}s")
        
        # Show results for first few points
        gm_values = result.GM.values.flatten()[:n_points]
        
        print(f"\n  GM values (J/mol):")
        for i in range(min(5, n_points)):
            print(f"    Point {i+1}: {gm_values[i]:.2f}")
        
        print(f"\n  GM statistics:")
        print(f"    Min: {np.min(gm_values):.2f}")
        print(f"    Max: {np.max(gm_values):.2f}")
        print(f"    Mean: {np.mean(gm_values):.2f}")
        
        # Check phases for first point
        print(f"\n  Phases at point 1:")
        for j in range(len(all_phases)):
            amount = result.NP.values[0, 0, 0, 0, 0, j]
            if amount > 1e-6:
                phase = result.Phase.values[0, 0, 0, 0, 0, j]
                if phase and phase != '':
                    print(f"    {phase}: {amount:.4f}")
        
        print("\n  ✓ Multi-condition test PASSED")
        
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
    # Test 2: Verify results match single calculations
    print("\n" + "-" * 40)
    print("VERIFICATION: Compare multi vs single")
    print("-" * 40)
    
    # Pick first condition to verify
    single_cond = {
        v.T: 900,
        v.P: 101325,
        v.X('CU'): cu_fractions[0],
        v.X('FE'): fe_fractions[0]
    }
    
    print(f"Comparing first condition: X(CU)={cu_fractions[0]:.2f}, X(FE)={fe_fractions[0]:.2f}")
    
    try:
        result_single = equilibrium(db, ['AL', 'CU', 'FE'], all_phases,
                                  single_cond, verbose=False, gpu=True,
                                  calc_opts={'pdens': 60})
        
        gm_single = float(result_single.GM.values[0])
        gm_multi_first = gm_values[0]
        
        print(f"  Single calculation GM: {gm_single:.2f}")
        print(f"  Multi calculation GM:  {gm_multi_first:.2f}")
        print(f"  Difference: {abs(gm_single - gm_multi_first):.6f}")
        
        if abs(gm_single - gm_multi_first) < 0.01:
            print("  ✓ Results match!")
        else:
            print("  ✗ Results differ significantly")
            
    except Exception as e:
        print(f"  ✗ Verification failed: {e}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    test_alcufe_multi()