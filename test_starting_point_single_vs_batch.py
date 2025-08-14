#!/usr/bin/env python
"""Test if starting points match in single vs batch execution for GPU."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def run_test():
    """Test starting point consistency between single and batch execution."""
    
    dbf = Database('Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2', 'L12', 'ALCU_THETA', 'AL13FE4', 'AL5FE2']
    
    # Test condition that fails in batch
    test_conditions = [
        (0.1, 0.1, 600),  # X(AL)=0.1, X(CU)=0.1, T=600K
        (0.3, 0.3, 900),  # X(AL)=0.3, X(CU)=0.3, T=900K
    ]
    
    print("=" * 80)
    print("STARTING POINT COMPARISON: SINGLE vs BATCH")
    print("=" * 80)
    
    for x_al, x_cu, temp in test_conditions:
        x_fe = 1.0 - x_al - x_cu
        print(f"\nCondition: X(AL)={x_al:.2f}, X(CU)={x_cu:.2f}, X(FE)={x_fe:.2f}, T={temp}K")
        print("-" * 60)
        
        # Single condition - GPU
        print("\n1. SINGLE CONDITION (GPU):")
        single_cond = {
            v.X('AL'): x_al,
            v.X('CU'): x_cu,
            v.T: temp,
            v.P: 101325
        }
        
        # Enable debug output to see starting points
        import os
        os.environ['PYCALPHAD_DEBUG_STARTING_POINT'] = '1'
        
        result_single_gpu = equilibrium(dbf, comps, phases, single_cond,
                                        calc_opts={'pdens': 50},
                                        gpu=True, verbose=True)
        
        print(f"   GM = {result_single_gpu.GM.values.item():.6f}")
        
        # Batch of conditions - GPU (including our test condition)
        print("\n2. BATCH CONDITIONS (GPU):")
        batch_cond = {
            v.X('AL'): [x_al, x_al + 0.1] if x_al <= 0.8 else [x_al - 0.1, x_al],
            v.X('CU'): [x_cu, x_cu + 0.1] if x_cu <= 0.8 else [x_cu - 0.1, x_cu],
            v.T: temp,
            v.P: 101325
        }
        
        result_batch_gpu = equilibrium(dbf, comps, phases, batch_cond,
                                       calc_opts={'pdens': 50},
                                       gpu=True, verbose=True)
        
        # Extract result for our test condition (first in batch)
        gm_batch = result_batch_gpu.GM.values[0] if result_batch_gpu.GM.values.ndim > 0 else result_batch_gpu.GM.values.item()
        print(f"   GM (first condition) = {gm_batch:.6f}")
        
        # Compare results
        gm_diff = abs(result_single_gpu.GM.values.item() - gm_batch)
        
        print("\n3. COMPARISON:")
        print(f"   Single GPU GM: {result_single_gpu.GM.values.item():.6f}")
        print(f"   Batch GPU GM:  {gm_batch:.6f}")
        print(f"   Difference:    {gm_diff:.6f} J/mol")
        
        if gm_diff < 1.0:
            print("   ✓ Results match!")
        else:
            print("   ✗ Results DO NOT match - batch execution gives different result!")
            print("   This indicates starting points or memory access differ between single/batch")

if __name__ == "__main__":
    run_test()