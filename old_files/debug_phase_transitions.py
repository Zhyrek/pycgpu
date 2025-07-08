#!/usr/bin/env python
"""
Debug single-phase vs multi-phase GPU calculations.
"""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings('ignore')

def test_phases(phases, description):
    print(f"\n{description}:")
    print(f"  Phases: {phases}")
    
    dbf = Database("NbTi.tdb")
    # Use the exact test_script.py conditions but smaller grid
    conditions = {v.X("TI"): (0.1, 0.9, 0.2), v.T: (600, 800, 100)}  # 5×3 = 15 conditions
    
    try:
        gpu_result = equilibrium(dbf, ['NB', 'TI', 'VA'], phases, conditions, gpu=True, verbose=False)
        print(f"  ✅ SUCCESS: {gpu_result.GM.shape} conditions calculated")
        
        # Check phase fractions
        active_phases = gpu_result.NP.values[gpu_result.NP.values > 1e-6]
        print(f"  Active phase fractions: {len(active_phases)} phases active")
        return True
        
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False

def main():
    print("Phase Transition Debug")
    print("="*50)
    
    # Test different phase combinations
    test_phases(["BCC_A2"], "Single phase (BCC_A2 only)")
    test_phases(["LIQUID"], "Single phase (LIQUID only)")  
    test_phases(["BCC_A2", "LIQUID"], "Two phases (BCC_A2 + LIQUID)")
    
    # Test the exact test_script.py conditions
    print(f"\ntest_script.py exact conditions:")
    
    dbf = Database("NbTi.tdb")
    conditions = {v.X("TI"): (0, 1, 0.05), v.T: (500, 1000, 100)}
    phases = ["LIQUID", "BCC_A2"]
    
    print(f"  Conditions: X_TI=(0,1,0.05), T=(500,1000,100)")
    print(f"  Expected grid: 21 × 6 = 126 conditions")
    print(f"  Phases: {phases}")
    
    try:
        gpu_result = equilibrium(dbf, ['NB', 'TI', 'VA'], phases, conditions, gpu=True, verbose=True)
        print(f"  ✅ SUCCESS!")
        
    except Exception as e:
        print(f"  ❌ FAILED: {e}")

if __name__ == "__main__":
    main()