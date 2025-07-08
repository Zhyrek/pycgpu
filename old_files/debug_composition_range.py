#!/usr/bin/env python
"""
Test different composition ranges to find the problematic region.
"""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings('ignore')

def test_composition_range(x_range, description):
    print(f"\n{description}:")
    print(f"  X_TI range: {x_range[0]} to {x_range[-1]} ({len(x_range)} points)")
    
    dbf = Database("NbTi.tdb")
    conditions = {v.X("TI"): x_range, v.T: [600, 700, 800]}  # 3 temperatures
    
    try:
        gpu_result = equilibrium(dbf, ['NB', 'TI', 'VA'], ['LIQUID', 'BCC_A2'], 
                                conditions, gpu=True, verbose=False)
        total_conditions = len(x_range) * 3
        print(f"  ✅ SUCCESS: {total_conditions} conditions calculated")
        return True
        
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False

def main():
    print("Composition Range Debug")
    print("="*50)
    
    # Test progressively more challenging ranges
    test_composition_range([0.1, 0.3, 0.5, 0.7, 0.9], "Safe middle range")
    test_composition_range([0.05, 0.2, 0.4, 0.6, 0.8, 0.95], "Wider range")
    test_composition_range([0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99], "Near-edge range")
    test_composition_range([1e-3, 0.1, 0.5, 0.9, 0.999], "Very dilute included")
    test_composition_range([1e-6, 0.1, 0.5, 0.9, 0.999999], "Ultra-dilute included")
    
    # Test the problematic test_script.py range
    print(f"\ntest_script.py problematic range:")
    x_range = np.linspace(1e-10, 0.95, 20)
    print(f"  X_TI: {x_range[0]:.2e} to {x_range[-1]:.3f} ({len(x_range)} points)")
    print(f"  Most dilute compositions: {x_range[:3]}")
    print(f"  Most concentrated: {x_range[-3:]}")
    
    dbf = Database("NbTi.tdb")
    conditions = {v.X("TI"): x_range, v.T: [600, 700, 800]}
    
    try:
        print(f"\nTrying with verbose output...")
        gpu_result = equilibrium(dbf, ['NB', 'TI', 'VA'], ['LIQUID', 'BCC_A2'], 
                                conditions, gpu=True, verbose=True)
                                
    except Exception as e:
        print(f"  ❌ FAILED: {e}")

if __name__ == "__main__":
    main()