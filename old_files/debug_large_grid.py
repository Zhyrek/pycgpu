#!/usr/bin/env python
"""
Debug large condition grids like test_script.py.
"""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings('ignore')

def test_grid_size(temperatures, compositions, description):
    print(f"\n{description}:")
    print(f"  Conditions: {len(temperatures)} temps × {len(compositions)} comps = {len(temperatures) * len(compositions)} total")
    
    dbf = Database("NbTi.tdb")
    conditions = {v.T: temperatures, v.X('TI'): compositions}
    
    try:
        gpu_result = equilibrium(dbf, ['NB', 'TI'], ['BCC_A2'], conditions, gpu=True, verbose=False)
        print(f"  ✅ SUCCESS: GPU calculation completed")
        print(f"  Result GM shape: {gpu_result.GM.shape}")
        print(f"  Sample GM values: {gpu_result.GM.values.flatten()[:3]}")
        return True
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False

def main():
    print("Large Grid Debug - Testing different grid sizes")
    print("="*60)
    
    # Test progressively larger grids
    test_grid_size([800, 900], [0.1, 0.5], "2×2 = 4 conditions")
    test_grid_size([800, 900], [0.1, 0.3, 0.5, 0.7], "2×4 = 8 conditions") 
    test_grid_size([700, 800, 900], [0.1, 0.3, 0.5, 0.7], "3×4 = 12 conditions")
    test_grid_size([600, 700, 800, 900], [0.1, 0.3, 0.5, 0.7], "4×4 = 16 conditions")
    test_grid_size([500, 600, 700, 800, 900], [0.1, 0.3, 0.5, 0.7], "5×4 = 20 conditions")
    
    # Test the exact test_script.py conditions
    temperatures = [500, 600, 700, 800, 900]
    compositions = np.linspace(1e-10, 0.95, 20)
    
    print(f"\ntest_script.py exact conditions:")
    print(f"  Temperatures: {temperatures}")
    print(f"  Compositions: {len(compositions)} from {compositions[0]:.2e} to {compositions[-1]:.2f}")
    print(f"  Total conditions: {len(temperatures) * len(compositions)} = 100")
    
    try:
        dbf = Database("NbTi.tdb")
        conditions = {v.T: temperatures, v.X('TI'): compositions}
        
        # Try with verbose to see what fails
        print(f"\nRunning 100-condition test with verbose output...")
        gpu_result = equilibrium(dbf, ['NB', 'TI'], ['BCC_A2'], conditions, gpu=True, verbose=True)
        
    except Exception as e:
        print(f"  ❌ test_script.py conditions FAILED: {e}")

if __name__ == "__main__":
    main()