#\!/usr/bin/env python
"""Test to capture GPU constraint calculation debug output"""

import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, "/mnt/c/users/scott/Documents/pycalphad")

from pycalphad import Database, equilibrium
import pycalphad.variables as v

dbf = Database("Al-Cu-Fe.tdb")

# Test with ternary system - AL-CU-FE with only LIQUID phase
print("=" * 60)
print("Testing AL-CU-FE ternary system with GPU (LIQUID only)")
print("=" * 60)

try:
    result = equilibrium(dbf, ["AL","CU","FE","VA"], ["LIQUID"], 
                        {"T": 973.15, "P": 101325, "X_AL": 0.5, "X_CU": 0.2}, 
                        verbose=True, calc_opts={"pdens": 50}, gpu=True)
    print("GPU calculation completed successfully")
except Exception as e:
    print(f"GPU calculation failed: {e}")
    import traceback
    traceback.print_exc()
