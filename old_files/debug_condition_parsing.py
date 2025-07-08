#!/usr/bin/env python
"""
Debug condition parsing for range notation.
"""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.workspace import Workspace
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Condition Parsing Debug")
    print("="*50)
    
    dbf = Database("NbTi.tdb")
    
    # Test different ways to specify conditions
    
    print("\n1. Explicit arrays:")
    x_vals = np.linspace(0, 1, 21)
    t_vals = np.linspace(500, 1000, 6)
    conditions1 = {v.X("TI"): x_vals, v.T: t_vals}
    print(f"  X_TI: {len(x_vals)} values")
    print(f"  T: {len(t_vals)} values") 
    print(f"  Expected total: {len(x_vals) * len(t_vals)} = {len(x_vals) * len(t_vals)}")
    
    # Create workspace to see what it generates
    wks1 = Workspace(database=dbf, components=['NB', 'TI', 'VA'], 
                     phases=['LIQUID', 'BCC_A2'], conditions=conditions1)
    print(f"  Workspace conditions: {len(list(wks1.conditions.keys()))} keys")
    for key, val in wks1.conditions.items():
        print(f"    {key}: {np.array(val).shape} {type(val)}")
    
    print("\n2. Range notation (test_script.py style):")
    conditions2 = {v.X("TI"): (0, 1, 0.05), v.T: (500, 1000, 100)}
    print(f"  X_TI: (0, 1, 0.05)")
    print(f"  T: (500, 1000, 100)")
    
    # Create workspace to see what it generates  
    wks2 = Workspace(database=dbf, components=['NB', 'TI', 'VA'], 
                     phases=['LIQUID', 'BCC_A2'], conditions=conditions2)
    print(f"  Workspace conditions: {len(list(wks2.conditions.keys()))} keys")
    for key, val in wks2.conditions.items():
        print(f"    {key}: {np.array(val).shape} {type(val)}")
        if hasattr(val, '__len__') and len(val) < 10:
            print(f"      values: {val}")
        elif hasattr(val, '__len__'):
            print(f"      range: {val[0]} to {val[-1]} ({len(val)} values)")
    
    # Calculate expected grid size
    x_range = list(wks2.conditions[v.X('TI')])
    t_range = list(wks2.conditions[v.T])
    expected_total = len(x_range) * len(t_range)
    print(f"  Expected total: {len(x_range)} × {len(t_range)} = {expected_total}")

if __name__ == "__main__":
    main()