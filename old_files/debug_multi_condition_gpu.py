#!/usr/bin/env python
"""
Debug multi-condition GPU vs single-condition GPU.
"""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Multi-Condition GPU Debug")
    print("="*50)
    
    dbf = Database("NbTi.tdb") 
    
    print("\n1. Single condition (WORKING):")
    conditions_single = {v.T: 800, v.P: 101325, v.X('TI'): 0.5}
    
    gpu_result_single = equilibrium(dbf, ['NB', 'TI'], ['BCC_A2'], conditions_single, gpu=True, verbose=False)
    print(f"  Single condition GM: {gpu_result_single.GM.values}")
    
    print("\n2. Multi-condition (BROKEN):")
    conditions_multi = {v.T: [800, 900], v.P: 101325, v.X('TI'): 0.5}
    
    print("Running with verbose to see where it fails...")
    gpu_result_multi = equilibrium(dbf, ['NB', 'TI'], ['BCC_A2'], conditions_multi, gpu=True, verbose=True)

if __name__ == "__main__":
    main()