#!/usr/bin/env python
"""
Debug script to check what data is being transferred to the GPU kernel.
"""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings('ignore')

def main():
    print("GPU Data Transfer Debug")
    print("="*50)
    
    # Use the NbTi database for debugging
    try:
        dbf = Database("NbTi.tdb") 
        
        # Simple test case
        conditions = {v.T: 800, v.P: 101325, v.X('TI'): 0.5}
        
        print("\nRunning GPU equilibrium with verbose output...")
        gpu_result = equilibrium(dbf, ['NB', 'TI'], ['BCC_A2'], conditions, gpu=True, verbose=True)
        
        print(f"\nGPU result:")
        print(f"  GM: {gpu_result.GM.values}")
        print(f"  MU: {gpu_result.MU.values}")
        
        # Compare with CPU
        print(f"\nComparing with CPU result:")
        cpu_result = equilibrium(dbf, ['NB', 'TI'], ['BCC_A2'], conditions, gpu=False, verbose=False)
        print(f"  CPU GM: {cpu_result.GM.values}")
        print(f"  CPU MU: {cpu_result.MU.values}")
        
        print(f"\nDifferences:")
        print(f"  GM diff: {np.abs(gpu_result.GM.values - cpu_result.GM.values)}")
        print(f"  MU diff: {np.abs(gpu_result.MU.values - cpu_result.MU.values)}")
        
    except FileNotFoundError:
        print("NbTi.tdb not found")
        
        # Fall back to simple Al-Ni test
        dbf = Database("""
        ELEMENT AL FCC_A1 26.98 69.95 28.30 !
        ELEMENT NI FCC_A1 58.69 67.40 29.87 !
        ELEMENT VA VACUUM 0.00 0.00 0.00 !
        PHASE FCC_A1 %  2 1 1 !
        CONSTITUENT FCC_A1 : AL,NI : VA : !
        PARAMETER G(FCC_A1,AL:VA;0) 298.15 -7976.15+137.0715*T-24.36720*T*LN(T); 700 Y !
        PARAMETER G(FCC_A1,NI:VA;0) 298.15 -5179.159+117.8540*T-22.09600*T*LN(T); 1728 Y !
        """)
        
        conditions = {v.T: 1000, v.P: 101325, v.X('NI'): 0.5}
        
        print("\nRunning Al-Ni GPU equilibrium with verbose output...")
        gpu_result = equilibrium(dbf, ['AL', 'NI', 'VA'], ['FCC_A1'], conditions, gpu=True, verbose=True)

if __name__ == "__main__":
    main()