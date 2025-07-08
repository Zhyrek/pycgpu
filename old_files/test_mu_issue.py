#!/usr/bin/env python
"""
Test to debug missing MU values in multi-condition GPU calculations.
"""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings('ignore')

def main():
    # Create simple Al-Ni database
    dbf = Database("""
    ELEMENT AL FCC_A1 26.98 69.95 28.30 !
    ELEMENT NI FCC_A1 58.69 67.40 29.87 !
    ELEMENT VA VACUUM 0.00 0.00 0.00 !
    
    PHASE FCC_A1 %  2 1 1 !
    CONSTITUENT FCC_A1 : AL,NI : VA : !
    
    PARAMETER G(FCC_A1,AL:VA;0) 298.15 -7976.15+137.0715*T-24.36720*T*LN(T)
        -0.001884662*T**2-8.77664E-07*T**3+74092*T**(-1); 700 Y
        -11276.24+223.0481*T-38.58443*T*LN(T)+0.018531982*T**2
        -5.764227E-06*T**3+74092*T**(-1); 933.6 Y
        -11277.68+188.6620*T-31.74819*T*LN(T)-1230.622E25*T**(-9); 2900 N !
    
    PARAMETER G(FCC_A1,NI:VA;0) 298.15 -5179.159+117.8540*T-22.09600*T*LN(T)
        -0.0048407*T**2; 1728 Y
        -27840.62+279.1350*T-43.10*T*LN(T)+1127.54E28*T**(-9); 3000 N !
    
    PARAMETER G(FCC_A1,AL,NI:VA;0) 298.15 -162407.75+16.212965*T; 6000 N !
    PARAMETER G(FCC_A1,AL,NI:VA;1) 298.15 +73417.798-34.914168*T; 6000 N !
    PARAMETER G(FCC_A1,AL,NI:VA;2) 298.15 +33471.014-9.8373558*T; 6000 N !
    """)
    
    print("MU Calculation Debug Test")
    print("="*50)
    
    # Test 1: Single condition (works)
    print("\nTest 1: Single condition")
    conditions = {v.T: 1000, v.P: 101325, v.X('NI'): 0.5}
    
    cpu_result = equilibrium(dbf, ['AL', 'NI', 'VA'], ['FCC_A1'], conditions, gpu=False)
    gpu_result = equilibrium(dbf, ['AL', 'NI', 'VA'], ['FCC_A1'], conditions, gpu=True, verbose=True)
    
    print(f"\nCPU MU: {cpu_result.MU.values.flatten()}")
    print(f"GPU MU: {gpu_result.MU.values.flatten()}")
    
    # Test 2: Multiple temperatures (MU fails)
    print("\n\nTest 2: Multiple temperatures")
    conditions = {v.T: [800, 1000, 1200], v.P: 101325, v.X('NI'): 0.5}
    
    cpu_result = equilibrium(dbf, ['AL', 'NI', 'VA'], ['FCC_A1'], conditions, gpu=False)
    gpu_result = equilibrium(dbf, ['AL', 'NI', 'VA'], ['FCC_A1'], conditions, gpu=True, verbose=False)
    
    print(f"\nCPU MU shape: {cpu_result.MU.shape}")
    print(f"GPU MU shape: {gpu_result.MU.shape}")
    
    print(f"\nCPU MU values:")
    for i in range(3):
        print(f"  T={800+i*200}K: {cpu_result.MU.values[0,0,i,0,:]}")
    
    print(f"\nGPU MU values:")
    for i in range(3):
        print(f"  T={800+i*200}K: {gpu_result.MU.values[0,0,i,0,:]}")
        
    # Check if GPU is extracting MU correctly
    print("\n\nDetailed GPU MU extraction check:")
    print(f"Full GPU MU array:\n{gpu_result.MU.values}")

if __name__ == "__main__":
    main()