#!/usr/bin/env python
"""
Simple test to debug GPU vs CPU differences.
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
    
    # Test simple single point calculation
    conditions = {v.T: 1000, v.P: 101325, v.X('NI'): 0.5}
    
    print("Simple GPU vs CPU Test")
    print("="*50)
    
    # CPU calculation
    print("\nCPU calculation:")
    cpu_result = equilibrium(dbf, ['AL', 'NI', 'VA'], ['FCC_A1'], conditions, gpu=False)
    
    print(f"  GM: {cpu_result.GM.values}")
    print(f"  MU: {cpu_result.MU.values}")
    print(f"  NP: {cpu_result.NP.values}")
    print(f"  Phase: {cpu_result.Phase.values}")
    print(f"  X shape: {cpu_result.X.shape}")
    print(f"  Y shape: {cpu_result.Y.shape}")
    
    # GPU calculation
    print("\nGPU calculation:")
    gpu_result = equilibrium(dbf, ['AL', 'NI', 'VA'], ['FCC_A1'], conditions, gpu=True, verbose=False)
    
    print(f"  GM: {gpu_result.GM.values}")
    print(f"  MU: {gpu_result.MU.values}")
    print(f"  NP: {gpu_result.NP.values}")
    print(f"  Phase: {gpu_result.Phase.values}")
    print(f"  X shape: {gpu_result.X.shape}")
    print(f"  Y shape: {gpu_result.Y.shape}")
    
    # Detailed comparison
    print("\nDetailed Comparison:")
    print(f"  GM difference: {np.abs(cpu_result.GM.values - gpu_result.GM.values)}")
    print(f"  MU differences: {np.abs(cpu_result.MU.values - gpu_result.MU.values)}")
    
    # Check phase ordering
    print("\nPhase Array Analysis:")
    print(f"  CPU phases: {list(cpu_result.Phase.values.flatten())}")
    print(f"  GPU phases: {list(gpu_result.Phase.values.flatten())}")
    
    cpu_active = cpu_result.NP.values.flatten() > 1e-10
    gpu_active = gpu_result.NP.values.flatten() > 1e-10
    print(f"  CPU active phases: {np.where(cpu_active)[0]} -> {cpu_result.Phase.values.flatten()[cpu_active]}")
    print(f"  GPU active phases: {np.where(gpu_active)[0]} -> {gpu_result.Phase.values.flatten()[gpu_active]}")

if __name__ == "__main__":
    main()