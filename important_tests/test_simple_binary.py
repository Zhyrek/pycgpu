#!/usr/bin/env python
"""Test simple binary system to isolate constraint issue."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def main():
    # Create a simple binary database
    dbf_content = """
ELEMENT AL   FCC_A1    26.98154  4540.0  28.30 !
ELEMENT CU   FCC_A1    63.546    5004.0  33.15 !
ELEMENT VA   VACUUM    0.0       0.0     0.0 !

PHASE FCC_A1 % 1 1.0 !
CONSTITUENT FCC_A1 : AL,CU : !

PHASE LIQUID % 1 1.0 !
CONSTITUENT LIQUID : AL,CU : !

PARAMETER G(FCC_A1,AL;0) 298.15 -7976.15+137.093*T-24.3672*T*LN(T)
    -0.001884662*T**2-8.77664E-07*T**3+74092*T**(-1); 6000 N !
PARAMETER G(FCC_A1,CU;0) 298.15 -7770.458+130.485*T-24.112*T*LN(T)
    -0.00265684*T**2+1.29223E-07*T**3+52478*T**(-1); 6000 N !
PARAMETER G(FCC_A1,AL,CU;0) 298.15 -53520+2*T; 6000 N !

PARAMETER G(LIQUID,AL;0) 298.15 +11005.029-11.841867*T
    +7.934E-20*T**7; 6000 N !
PARAMETER G(LIQUID,CU;0) 298.15 +12964.735-9.511904*T
    -5.849E-21*T**7; 6000 N !
PARAMETER G(LIQUID,AL,CU;0) 298.15 -66622+8.1*T; 6000 N !
"""
    
    # Write to temp file
    with open('simple_alcu.tdb', 'w') as f:
        f.write(dbf_content)
    
    dbf = Database('simple_alcu.tdb')
    comps = ['AL', 'CU', 'VA']
    phases = ['FCC_A1', 'LIQUID']
    
    conditions = {
        v.X('AL'): 0.70,
        v.T: 800,
        v.P: 101325
    }
    
    print("=" * 80)
    print("SIMPLE BINARY TEST (AL-CU)")
    print("=" * 80)
    print("\nTarget: X(AL) = 0.70")
    
    # CPU
    print("\nCPU Result:")
    cpu_result = equilibrium(dbf, comps, phases, conditions, gpu=False)
    cpu_phases = cpu_result.Phase.values.flatten()
    cpu_np = cpu_result.NP.values.flatten()
    x_vals = cpu_result.X.values.reshape(-1, 2)  # AL, CU
    
    cpu_al = 0
    for i, amount in enumerate(cpu_np):
        if not np.isnan(amount) and amount > 0.001:
            phase = cpu_phases[i]
            print(f"  {phase}: {amount:.4f}, X(AL)={x_vals[i][0]:.4f}")
            cpu_al += amount * x_vals[i][0]
    print(f"  Bulk X(AL) = {cpu_al:.4f}")
    
    # GPU
    print("\nGPU Result:")
    gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True)
    gpu_phases = gpu_result.Phase.values.flatten()
    gpu_np = gpu_result.NP.values.flatten()
    x_vals = gpu_result.X.values.reshape(-1, 2)  # AL, CU
    
    gpu_al = 0
    for i, amount in enumerate(gpu_np):
        if not np.isnan(amount) and amount > 0.001:
            phase = gpu_phases[i]
            print(f"  {phase}: {amount:.4f}, X(AL)={x_vals[i][0]:.4f}")
            gpu_al += amount * x_vals[i][0]
    print(f"  Bulk X(AL) = {gpu_al:.4f}")
    
    # Comparison
    print("\n" + "=" * 80)
    if abs(cpu_al - 0.70) < 0.001:
        print("✓ CPU respects constraint")
    else:
        print("✗ CPU violates constraint")
    
    if abs(gpu_al - 0.70) < 0.001:
        print("✓ GPU respects constraint")
    else:
        print("✗ GPU violates constraint")
    
    # Clean up
    os.remove('simple_alcu.tdb')

if __name__ == "__main__":
    main()