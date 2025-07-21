#!/usr/bin/env python3
"""
Test final results after fixes
"""

import numpy as np
from pycalphad import Database, equilibrium
import warnings
warnings.filterwarnings('ignore')

# Load database
db = Database('NbTi.tdb')

# Test key conditions
print("GPU vs CPU Results After Fixes:")
print("="*60)

# X(TI)=0.1, T=600K (the main failing case)
cond = {'T': 600, 'P': 101325, 'X(TI)': 0.1}

try:
    gpu_eq = equilibrium(db, ['NB', 'TI'], 'BCC_A2', cond,
                       model=None, verbose=False,
                       calc_opts={'pdens': 50}, gpu=True)
    
    cpu_eq = equilibrium(db, ['NB', 'TI'], 'BCC_A2', cond,
                       model=None, verbose=False,
                       calc_opts={'pdens': 50}, gpu=False)
    
    gpu_x = float(gpu_eq.X('BCC_A2', 'TI').values[0])
    cpu_x = float(cpu_eq.X('BCC_A2', 'TI').values[0])
    
    gpu_gm = float(gpu_eq.GM.values[0])
    cpu_gm = float(cpu_eq.GM.values[0])
    
    print(f"Condition: X(TI)={cond['X(TI)']}, T={cond['T']}K")
    print(f"\nComposition X(TI):")
    print(f"  CPU result: {cpu_x:.6f}")
    print(f"  GPU result: {gpu_x:.6f}")
    print(f"  Difference: {abs(gpu_x - cpu_x):.6f} ({100*abs(gpu_x - cpu_x)/cpu_x:.3f}%)")
    
    print(f"\nGibbs energy (J/mol):")
    print(f"  CPU result: {cpu_gm:.1f}")
    print(f"  GPU result: {gpu_gm:.1f}")
    print(f"  Difference: {abs(gpu_gm - cpu_gm):.1f}")
    
    print(f"\nStatus: {'PASS' if abs(gpu_x - cpu_x) < 0.001 else 'FAIL'}")
    
except Exception as e:
    print(f"Error: {str(e)}")

print("\n" + "="*60)

# Test a few more conditions
test_conditions = [
    {'T': 600, 'P': 101325, 'X(TI)': 0.5},   # Two-phase region
    {'T': 600, 'P': 101325, 'X(TI)': 0.9},   # Single phase 
    {'T': 1000, 'P': 101325, 'X(TI)': 0.1},  # High temp
]

print("\nAdditional conditions:")
print(f"{'T (K)':>6} {'X(TI)':>6} {'CPU X':>10} {'GPU X':>10} {'Diff':>10} {'Status':>8}")
print("-"*50)

for cond in test_conditions:
    try:
        gpu_eq = equilibrium(db, ['NB', 'TI'], 'BCC_A2', cond,
                           model=None, verbose=False,
                           calc_opts={'pdens': 50}, gpu=True)
        
        cpu_eq = equilibrium(db, ['NB', 'TI'], 'BCC_A2', cond,
                           model=None, verbose=False,
                           calc_opts={'pdens': 50}, gpu=False)
        
        gpu_x = float(gpu_eq.X('BCC_A2', 'TI').values[0])
        cpu_x = float(cpu_eq.X('BCC_A2', 'TI').values[0])
        x_diff = abs(gpu_x - cpu_x)
        
        status = "PASS" if x_diff < 0.001 else "FAIL"
        
        print(f"{cond['T']:6.0f} {cond['X(TI)']:6.2f} {cpu_x:10.6f} {gpu_x:10.6f} {x_diff:10.6f} {status:>8}")
        
    except Exception as e:
        print(f"{cond['T']:6.0f} {cond['X(TI)']:6.2f} {'ERROR':>10}")