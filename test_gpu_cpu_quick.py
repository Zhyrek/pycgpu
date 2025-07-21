#!/usr/bin/env python3
"""
Quick test of GPU vs CPU after fixes
"""

import numpy as np
from pycalphad import Database, equilibrium
import warnings
warnings.filterwarnings('ignore')

# Load database
db = Database('NbTi.tdb')

# Test key failing conditions
test_conditions = [
    {'T': 600, 'P': 101325, 'X(TI)': 0.1},   # Was failing with 0.102653 vs 0.100000
    {'T': 600, 'P': 101325, 'X(TI)': 0.5},   # Two-phase region
    {'T': 600, 'P': 101325, 'X(TI)': 0.9},   # Single phase 
    {'T': 1000, 'P': 101325, 'X(TI)': 0.1},  # High temp single phase
    {'T': 300, 'P': 101325, 'X(TI)': 0.1},   # Low temp
]

print("Testing GPU vs CPU after applying fixes:")
print("1. Phase consolidation when all phases would be removed")
print("2. Workspace DOF passed to formulamole_grad")  
print("3. System amount constraint only includes active phases")
print("="*70)
print(f"{'T (K)':>6} {'X(TI)':>6} {'CPU GM':>10} {'GPU GM':>10} {'ΔGM':>8} {'CPU X':>8} {'GPU X':>8} {'Status':>8}")
print("-"*70)

passed = 0
for cond in test_conditions:
    try:
        # Run calculations with verbose=False to suppress debug output
        gpu_eq = equilibrium(db, ['NB', 'TI'], 'BCC_A2', cond,
                           model=None, verbose=False,
                           calc_opts={'pdens': 50}, gpu=True)
        
        cpu_eq = equilibrium(db, ['NB', 'TI'], 'BCC_A2', cond,
                           model=None, verbose=False,
                           calc_opts={'pdens': 50}, gpu=False)
        
        # Extract values
        gpu_gm = float(gpu_eq.GM.values[0])
        cpu_gm = float(cpu_eq.GM.values[0])
        gm_diff = abs(gpu_gm - cpu_gm)
        
        gpu_x = float(gpu_eq.X('BCC_A2', 'TI').values[0])
        cpu_x = float(cpu_eq.X('BCC_A2', 'TI').values[0])
        x_diff = abs(gpu_x - cpu_x)
        
        if gm_diff < 1.0 and x_diff < 0.001:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            
        print(f"{cond['T']:6.0f} {cond['X(TI)']:6.2f} {cpu_gm:10.1f} {gpu_gm:10.1f} {gm_diff:8.1f} {cpu_x:8.4f} {gpu_x:8.4f} {status:>8}")
        
    except Exception as e:
        print(f"{cond['T']:6.0f} {cond['X(TI)']:6.2f} {'ERROR':>10} {'ERROR':>10} {'ERROR':>8} {'ERROR':>8} {'ERROR':>8} {'ERROR':>8}")
        print(f"       Error: {str(e)}")

print("-"*70)
print(f"\nSummary: {passed}/{len(test_conditions)} passed ({100*passed/len(test_conditions):.0f}%)")

# Special check for the main failing case
print(f"\nSpecial case X(TI)=0.1, T=600K:")
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
    
    print(f"  Previous GPU result: X(TI) = 0.102653")
    print(f"  Current GPU result:  X(TI) = {gpu_x:.6f}")
    print(f"  CPU result:          X(TI) = {cpu_x:.6f}")
    print(f"  Difference: {abs(gpu_x - cpu_x):.6f}")
    
    if abs(gpu_x - cpu_x) < 0.001:
        print("  FIXED!")
    else:
        print("  Still failing")
        
except Exception as e:
    print(f"  Error in special case: {str(e)}")