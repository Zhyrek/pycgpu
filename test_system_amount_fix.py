#!/usr/bin/env python3
"""
Test the GPU system amount constraint fix
"""

import numpy as np
from pycalphad import Database, equilibrium
import warnings
warnings.filterwarnings('ignore')

# Load database
db = Database('NbTi.tdb')

# Test the failing condition: X(TI)=0.1, T=600K
# This is in the single-phase region above the miscibility gap
cond = {'T': 600, 'P': 101325, 'X(TI)': 0.1}

print("Testing GPU system amount constraint fix")
print("Condition: X(TI)=0.1, T=600K (single-phase region)")
print("="*60)

try:
    # Run GPU calculation
    gpu_eq = equilibrium(db, ['NB', 'TI'], 'BCC_A2', cond,
                       model=None, verbose=False,
                       calc_opts={'pdens': 50}, gpu=True)
    
    # Run CPU calculation  
    cpu_eq = equilibrium(db, ['NB', 'TI'], 'BCC_A2', cond,
                       model=None, verbose=False,
                       calc_opts={'pdens': 50}, gpu=False)
    
    # Extract results
    gpu_x = float(gpu_eq.isel(vertex=0).X_BCC_A2_TI.values)
    cpu_x = float(cpu_eq.isel(vertex=0).X_BCC_A2_TI.values)
    
    gpu_gm = float(gpu_eq.GM.values[0])
    cpu_gm = float(cpu_eq.GM.values[0])
    
    print(f"\nComposition X(TI):")
    print(f"  CPU: {cpu_x:.6f}")
    print(f"  GPU: {gpu_x:.6f}")
    print(f"  Difference: {abs(gpu_x - cpu_x):.6f} ({100*abs(gpu_x - cpu_x)/cpu_x:.3f}%)")
    
    print(f"\nGibbs energy (J/mol):")
    print(f"  CPU: {cpu_gm:.1f}")
    print(f"  GPU: {gpu_gm:.1f}")
    print(f"  Difference: {abs(gpu_gm - cpu_gm):.1f}")
    
    print(f"\nPrevious error: 8.672 J/mol (FAIL)")
    print(f"Current status: {'PASS' if abs(gpu_gm - cpu_gm) < 1.0 else 'FAIL'}")
    
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("The fix addresses the issue where GPU wrote system amount constraint")
print("inside phase loops, causing different matrix structure when phases")
print("were consolidated from 2 to 1.")