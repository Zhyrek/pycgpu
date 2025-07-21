#!/usr/bin/env python3
"""
Verify that the phase consolidation fix is working
"""

import numpy as np
from pycalphad import Database, equilibrium

# Load database
db = Database('NbTi.tdb')

# Test a single condition that was failing before
# X(TI)=0.1, T=600K was showing GPU: 0.102653, CPU: 0.100000
cond = {'T': 600, 'P': 101325, 'X(TI)': 0.1}

print("Testing condition that was failing: X(TI)=0.1, T=600K")
print("="*60)

try:
    # GPU calculation
    print("\nRunning GPU calculation...")
    gpu_eq = equilibrium(db, ['NB', 'TI'], 'BCC_A2',
                        cond, model=None, verbose=False,
                        calc_opts={'pdens': 50}, gpu=True)
    
    # CPU calculation
    print("\nRunning CPU calculation...")
    cpu_eq = equilibrium(db, ['NB', 'TI'], 'BCC_A2',
                        cond, model=None, verbose=False,
                        calc_opts={'pdens': 50}, gpu=False)
    
    # Extract values
    gpu_gm = float(gpu_eq.GM.values[0])
    cpu_gm = float(cpu_eq.GM.values[0])
    
    gpu_x_ti = float(gpu_eq.X('BCC_A2', 'TI').values[0])
    cpu_x_ti = float(cpu_eq.X('BCC_A2', 'TI').values[0])
    
    # Number of phases
    gpu_phases = list(gpu_eq.Phase.unique())
    cpu_phases = list(cpu_eq.Phase.unique())
    
    print("\nResults:")
    print(f"  CPU: GM = {cpu_gm:.2f} J/mol, X(TI) = {cpu_x_ti:.6f}")
    print(f"  GPU: GM = {gpu_gm:.2f} J/mol, X(TI) = {gpu_x_ti:.6f}")
    print(f"  Energy difference: {abs(gpu_gm - cpu_gm):.2f} J/mol")
    print(f"  Composition difference: {abs(gpu_x_ti - cpu_x_ti):.6f}")
    print(f"  CPU phases: {cpu_phases}")
    print(f"  GPU phases: {gpu_phases}")
    
    # Check if fixed
    if abs(gpu_x_ti - cpu_x_ti) < 0.001:
        print("\n✓ FIXED! GPU and CPU now agree on composition.")
    else:
        print("\n✗ NOT FIXED. GPU still deviates from CPU.")
        
except Exception as e:
    print(f"\nError: {str(e)}")
    import traceback
    traceback.print_exc()