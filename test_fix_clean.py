#!/usr/bin/env python3
"""
Clean test to verify phase consolidation fix
"""

import numpy as np
from pycalphad import Database, equilibrium
import os

# Suppress debug output
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Load database
db = Database('NbTi.tdb')

# Test the specific failing condition
cond = {'T': 600, 'P': 101325, 'X(TI)': 0.1}

print("Testing X(TI)=0.1, T=600K (was showing GPU: 0.102653, CPU: 0.100000)")

# Run calculations quietly
gpu_eq = equilibrium(db, ['NB', 'TI'], 'BCC_A2', cond, 
                    model=None, verbose=False,
                    calc_opts={'pdens': 50}, gpu=True)

cpu_eq = equilibrium(db, ['NB', 'TI'], 'BCC_A2', cond,
                    model=None, verbose=False, 
                    calc_opts={'pdens': 50}, gpu=False)

# Extract values
gpu_x_ti = float(gpu_eq.X('BCC_A2', 'TI').values[0])
cpu_x_ti = float(cpu_eq.X('BCC_A2', 'TI').values[0])

print(f"\nResults:")
print(f"  CPU X(TI) = {cpu_x_ti:.6f}")
print(f"  GPU X(TI) = {gpu_x_ti:.6f}")
print(f"  Difference = {abs(gpu_x_ti - cpu_x_ti):.6f}")

if abs(gpu_x_ti - 0.102653) < 0.001:
    print("\n✗ GPU still shows old incorrect value (~0.102653)")
elif abs(gpu_x_ti - cpu_x_ti) < 0.001:
    print("\n✓ FIXED! GPU now matches CPU")
else:
    print(f"\n? GPU shows new value but still doesn't match CPU")

# Test more conditions
print("\n\nTesting additional conditions:")
print("-"*50)
print("Condition        CPU X(TI)    GPU X(TI)    Status")
print("-"*50)

test_conditions = [
    {'T': 600, 'P': 101325, 'X(TI)': 0.1},
    {'T': 600, 'P': 101325, 'X(TI)': 0.9},
    {'T': 1000, 'P': 101325, 'X(TI)': 0.1},
    {'T': 1000, 'P': 101325, 'X(TI)': 0.9},
]

for tc in test_conditions:
    gpu_eq = equilibrium(db, ['NB', 'TI'], 'BCC_A2', tc,
                        model=None, verbose=False,
                        calc_opts={'pdens': 50}, gpu=True)
    
    cpu_eq = equilibrium(db, ['NB', 'TI'], 'BCC_A2', tc,
                        model=None, verbose=False,
                        calc_opts={'pdens': 50}, gpu=False)
    
    gpu_x = float(gpu_eq.X('BCC_A2', 'TI').values[0])
    cpu_x = float(cpu_eq.X('BCC_A2', 'TI').values[0])
    
    status = "PASS" if abs(gpu_x - cpu_x) < 0.001 else "FAIL"
    
    print(f"T={tc['T']:4.0f}, X={tc['X(TI)']:.1f}    {cpu_x:.6f}     {gpu_x:.6f}     {status}")