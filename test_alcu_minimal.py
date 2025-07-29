#!/usr/bin/env python
"""
Minimal test for ALCU_ZETA phase comparison
"""

import pycalphad as cp
import numpy as np

# Load database
db = cp.Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']

# Test with phases that are more likely to converge
phases = ['FCC_A1', 'ALCU_ZETA']

# Original problematic condition
conditions = {
    cp.v.X('AL'): 0.7,
    cp.v.X('CU'): 0.2,
    cp.v.T: 600,
    cp.v.P: 101325
}

print("Testing ALCU_ZETA phase CPU vs GPU")
print("="*50)
print(f"Phases: {phases}")
print(f"X(AL)={conditions[cp.v.X('AL')]}, X(CU)={conditions[cp.v.X('CU')]}, T={conditions[cp.v.T]}K")

# CPU
print("\nCPU calculation...")
try:
    eq_cpu = cp.equilibrium(db, components, phases, conditions, verbose=False)
    cpu_gm = float(eq_cpu.GM.values.flat[0])
    print(f"  GM = {cpu_gm:.3f} J/mol")
    
    # Show stable phases
    for i, phase in enumerate(eq_cpu.Phase.values.flat):
        if i < len(eq_cpu.NP.values.flat) and eq_cpu.NP.values.flat[i] > 1e-6:
            print(f"  {phase}: NP = {eq_cpu.NP.values.flat[i]:.6f}")
except Exception as e:
    print(f"  Error: {e}")
    cpu_gm = None

# GPU
print("\nGPU calculation...")
try:
    eq_gpu = cp.equilibrium(db, components, phases, conditions, gpu=True, verbose=False)
    gpu_gm = float(eq_gpu.GM.values.flat[0])
    print(f"  GM = {gpu_gm:.3f} J/mol")
    
    # Show stable phases
    for i, phase in enumerate(eq_gpu.Phase.values.flat):
        if i < len(eq_gpu.NP.values.flat) and eq_gpu.NP.values.flat[i] > 1e-6:
            print(f"  {phase}: NP = {eq_gpu.NP.values.flat[i]:.6f}")
            
    if cpu_gm is not None:
        error = abs(cpu_gm - gpu_gm)
        print(f"\nError: {error:.3f} J/mol")
        
        if error < 806:
            improvement = 806 / error
            print(f"Improvement from 806 J/mol: {improvement:.1f}x")
        
        if error < 1:
            print("✅ PASS")
        else:
            print("❌ FAIL")
            
except Exception as e:
    print(f"  Error: {e}")