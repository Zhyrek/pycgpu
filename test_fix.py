#!/usr/bin/env python
"""Test the GPU mole fraction constraint RHS fix."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test the problematic condition
conditions = {v.X('TI'): 0.5, v.T: 700, v.P: 101325}

print("Testing GPU fix for X(TI)=0.5, T=700K")
print("=" * 50)

try:
    # CPU calculation
    print("Running CPU calculation...")
    result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
    cpu_gm = result_cpu.GM.values.flatten()[0]
    cpu_phase_fractions = result_cpu.NP.values.flatten()
    
    print(f"CPU Results:")
    print(f"  GM: {cpu_gm:.6f} J/mol")
    print(f"  Phase fractions: {cpu_phase_fractions}")
    print(f"  Active phases: {sum(1 for p in cpu_phase_fractions if p > 1e-12)}")
    
    # GPU calculation
    print("\nRunning GPU calculation...")
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    gpu_gm = result_gpu.GM.values.flatten()[0]
    gpu_phase_fractions = result_gpu.NP.values.flatten()
    
    print(f"GPU Results:")
    print(f"  GM: {gpu_gm:.6f} J/mol")
    print(f"  Phase fractions: {gpu_phase_fractions}")
    print(f"  Active phases: {sum(1 for p in gpu_phase_fractions if p > 1e-12)}")
    
    # Compare
    error = abs(cpu_gm - gpu_gm)
    print(f"\nComparison:")
    print(f"  Absolute error: {error:.6f} J/mol")
    print(f"  Relative error: {error/abs(cpu_gm)*100:.4f}%")
    
    # Check if phases match
    cpu_active = sum(1 for p in cpu_phase_fractions if p > 1e-12)
    gpu_active = sum(1 for p in gpu_phase_fractions if p > 1e-12)
    
    if cpu_active == gpu_active and error < 10.0:
        print(f"  ✅ SUCCESS: Same number of phases and error < 10 J/mol")
    elif cpu_active == gpu_active:
        print(f"  ⚠️  PARTIAL: Same phases but error = {error:.1f} J/mol")
    else:
        print(f"  ❌ FAIL: Different phase configurations (CPU: {cpu_active}, GPU: {gpu_active})")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("=" * 50)