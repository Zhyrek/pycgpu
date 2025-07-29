#!/usr/bin/env python
"""
Direct test of ALCU_ZETA energy with multiple phases
"""

import pycalphad as cp
import numpy as np

# Load database
db = cp.Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']

# Test with multiple phases including ALCU_ZETA
phases = ['FCC_A1', 'ALCU_ZETA', 'BCC_B2']

# Test conditions - same as original
conditions = {
    cp.v.X('AL'): 0.7,
    cp.v.X('CU'): 0.2,
    cp.v.T: 600,
    cp.v.P: 101325
}

print("Testing ALCU_ZETA with multiple phases")
print("=" * 60)
print(f"Phases: {phases}")
print(f"Conditions: X(AL)={conditions[cp.v.X('AL')]}, X(CU)={conditions[cp.v.X('CU')]}, T={conditions[cp.v.T]}K")
print(f"X(FE) = {1 - conditions[cp.v.X('AL')] - conditions[cp.v.X('CU')]}")

# CPU calculation
print("\nCPU calculation...")
try:
    eq_cpu = cp.equilibrium(db, components, phases, conditions, verbose=False)
    cpu_gm = float(eq_cpu.GM.values.flat[0])
    print(f"  GM = {cpu_gm:.3f} J/mol")
    
    # Check which phases are stable
    cpu_phases = {}
    for i, phase in enumerate(eq_cpu.Phase.values.flat):
        if i < len(eq_cpu.NP.values.flat) and eq_cpu.NP.values.flat[i] > 1e-6:
            cpu_phases[phase] = eq_cpu.NP.values.flat[i]
            print(f"  {phase}: NP = {eq_cpu.NP.values.flat[i]:.6f}")
            
    # Check ALCU_ZETA site fractions if stable
    if 'ALCU_ZETA' in cpu_phases:
        print("\n  ALCU_ZETA site fractions (CPU):")
        # Find the index of ALCU_ZETA in the phase list
        zeta_idx = list(eq_cpu.Phase.values.flat).index('ALCU_ZETA')
        if hasattr(eq_cpu, 'Y'):
            Y_values = eq_cpu.Y.sel(vertex=zeta_idx).values
            if Y_values.size >= 3:
                print(f"    Y(ALCU_ZETA,0,AL) = {Y_values.flat[0]:.6f}")
                print(f"    Y(ALCU_ZETA,1,CU) = {Y_values.flat[1]:.6f}")  
                print(f"    Y(ALCU_ZETA,1,FE) = {Y_values.flat[2]:.6f}")
                
except Exception as e:
    print(f"  Error: {e}")
    cpu_gm = None

# GPU calculation
print("\nGPU calculation...")
try:
    eq_gpu = cp.equilibrium(db, components, phases, conditions, gpu=True, verbose=False)
    gpu_gm = float(eq_gpu.GM.values.flat[0])
    print(f"  GM = {gpu_gm:.3f} J/mol")
    
    # Check which phases are stable
    gpu_phases = {}
    for i, phase in enumerate(eq_gpu.Phase.values.flat):
        if i < len(eq_gpu.NP.values.flat) and eq_gpu.NP.values.flat[i] > 1e-6:
            gpu_phases[phase] = eq_gpu.NP.values.flat[i]
            print(f"  {phase}: NP = {eq_gpu.NP.values.flat[i]:.6f}")
            
    # Check ALCU_ZETA site fractions if stable
    if 'ALCU_ZETA' in gpu_phases:
        print("\n  ALCU_ZETA site fractions (GPU):")
        # Find the index of ALCU_ZETA in the phase list
        zeta_idx = list(eq_gpu.Phase.values.flat).index('ALCU_ZETA')
        if hasattr(eq_gpu, 'Y'):
            Y_values = eq_gpu.Y.sel(vertex=zeta_idx).values
            if Y_values.size >= 3:
                print(f"    Y(ALCU_ZETA,0,AL) = {Y_values.flat[0]:.6f}")
                print(f"    Y(ALCU_ZETA,1,CU) = {Y_values.flat[1]:.6f}")
                print(f"    Y(ALCU_ZETA,1,FE) = {Y_values.flat[2]:.6f}")
                
    if cpu_gm is not None:
        error = abs(cpu_gm - gpu_gm)
        print(f"\nError: {error:.3f} J/mol")
        
        # Check if both have ALCU_ZETA stable
        if 'ALCU_ZETA' in cpu_phases and 'ALCU_ZETA' in gpu_phases:
            print(f"\nALCU_ZETA is stable in both calculations")
            print(f"CPU NP = {cpu_phases['ALCU_ZETA']:.6f}")
            print(f"GPU NP = {gpu_phases['ALCU_ZETA']:.6f}")
            print(f"NP difference = {abs(cpu_phases['ALCU_ZETA'] - gpu_phases['ALCU_ZETA']):.6f}")
        elif 'ALCU_ZETA' in cpu_phases:
            print(f"\nALCU_ZETA is stable only in CPU calculation")
        elif 'ALCU_ZETA' in gpu_phases:
            print(f"\nALCU_ZETA is stable only in GPU calculation")
        else:
            print(f"\nALCU_ZETA is not stable in either calculation")
            
except Exception as e:
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()

# Now let's try a single-phase calculation for ALCU_ZETA to see the energy directly
print("\n" + "="*60)
print("Single-phase ALCU_ZETA test")
print("="*60)

# Force ALCU_ZETA to be stable at its ideal composition
alcu_conditions = {
    cp.v.X('AL'): 0.45,  # 9/20
    cp.v.X('CU'): 0.55,  # 11/20  
    cp.v.T: 600,
    cp.v.P: 101325
}

print(f"\nConditions: X(AL)={alcu_conditions[cp.v.X('AL')]}, X(CU)={alcu_conditions[cp.v.X('CU')]}")

# CPU single phase
print("\nCPU single-phase ALCU_ZETA:")
try:
    eq_cpu_single = cp.equilibrium(db, components, ['ALCU_ZETA'], alcu_conditions, verbose=False)
    cpu_single_gm = float(eq_cpu_single.GM.values.flat[0])
    print(f"  GM = {cpu_single_gm:.3f} J/mol")
except Exception as e:
    print(f"  Error: {e}")
    cpu_single_gm = None

# GPU single phase  
print("\nGPU single-phase ALCU_ZETA:")
try:
    eq_gpu_single = cp.equilibrium(db, components, ['ALCU_ZETA'], alcu_conditions, gpu=True, verbose=False)
    gpu_single_gm = float(eq_gpu_single.GM.values.flat[0])
    print(f"  GM = {gpu_single_gm:.3f} J/mol")
    
    if cpu_single_gm is not None:
        single_error = abs(cpu_single_gm - gpu_single_gm)
        print(f"\nSingle-phase error: {single_error:.3f} J/mol")
except Exception as e:
    print(f"  Error: {e}")
    gpu_single_gm = None