#!/usr/bin/env python
"""Test to examine GPU phase results in detail."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def test_gpu_phase_results():
    """Examine GPU phase results in detail."""
    
    # Load the Al-Cu-Fe database
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    
    # All 21 phases
    phases = ['AL13FE4', 'AL2FE', 'AL5FE2', 'AL5FE4', 'ALCU_DELTA', 'ALCU_EPSILON', 
              'ALCU_ETA', 'ALCU_PRIME', 'ALCU_THETA', 'ALCU_ZETA', 'BCC_A2', 'BCC_B2', 
              'FCC_A1', 'GAMMA_D83', 'GAMMA_H', 'L12', 'LIQUID', 'TS01T1', 'TS01T2', 
              'TS01T3', 'TS01TI']
    
    # Test condition
    conditions = {
        v.X('AL'): 0.40,
        v.X('CU'): 0.25,
        v.T: 700,
        v.P: 101325
    }
    
    print("=" * 80)
    print("Examining GPU phase results in detail")
    print("=" * 80)
    print(f"\nCondition: X(AL)=0.40, X(CU)=0.25, X(FE)=0.35, T=700K")
    
    # Run GPU calculation
    gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    
    print("\n--- GPU Result Structure ---")
    print(f"GM shape: {gpu_result.GM.shape}")
    print(f"GM value: {gpu_result.GM.values.item():.6f} J/mol")
    print(f"NP shape: {gpu_result.NP.shape}")
    print(f"Phase shape: {gpu_result.Phase.shape}")
    print(f"X shape: {gpu_result.X.shape}")
    print(f"MU shape: {gpu_result.MU.shape}")
    
    # Extract all phase amounts
    print("\n--- All Phase Amounts (including zeros) ---")
    np_values = gpu_result.NP.values.flatten()
    phase_values = gpu_result.Phase.values.flatten()
    
    for i, (phase, amount) in enumerate(zip(phase_values, np_values)):
        if phase and phase != '':  # Skip empty phase slots
            print(f"  Slot {i}: {phase:15s} = {amount:.10f}")
    
    # Check for any non-zero amounts
    non_zero_mask = np_values > 0
    if np.any(non_zero_mask):
        print(f"\nFound {np.sum(non_zero_mask)} non-zero phase amounts")
        non_zero_phases = phase_values[non_zero_mask]
        non_zero_amounts = np_values[non_zero_mask]
        for phase, amount in zip(non_zero_phases, non_zero_amounts):
            print(f"  {phase}: {amount:.10f}")
    else:
        print("\n✗ WARNING: No non-zero phase amounts found!")
    
    # Check chemical potentials
    print("\n--- Chemical Potentials ---")
    mu_values = gpu_result.MU.values.flatten()
    for comp, mu in zip(['AL', 'CU', 'FE'], mu_values):
        print(f"  MU({comp}) = {mu:.2f} J/mol")
    
    # Check compositions
    print("\n--- Phase Compositions ---")
    x_values = gpu_result.X.values
    for i, phase in enumerate(phase_values):
        if phase and phase != '':
            phase_comps = x_values.reshape(-1, 3)[i]  # 3 components (AL, CU, FE)
            if not np.all(np.isnan(phase_comps)):
                print(f"  {phase}: AL={phase_comps[0]:.4f}, CU={phase_comps[1]:.4f}, FE={phase_comps[2]:.4f}")
    
    # Run CPU calculation for comparison
    cpu_result = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
    
    print("\n--- CPU Result for Comparison ---")
    print(f"GM value: {cpu_result.GM.values.item():.6f} J/mol")
    
    # Extract CPU phase amounts
    cpu_np_values = cpu_result.NP.values.flatten()
    cpu_phase_values = cpu_result.Phase.values.flatten()
    
    print("\nCPU stable phases:")
    for phase, amount in zip(cpu_phase_values, cpu_np_values):
        if phase and phase != '' and amount > 1e-6:
            print(f"  {phase:15s} = {amount:.10f}")
    
    # Compare totals
    print("\n--- Comparison ---")
    gpu_total = np.sum(np_values[~np.isnan(np_values)])
    cpu_total = np.sum(cpu_np_values[~np.isnan(cpu_np_values)])
    print(f"GPU total phase amount: {gpu_total:.10f}")
    print(f"CPU total phase amount: {cpu_total:.10f}")
    
    if abs(gpu_total - 1.0) > 1e-6:
        print(f"✗ GPU total phase amount is not 1.0!")
    if abs(cpu_total - 1.0) > 1e-6:
        print(f"✗ CPU total phase amount is not 1.0!")
    
    # Check if the issue is with phase amount threshold
    print("\n--- Checking Different Thresholds ---")
    for threshold in [1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
        gpu_stable = [(p, a) for p, a in zip(phase_values, np_values) if p and a > threshold]
        print(f"Threshold {threshold:1.0e}: {len(gpu_stable)} stable phases")
        if gpu_stable:
            for p, a in gpu_stable:
                print(f"    {p}: {a:.10f}")

if __name__ == "__main__":
    test_gpu_phase_results()