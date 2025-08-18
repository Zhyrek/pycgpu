#!/usr/bin/env python
"""Debug GPU results to understand phase reporting issue."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def debug_gpu_results():
    """Debug GPU results in detail."""
    
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    
    phases = ['AL13FE4', 'AL2FE', 'AL5FE2', 'AL5FE4', 'ALCU_DELTA', 'ALCU_EPSILON', 
              'ALCU_ETA', 'ALCU_PRIME', 'ALCU_THETA', 'ALCU_ZETA', 'BCC_A2', 'BCC_B2', 
              'FCC_A1', 'GAMMA_D83', 'GAMMA_H', 'L12', 'LIQUID', 'TS01T1', 'TS01T2', 
              'TS01T3', 'TS01TI']
    
    conditions = {
        v.X('AL'): 0.40,
        v.X('CU'): 0.25,
        v.T: 700,
        v.P: 101325
    }
    
    print("=" * 80)
    print("GPU Results Debug")
    print("=" * 80)
    
    # Run GPU without pdens
    print("\n--- GPU Calculation (no pdens) ---")
    gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    
    print(f"GPU GM: {gpu_result.GM.values.item():.6f} J/mol")
    
    # Debug phase data structure
    print("\n--- Raw Phase Data ---")
    print(f"Phase array shape: {gpu_result.Phase.shape}")
    print(f"Phase array dtype: {gpu_result.Phase.dtype}")
    print(f"NP array shape: {gpu_result.NP.shape}")
    print(f"NP array dtype: {gpu_result.NP.dtype}")
    
    # Print raw arrays
    phase_array = gpu_result.Phase.values.flatten()
    np_array = gpu_result.NP.values.flatten()
    
    print(f"\nTotal slots in phase array: {len(phase_array)}")
    print(f"Total slots in NP array: {len(np_array)}")
    
    # Print ALL entries (including empty ones)
    print("\n--- All Phase Slots ---")
    for i, (phase, amount) in enumerate(zip(phase_array, np_array)):
        if not np.isnan(amount):
            print(f"  Slot {i:2}: phase='{phase}' (type={type(phase).__name__}), amount={amount:.10f}")
    
    # Check for non-empty phases
    non_empty_phases = [(i, p, a) for i, (p, a) in enumerate(zip(phase_array, np_array)) 
                        if p is not None and p != '' and not np.isnan(a)]
    
    print(f"\nNon-empty phase slots: {len(non_empty_phases)}")
    for i, phase, amount in non_empty_phases:
        print(f"  Slot {i}: {phase} = {amount:.10f}")
    
    # Check for non-zero amounts
    non_zero_amounts = [(i, p, a) for i, (p, a) in enumerate(zip(phase_array, np_array)) 
                        if not np.isnan(a) and abs(a) > 0]
    
    print(f"\nNon-zero amounts: {len(non_zero_amounts)}")
    for i, phase, amount in non_zero_amounts:
        print(f"  Slot {i}: {phase} = {amount:.10f}")
    
    # Check different thresholds
    print("\n--- Phase Detection with Different Thresholds ---")
    for threshold in [0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
        detected = [(p, a) for p, a in zip(phase_array, np_array) 
                    if p and p != '' and not np.isnan(a) and a > threshold]
        print(f"Threshold {threshold:1.0e}: {len(detected)} phases")
        if detected and threshold <= 1e-6:
            for p, a in detected:
                print(f"    {p}: {a:.10f}")
    
    # Check sum of phase amounts
    valid_amounts = np_array[~np.isnan(np_array)]
    total = np.sum(valid_amounts)
    print(f"\nSum of all phase amounts: {total:.10f}")
    
    # Run CPU for comparison
    print("\n--- CPU Calculation (no pdens) ---")
    cpu_result = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
    
    cpu_phase_array = cpu_result.Phase.values.flatten()
    cpu_np_array = cpu_result.NP.values.flatten()
    
    cpu_detected = [(p, a) for p, a in zip(cpu_phase_array, cpu_np_array) 
                    if p and p != '' and not np.isnan(a) and a > 1e-6]
    
    print(f"CPU detected {len(cpu_detected)} stable phases:")
    for p, a in cpu_detected:
        print(f"  {p}: {a:.10f}")
    
    # Direct comparison of first few slots
    print("\n--- Direct Slot Comparison (first 10 slots) ---")
    print("Slot | CPU Phase      | CPU Amount    | GPU Phase      | GPU Amount")
    print("-" * 70)
    for i in range(min(10, len(phase_array))):
        cpu_p = cpu_phase_array[i] if i < len(cpu_phase_array) else "N/A"
        cpu_a = cpu_np_array[i] if i < len(cpu_np_array) else np.nan
        gpu_p = phase_array[i]
        gpu_a = np_array[i]
        
        cpu_p_str = str(cpu_p) if cpu_p else "(empty)"
        gpu_p_str = str(gpu_p) if gpu_p else "(empty)"
        cpu_a_str = f"{cpu_a:.10f}" if not np.isnan(cpu_a) else "NaN"
        gpu_a_str = f"{gpu_a:.10f}" if not np.isnan(gpu_a) else "NaN"
        
        print(f"{i:4} | {cpu_p_str:14} | {cpu_a_str:13} | {gpu_p_str:14} | {gpu_a_str:13}")

if __name__ == "__main__":
    debug_gpu_results()