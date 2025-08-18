#!/usr/bin/env python
"""Test with fixed phase extraction logic."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def extract_stable_phases(result, threshold=1e-6):
    """Extract stable phases from equilibrium result properly."""
    phase_array = result.Phase.values.flatten()
    np_array = result.NP.values.flatten()
    
    stable_phases = []
    for phase, amount in zip(phase_array, np_array):
        if phase and phase != '' and not np.isnan(amount) and amount > threshold:
            stable_phases.append((phase, amount))
    
    return stable_phases

def test_with_fixed_extraction():
    """Test with properly fixed phase extraction."""
    
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
    print("Testing with Fixed Phase Extraction")
    print("=" * 80)
    print(f"\nCondition: X(AL)=0.40, X(CU)=0.25, X(FE)=0.35, T=700K")
    
    # CPU calculation
    print("\n--- CPU Calculation ---")
    cpu_result = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
    cpu_gm = cpu_result.GM.values.item()
    cpu_phases = extract_stable_phases(cpu_result)
    
    print(f"CPU GM: {cpu_gm:.6f} J/mol")
    print(f"CPU Stable phases: {len(cpu_phases)}")
    for phase, amount in cpu_phases:
        print(f"  {phase}: {amount:.6f}")
    
    # GPU calculation
    print("\n--- GPU Calculation ---")
    gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    gpu_gm = gpu_result.GM.values.item()
    gpu_phases = extract_stable_phases(gpu_result)
    
    print(f"GPU GM: {gpu_gm:.6f} J/mol")
    print(f"GPU Stable phases: {len(gpu_phases)}")
    for phase, amount in gpu_phases:
        print(f"  {phase}: {amount:.6f}")
    
    # Compare results
    print("\n--- Comparison ---")
    gm_diff = abs(gpu_gm - cpu_gm)
    print(f"GM difference: {gm_diff:.6f} J/mol")
    
    cpu_phase_names = set(p for p, _ in cpu_phases)
    gpu_phase_names = set(p for p, _ in gpu_phases)
    
    missing_in_gpu = cpu_phase_names - gpu_phase_names
    extra_in_gpu = gpu_phase_names - cpu_phase_names
    common_phases = cpu_phase_names & gpu_phase_names
    
    if missing_in_gpu:
        print(f"✗ Phases missing in GPU: {', '.join(missing_in_gpu)}")
    if extra_in_gpu:
        print(f"✗ Extra phases in GPU: {', '.join(extra_in_gpu)}")
    if common_phases:
        print(f"✓ Common phases: {', '.join(common_phases)}")
    
    # Compare phase amounts for common phases
    if common_phases:
        print("\n--- Phase Amount Comparison ---")
        cpu_dict = {p: a for p, a in cpu_phases}
        gpu_dict = {p: a for p, a in gpu_phases}
        
        for phase in sorted(common_phases):
            cpu_amount = cpu_dict[phase]
            gpu_amount = gpu_dict[phase]
            diff = abs(gpu_amount - cpu_amount)
            print(f"{phase}:")
            print(f"  CPU: {cpu_amount:.6f}")
            print(f"  GPU: {gpu_amount:.6f}")
            print(f"  Diff: {diff:.6f}")
    
    # Final verdict
    print("\n--- Final Analysis ---")
    if not missing_in_gpu and not extra_in_gpu and gm_diff < 0.001:
        print("✓✓✓ TEST PASSED - CPU and GPU agree!")
    elif not missing_in_gpu and not extra_in_gpu:
        print("⚠ Phase sets match but energy differs")
    elif missing_in_gpu and not extra_in_gpu:
        print("✗ GPU is missing some phases (under-prediction)")
        print("This suggests the GPU solver may be converging to a local minimum")
        print("or missing phase stability checks")
    elif not missing_in_gpu and extra_in_gpu:
        print("✗ GPU has extra phases (over-prediction)")
        print("This suggests the GPU solver may have incorrect phase stability criteria")
    else:
        print("✗ GPU has both missing and extra phases")
        print("This indicates significant differences in the solver behavior")
    
    # Specific issue for this case
    if 'L12' in missing_in_gpu:
        print("\n--- L12 Phase Issue ---")
        print("GPU is missing the L12 phase which CPU finds stable")
        print(f"CPU L12 amount: {cpu_dict.get('L12', 0):.6f}")
        print("This is a small phase fraction (~2.6%) that GPU solver may be missing")
        print("due to convergence criteria or phase search strategy")

if __name__ == "__main__":
    test_with_fixed_extraction()