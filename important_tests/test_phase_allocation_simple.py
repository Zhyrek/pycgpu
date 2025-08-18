#!/usr/bin/env python
"""Simple test to check phase allocation and AL13FE4 detection."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def test_al13fe4_detection():
    """Test if AL13FE4 is properly detected in equilibrium calculations."""
    
    # Load the Al-Cu-Fe database
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    
    # All 21 phases - but focus on AL13FE4
    phases = ['AL13FE4', 'AL2FE', 'AL5FE2', 'AL5FE4', 'ALCU_DELTA', 'ALCU_EPSILON', 
              'ALCU_ETA', 'ALCU_PRIME', 'ALCU_THETA', 'ALCU_ZETA', 'BCC_A2', 'BCC_B2', 
              'FCC_A1', 'GAMMA_D83', 'GAMMA_H', 'L12', 'LIQUID', 'TS01T1', 'TS01T2', 
              'TS01T3', 'TS01TI']
    
    print("=" * 80)
    print("Testing AL13FE4 phase detection")
    print("=" * 80)
    print(f"\nTotal phases: {len(phases)}")
    
    # Test condition where AL13FE4 should be stable
    conditions = {
        v.X('AL'): 0.40,
        v.X('CU'): 0.25,
        v.T: 700,
        v.P: 101325
    }
    
    print(f"\nCondition: X(AL)=0.40, X(CU)=0.25, X(FE)=0.35, T=700K")
    
    # First test with just AL13FE4, AL2FE, and AL5FE2 to isolate the problem
    test_phases = ['AL13FE4', 'AL2FE', 'AL5FE2']
    
    print(f"\n--- Testing with subset: {', '.join(test_phases)} ---")
    
    # CPU calculation with subset
    cpu_subset = equilibrium(dbf, comps, test_phases, conditions, gpu=False, verbose=False)
    cpu_gm_subset = cpu_subset.GM.values.item()
    
    # Get stable phases
    cpu_phases_subset = []
    for phase in test_phases:
        if phase in cpu_subset.Phase.values:
            idx = np.where(cpu_subset.Phase.values == phase)[0]
            if len(idx) > 0:
                np_val = cpu_subset.NP.values.flat[idx[0]]
                if np_val > 1e-6:
                    cpu_phases_subset.append((phase, np_val))
    
    print(f"CPU result (subset):")
    print(f"  GM = {cpu_gm_subset:.2f} J/mol")
    for phase, amount in cpu_phases_subset:
        print(f"  {phase}: {amount:.6f}")
    
    # GPU calculation with subset
    gpu_subset = equilibrium(dbf, comps, test_phases, conditions, gpu=True, verbose=False)
    gpu_gm_subset = gpu_subset.GM.values.item()
    
    # Get stable phases
    gpu_phases_subset = []
    for phase in test_phases:
        if phase in gpu_subset.Phase.values:
            idx = np.where(gpu_subset.Phase.values == phase)[0]
            if len(idx) > 0:
                np_val = gpu_subset.NP.values.flat[idx[0]]
                if np_val > 1e-6:
                    gpu_phases_subset.append((phase, np_val))
    
    print(f"GPU result (subset):")
    print(f"  GM = {gpu_gm_subset:.2f} J/mol")
    for phase, amount in gpu_phases_subset:
        print(f"  {phase}: {amount:.6f}")
    
    # Check if AL13FE4 is missing
    cpu_has_al13fe4 = any(phase == 'AL13FE4' for phase, _ in cpu_phases_subset)
    gpu_has_al13fe4 = any(phase == 'AL13FE4' for phase, _ in gpu_phases_subset)
    
    if cpu_has_al13fe4 and not gpu_has_al13fe4:
        print("\n✗ AL13FE4 is missing in GPU calculation!")
    elif cpu_has_al13fe4 and gpu_has_al13fe4:
        print("\n✓ AL13FE4 is correctly detected by both CPU and GPU")
    
    # Now test with all phases
    print(f"\n--- Testing with all {len(phases)} phases ---")
    
    # CPU calculation with all phases
    cpu_all = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
    cpu_gm_all = cpu_all.GM.values.item()
    
    # Get stable phases
    cpu_phases_all = []
    for phase in phases:
        if phase in cpu_all.Phase.values:
            idx = np.where(cpu_all.Phase.values == phase)[0]
            if len(idx) > 0:
                np_val = cpu_all.NP.values.flat[idx[0]]
                if np_val > 1e-6:
                    cpu_phases_all.append((phase, np_val))
    
    print(f"CPU result (all phases):")
    print(f"  GM = {cpu_gm_all:.2f} J/mol")
    for phase, amount in cpu_phases_all:
        print(f"  {phase}: {amount:.6f}")
    
    # GPU calculation with all phases
    gpu_all = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    gpu_gm_all = gpu_all.GM.values.item()
    
    # Get stable phases
    gpu_phases_all = []
    for phase in phases:
        if phase in gpu_all.Phase.values:
            idx = np.where(gpu_all.Phase.values == phase)[0]
            if len(idx) > 0:
                np_val = gpu_all.NP.values.flat[idx[0]]
                if np_val > 1e-6:
                    gpu_phases_all.append((phase, np_val))
    
    print(f"GPU result (all phases):")
    print(f"  GM = {gpu_gm_all:.2f} J/mol")
    for phase, amount in gpu_phases_all:
        print(f"  {phase}: {amount:.6f}")
    
    # Final comparison
    cpu_phase_names_all = [p for p, _ in cpu_phases_all]
    gpu_phase_names_all = [p for p, _ in gpu_phases_all]
    
    missing_in_gpu = set(cpu_phase_names_all) - set(gpu_phase_names_all)
    extra_in_gpu = set(gpu_phase_names_all) - set(cpu_phase_names_all)
    
    print("\n--- Final Diagnosis ---")
    if missing_in_gpu:
        print(f"✗ Phases missing in GPU: {', '.join(missing_in_gpu)}")
    if extra_in_gpu:
        print(f"✗ Extra phases in GPU: {', '.join(extra_in_gpu)}")
    if not missing_in_gpu and not extra_in_gpu:
        print("✓ Phase detection matches between CPU and GPU")
    
    gm_diff = abs(gpu_gm_all - cpu_gm_all)
    print(f"\nGM difference: {gm_diff:.6f} J/mol")
    
    if gm_diff > 0.001:
        print(f"✗ Significant energy difference detected!")
    else:
        print(f"✓ Energy values match within tolerance")

if __name__ == "__main__":
    test_al13fe4_detection()