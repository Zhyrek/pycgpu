#!/usr/bin/env python
"""Test a specific failing condition with detailed debugging."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def test_specific_condition():
    """Test the specific failing condition from all-phases test."""
    
    # Load the Al-Cu-Fe database
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    
    # All 21 phases from the Al-Cu-Fe system
    phases = ['AL13FE4', 'AL2FE', 'AL5FE2', 'AL5FE4', 'ALCU_DELTA', 'ALCU_EPSILON', 
              'ALCU_ETA', 'ALCU_PRIME', 'ALCU_THETA', 'ALCU_ZETA', 'BCC_A2', 'BCC_B2', 
              'FCC_A1', 'GAMMA_D83', 'GAMMA_H', 'L12', 'LIQUID', 'TS01T1', 'TS01T2', 
              'TS01T3', 'TS01TI']
    
    # The specific failing condition: X(AL)=0.40, X(CU)=0.25, X(FE)=0.35, T=700K
    conditions = {
        v.X('AL'): 0.40,
        v.X('CU'): 0.25,
        v.T: 700,
        v.P: 101325
    }
    
    print("=" * 80)
    print("Testing specific failing condition:")
    print(f"X(AL)=0.40, X(CU)=0.25, X(FE)=0.35, T=700K")
    print("=" * 80)
    
    # Run CPU calculation (without pdens as per CLAUDE.md)
    print("\n--- CPU Calculation ---")
    cpu_result = equilibrium(dbf, comps, phases, conditions,
                            gpu=False, verbose=True)
    
    cpu_gm = cpu_result.GM.values.item()
    cpu_mu_al = cpu_result.MU.sel(component='AL').values.item()
    cpu_mu_cu = cpu_result.MU.sel(component='CU').values.item()
    cpu_mu_fe = cpu_result.MU.sel(component='FE').values.item()
    
    # Get stable phases for CPU
    cpu_phases = []
    for phase in phases:
        if phase in cpu_result.Phase.values:
            idx = np.where(cpu_result.Phase.values == phase)[0]
            if len(idx) > 0:
                np_val = cpu_result.NP.values.flat[idx[0]]
                if np_val > 1e-6:
                    cpu_phases.append(phase)
    
    print(f"CPU GM: {cpu_gm:.6f} J/mol")
    print(f"CPU MU(AL): {cpu_mu_al:.6f} J/mol")
    print(f"CPU MU(CU): {cpu_mu_cu:.6f} J/mol")
    print(f"CPU MU(FE): {cpu_mu_fe:.6f} J/mol")
    print(f"CPU Stable phases: {','.join(cpu_phases)}")
    
    # Run GPU calculation
    print("\n--- GPU Calculation ---")
    gpu_result = equilibrium(dbf, comps, phases, conditions,
                            gpu=True, verbose=True)
    
    gpu_gm = gpu_result.GM.values.item()
    gpu_mu_al = gpu_result.MU.sel(component='AL').values.item()
    gpu_mu_cu = gpu_result.MU.sel(component='CU').values.item()
    gpu_mu_fe = gpu_result.MU.sel(component='FE').values.item()
    
    # Get stable phases for GPU
    gpu_phases = []
    for phase in phases:
        if phase in gpu_result.Phase.values:
            idx = np.where(gpu_result.Phase.values == phase)[0]
            if len(idx) > 0:
                np_val = gpu_result.NP.values.flat[idx[0]]
                if np_val > 1e-6:
                    gpu_phases.append(phase)
    
    print(f"GPU GM: {gpu_gm:.6f} J/mol")
    print(f"GPU MU(AL): {gpu_mu_al:.6f} J/mol")
    print(f"GPU MU(CU): {gpu_mu_cu:.6f} J/mol")
    print(f"GPU MU(FE): {gpu_mu_fe:.6f} J/mol")
    print(f"GPU Stable phases: {','.join(gpu_phases)}")
    
    # Compare results
    print("\n--- Comparison ---")
    gm_diff = abs(gpu_gm - cpu_gm)
    mu_al_diff = abs(gpu_mu_al - cpu_mu_al)
    mu_cu_diff = abs(gpu_mu_cu - cpu_mu_cu)
    mu_fe_diff = abs(gpu_mu_fe - cpu_mu_fe)
    
    print(f"GM difference: {gm_diff:.6f} J/mol")
    print(f"MU(AL) difference: {mu_al_diff:.6f} J/mol")
    print(f"MU(CU) difference: {mu_cu_diff:.6f} J/mol")
    print(f"MU(FE) difference: {mu_fe_diff:.6f} J/mol")
    
    if set(cpu_phases) != set(gpu_phases):
        print(f"PHASE MISMATCH!")
        print(f"  CPU phases: {','.join(cpu_phases)}")
        print(f"  GPU phases: {','.join(gpu_phases)}")
    
    # Check tolerance
    tolerance = 0.001  # 0.001 J/mol for precise agreement
    if gm_diff < tolerance and mu_al_diff < tolerance and mu_cu_diff < tolerance and mu_fe_diff < tolerance:
        if set(cpu_phases) == set(gpu_phases):
            print("\n✓ TEST PASSED")
        else:
            print("\n✗ TEST FAILED (phase mismatch)")
    else:
        print("\n✗ TEST FAILED (numerical difference)")

if __name__ == "__main__":
    test_specific_condition()