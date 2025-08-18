#!/usr/bin/env python
"""Test the specific failing condition WITHOUT pdens to get correct comparison."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def test_without_pdens():
    """Test without pdens as per CLAUDE.md instructions."""
    
    # Load the Al-Cu-Fe database
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    
    # All 21 phases
    phases = ['AL13FE4', 'AL2FE', 'AL5FE2', 'AL5FE4', 'ALCU_DELTA', 'ALCU_EPSILON', 
              'ALCU_ETA', 'ALCU_PRIME', 'ALCU_THETA', 'ALCU_ZETA', 'BCC_A2', 'BCC_B2', 
              'FCC_A1', 'GAMMA_D83', 'GAMMA_H', 'L12', 'LIQUID', 'TS01T1', 'TS01T2', 
              'TS01T3', 'TS01TI']
    
    print("=" * 80)
    print("Testing WITHOUT pdens parameter (correct comparison per CLAUDE.md)")
    print("=" * 80)
    
    # Test condition where phase mismatch was reported
    conditions = {
        v.X('AL'): 0.40,
        v.X('CU'): 0.25,
        v.T: 700,
        v.P: 101325
    }
    
    print(f"\nCondition: X(AL)=0.40, X(CU)=0.25, X(FE)=0.35, T=700K")
    print("-" * 80)
    
    # CPU calculation WITHOUT pdens
    print("\n--- CPU Calculation (no pdens) ---")
    cpu_result = equilibrium(dbf, comps, phases, conditions, 
                            gpu=False, verbose=False)
    
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
                    cpu_phases.append((phase, np_val))
    
    print(f"CPU GM: {cpu_gm:.6f} J/mol")
    print(f"CPU MU(AL): {cpu_mu_al:.6f} J/mol")
    print(f"CPU MU(CU): {cpu_mu_cu:.6f} J/mol")
    print(f"CPU MU(FE): {cpu_mu_fe:.6f} J/mol")
    print(f"CPU Stable phases:")
    for phase, amount in cpu_phases:
        print(f"  {phase}: {amount:.6f}")
    
    # GPU calculation WITHOUT pdens
    print("\n--- GPU Calculation (no pdens) ---")
    gpu_result = equilibrium(dbf, comps, phases, conditions,
                            gpu=True, verbose=False)
    
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
                    gpu_phases.append((phase, np_val))
    
    print(f"GPU GM: {gpu_gm:.6f} J/mol")
    print(f"GPU MU(AL): {gpu_mu_al:.6f} J/mol")
    print(f"GPU MU(CU): {gpu_mu_cu:.6f} J/mol")
    print(f"GPU MU(FE): {gpu_mu_fe:.6f} J/mol")
    print(f"GPU Stable phases:")
    for phase, amount in gpu_phases:
        print(f"  {phase}: {amount:.6f}")
    
    # Compare results
    print("\n--- Comparison (WITHOUT pdens) ---")
    gm_diff = abs(gpu_gm - cpu_gm)
    mu_al_diff = abs(gpu_mu_al - cpu_mu_al)
    mu_cu_diff = abs(gpu_mu_cu - cpu_mu_cu)
    mu_fe_diff = abs(gpu_mu_fe - cpu_mu_fe)
    
    print(f"GM difference: {gm_diff:.6f} J/mol")
    print(f"MU(AL) difference: {mu_al_diff:.6f} J/mol")
    print(f"MU(CU) difference: {mu_cu_diff:.6f} J/mol")
    print(f"MU(FE) difference: {mu_fe_diff:.6f} J/mol")
    
    cpu_phase_names = [p for p, _ in cpu_phases]
    gpu_phase_names = [p for p, _ in gpu_phases]
    
    if set(cpu_phase_names) != set(gpu_phase_names):
        print(f"\nPHASE MISMATCH!")
        print(f"  CPU phases: {', '.join(cpu_phase_names)}")
        print(f"  GPU phases: {', '.join(gpu_phase_names)}")
        missing_in_gpu = set(cpu_phase_names) - set(gpu_phase_names)
        extra_in_gpu = set(gpu_phase_names) - set(cpu_phase_names)
        if missing_in_gpu:
            print(f"  Missing in GPU: {', '.join(missing_in_gpu)}")
        if extra_in_gpu:
            print(f"  Extra in GPU: {', '.join(extra_in_gpu)}")
    else:
        print(f"\n✓ Phase sets match!")
    
    # Check tolerance for numerical values
    tolerance = 0.001  # 0.001 J/mol
    numerical_match = (gm_diff < tolerance and 
                       mu_al_diff < tolerance and 
                       mu_cu_diff < tolerance and 
                       mu_fe_diff < tolerance)
    
    if numerical_match:
        print("✓ Numerical values match within tolerance")
    else:
        print("✗ Numerical differences exceed tolerance")
    
    if set(cpu_phase_names) == set(gpu_phase_names) and numerical_match:
        print("\n✓✓✓ TEST PASSED - CPU and GPU agree without pdens!")
    else:
        print("\n✗✗✗ TEST FAILED - Differences remain even without pdens")
        print("\nThis indicates a real issue in the GPU code that needs to be fixed.")

if __name__ == "__main__":
    test_without_pdens()