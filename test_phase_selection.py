#!/usr/bin/env python
"""Test phase selection and starting point differences between CPU and GPU."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def test_with_subset_phases(dbf, comps, x_al, x_cu, t):
    """Test different phase combinations."""
    
    all_phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2', 'L12_FCC', 
                  'ALCU_THETA', 'AL13FE4_D03', 'AL5FE2_D82']
    
    conditions = {
        v.X('AL'): x_al,
        v.X('CU'): x_cu,
        v.T: t,
        v.P: 101325
    }
    
    print("=" * 80)
    print(f"Testing X(AL)={x_al:.2f}, X(CU)={x_cu:.2f}, X(FE)={1-x_al-x_cu:.2f}, T={t}K")
    print("=" * 80)
    
    # Test with all phases
    print("\nWith ALL 8 phases:")
    cpu_all = equilibrium(dbf, comps, all_phases, conditions, gpu=False, verbose=False)
    gpu_all = equilibrium(dbf, comps, all_phases, conditions, gpu=True, verbose=False)
    cpu_gm_all = float(cpu_all.GM.values.squeeze())
    gpu_gm_all = float(gpu_all.GM.values.squeeze())
    diff_all = gpu_gm_all - cpu_gm_all
    print(f"  CPU: {cpu_gm_all:.2f} J/mol")
    print(f"  GPU: {gpu_gm_all:.2f} J/mol")
    print(f"  Diff: {diff_all:.2f} J/mol")
    
    # Get active phases for each
    cpu_active = []
    gpu_active = []
    for phase in all_phases:
        try:
            cpu_np = float(cpu_all.NP.sel(phase=phase).values.squeeze())
            if cpu_np > 1e-6:
                cpu_active.append(phase)
        except:
            pass
        try:
            gpu_np = float(gpu_all.NP.sel(phase=phase).values.squeeze())
            if gpu_np > 1e-6:
                gpu_active.append(phase)
        except:
            pass
    
    print(f"  CPU active: {cpu_active}")
    print(f"  GPU active: {gpu_active}")
    
    # Test with subsets
    test_sets = [
        ['LIQUID', 'FCC_A1', 'BCC_A2'],  # Simple 3-phase
        ['FCC_A1', 'BCC_A2', 'BCC_B2', 'L12_FCC'],  # Solid phases only
        ['LIQUID', 'FCC_A1', 'ALCU_THETA'],  # Liquid + 2 solids
        ['FCC_A1', 'BCC_A2', 'ALCU_THETA', 'AL13FE4_D03'],  # 4 complex phases
    ]
    
    for i, phases in enumerate(test_sets, 1):
        print(f"\nTest {i}: {phases}")
        cpu_test = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
        gpu_test = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
        cpu_gm = float(cpu_test.GM.values.squeeze())
        gpu_gm = float(gpu_test.GM.values.squeeze())
        diff = gpu_gm - cpu_gm
        
        status = "MATCH" if abs(diff) < 1.0 else "DIFF"
        print(f"  CPU: {cpu_gm:.2f}, GPU: {gpu_gm:.2f}, Diff: {diff:.2f} [{status}]")
        
        # Check which phases are active
        cpu_test_active = []
        gpu_test_active = []
        for phase in phases:
            try:
                cpu_np = float(cpu_test.NP.sel(phase=phase).values.squeeze())
                if cpu_np > 1e-6:
                    cpu_test_active.append(phase)
            except:
                pass
            try:
                gpu_np = float(gpu_test.NP.sel(phase=phase).values.squeeze())
                if gpu_np > 1e-6:
                    gpu_test_active.append(phase)
            except:
                pass
        
        if cpu_test_active != gpu_test_active:
            print(f"    Different phases! CPU: {cpu_test_active}, GPU: {gpu_test_active}")

def main():
    dbf = Database('Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    
    # Test the failing point
    test_with_subset_phases(dbf, comps, 0.40, 0.40, 600)
    
    # Test a nearby point that works
    print("\n" + "=" * 80)
    print("TESTING NEARBY WORKING POINT")
    test_with_subset_phases(dbf, comps, 0.39, 0.40, 600)

if __name__ == "__main__":
    main()