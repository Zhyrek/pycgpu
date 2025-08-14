#!/usr/bin/env python
"""Test which specific phase causes the divergence."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def test_with_phases(dbf, comps, phase_list, conditions):
    """Test with a specific set of phases."""
    
    # CPU
    cpu_result = equilibrium(dbf, comps, phase_list, conditions,
                            calc_opts={'pdens': 50},
                            gpu=False, verbose=False)
    cpu_gm = cpu_result.GM.values.item()
    
    # GPU  
    gpu_result = equilibrium(dbf, comps, phase_list, conditions,
                            calc_opts={'pdens': 50},
                            gpu=True, verbose=False)
    gpu_gm = gpu_result.GM.values.item()
    
    diff = abs(gpu_gm - cpu_gm)
    
    return cpu_gm, gpu_gm, diff

def main():
    """Test which specific phase causes issues."""
    
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    
    # The failing condition
    conditions = {
        v.X('AL'): 0.2,
        v.X('CU'): 0.5,
        v.T: 900,
        v.P: 101325
    }
    
    print("=" * 80)
    print("IDENTIFYING PROBLEMATIC PHASE")
    print("Condition: X(AL)=0.2, X(CU)=0.5, T=900K")
    print("=" * 80)
    
    # Base phases that work fine
    base_phases = ['LIQUID', 'FCC_A1', 'BCC_A2']
    
    # Additional phases to test one by one
    test_phases = ['BCC_B2', 'L12', 'ALCU_THETA', 'AL13FE4', 'AL5FE2']
    
    print("\nBase configuration (LIQUID + FCC_A1 + BCC_A2):")
    cpu_gm, gpu_gm, diff = test_with_phases(dbf, comps, base_phases, conditions)
    print(f"  CPU: {cpu_gm:.1f}, GPU: {gpu_gm:.1f}, Diff: {diff:.1f}")
    
    print("\nAdding each phase individually to base:")
    print("-" * 60)
    print("Added Phase  | CPU GM    | GPU GM    | Diff    | Status")
    print("-------------|-----------|-----------|---------|--------")
    
    for phase in test_phases:
        phase_list = base_phases + [phase]
        cpu_gm, gpu_gm, diff = test_with_phases(dbf, comps, phase_list, conditions)
        
        status = "✓ OK" if diff < 100 else "✗ DIVERGES"
        print(f"{phase:12s} | {cpu_gm:9.1f} | {gpu_gm:9.1f} | {diff:7.1f} | {status}")
    
    # Now test removing each phase from the full set
    print("\n" + "=" * 80)
    print("Testing by REMOVING each phase from full set:")
    print("-" * 60)
    
    all_phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2', 'L12', 'ALCU_THETA', 'AL13FE4', 'AL5FE2']
    
    print("Full set (all 8 phases):")
    cpu_gm, gpu_gm, diff = test_with_phases(dbf, comps, all_phases, conditions)
    print(f"  CPU: {cpu_gm:.1f}, GPU: {gpu_gm:.1f}, Diff: {diff:.1f}")
    
    print("\nRemoving each phase:")
    print("Removed Phase | CPU GM    | GPU GM    | Diff    | Status")
    print("--------------|-----------|-----------|---------|--------")
    
    for phase_to_remove in all_phases:
        phase_list = [p for p in all_phases if p != phase_to_remove]
        cpu_gm, gpu_gm, diff = test_with_phases(dbf, comps, phase_list, conditions)
        
        status = "✓ FIXED" if diff < 100 else "✗ Still bad"
        print(f"{phase_to_remove:13s} | {cpu_gm:9.1f} | {gpu_gm:9.1f} | {diff:7.1f} | {status}")
    
    # Test specific combinations
    print("\n" + "=" * 80)
    print("Testing specific combinations:")
    print("-" * 60)
    
    test_combos = [
        (['LIQUID', 'FCC_A1', 'BCC_B2'], "Just add BCC_B2"),
        (['LIQUID', 'FCC_A1', 'L12'], "Just add L12"),
        (['LIQUID', 'FCC_A1', 'ALCU_THETA'], "Just add ALCU_THETA"),
        (['LIQUID', 'FCC_A1', 'AL13FE4'], "Just add AL13FE4"),
        (['LIQUID', 'FCC_A1', 'AL5FE2'], "Just add AL5FE2"),
        (['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2'], "First 4 phases"),
        (['BCC_B2', 'L12', 'ALCU_THETA', 'AL13FE4'], "Without LIQUID/FCC"),
    ]
    
    for phase_list, description in test_combos:
        print(f"\n{description} ({', '.join(phase_list)}):")
        cpu_gm, gpu_gm, diff = test_with_phases(dbf, comps, phase_list, conditions)
        status = "✓" if diff < 100 else "✗"
        print(f"  CPU: {cpu_gm:.1f}, GPU: {gpu_gm:.1f}, Diff: {diff:.1f} {status}")
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("The specific phase causing divergence should be identified above.")
    print("=" * 80)

if __name__ == "__main__":
    main()