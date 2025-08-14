#!/usr/bin/env python
"""Test how results diverge as we add more phases."""

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
    
    # Get active phases
    cpu_np = cpu_result.NP.values.flatten()
    gpu_np = gpu_result.NP.values.flatten()
    
    cpu_active = [phase_list[i] for i in range(len(phase_list)) 
                  if i < len(cpu_np) and cpu_np[i] > 0.001]
    gpu_active = [phase_list[i] for i in range(len(phase_list)) 
                  if i < len(gpu_np) and gpu_np[i] > 0.001]
    
    return cpu_gm, gpu_gm, diff, cpu_active, gpu_active

def main():
    """Test with increasing phase complexity."""
    
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    all_phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2', 'L12', 'ALCU_THETA', 'AL13FE4', 'AL5FE2']
    
    # The failing condition
    conditions = {
        v.X('AL'): 0.2,
        v.X('CU'): 0.5,
        v.T: 900,
        v.P: 101325
    }
    
    print("=" * 80)
    print("TESTING PHASE COMPLEXITY EFFECT ON CONVERGENCE")
    print("Condition: X(AL)=0.2, X(CU)=0.5, T=900K")
    print("=" * 80)
    
    print("\nPhases | CPU GM    | GPU GM    | Diff    | Status | Active Phases")
    print("-------|-----------|-----------|---------|--------|---------------")
    
    # Test with increasing number of phases
    for n_phases in range(2, len(all_phases) + 1):
        phase_list = all_phases[:n_phases]
        
        cpu_gm, gpu_gm, diff, cpu_active, gpu_active = test_with_phases(
            dbf, comps, phase_list, conditions)
        
        status = "✓" if diff < 100 else "✗"
        
        # Format active phases
        if cpu_active == gpu_active:
            active_str = ', '.join(cpu_active)
        else:
            active_str = f"CPU: {', '.join(cpu_active)} | GPU: {', '.join(gpu_active)}"
        
        print(f"   {n_phases}   | {cpu_gm:9.1f} | {gpu_gm:9.1f} | {diff:7.1f} | {status}      | {active_str}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS:")
    print("-" * 80)
    
    # Test just the problematic phase combinations
    print("\nTesting specific phase combinations:")
    
    test_cases = [
        (['LIQUID', 'FCC_A1'], "Just the two active phases"),
        (['LIQUID', 'FCC_A1', 'BCC_A2'], "Active + 1 inactive"),
        (['LIQUID', 'FCC_A1', 'ALCU_THETA'], "Active + ALCU_THETA"),
        (['LIQUID', 'FCC_A1', 'L12'], "Active + L12"),
    ]
    
    for phase_list, description in test_cases:
        cpu_gm, gpu_gm, diff, cpu_active, gpu_active = test_with_phases(
            dbf, comps, phase_list, conditions)
        
        status = "✓" if diff < 100 else "✗"
        print(f"\n{description}:")
        print(f"  Phases tested: {', '.join(phase_list)}")
        print(f"  CPU GM: {cpu_gm:.1f}, GPU GM: {gpu_gm:.1f}, Diff: {diff:.1f} {status}")
        print(f"  Active: {', '.join(cpu_active)}")
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("The divergence appears when we have more phases in the calculation,")
    print("even if only 2 phases are actually stable. This suggests the issue")
    print("is in how the solver handles the larger phase space, not a fundamental")
    print("binary vs ternary problem.")
    print("=" * 80)

if __name__ == "__main__":
    main()