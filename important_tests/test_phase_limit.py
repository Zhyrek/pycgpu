#!/usr/bin/env python
"""Test GPU phase limit behavior."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def main():
    """Test phase limit."""
    
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    
    # Get ALL phases
    all_phases = list(dbf.phases.keys())
    
    print("=" * 80)
    print("GPU PHASE LIMIT TEST")
    print("=" * 80)
    
    # Test condition
    conditions = {
        v.X('AL'): 0.2,
        v.X('CU'): 0.5,
        v.T: 900,
        v.P: 101325
    }
    
    # Test with different numbers of phases
    for n_phases in [4, 6, 8, 10, 12, 15, 21]:
        phase_subset = all_phases[:n_phases]
        
        print(f"\n" + "-" * 40)
        print(f"Testing with {n_phases} phases:")
        print(f"Phases: {phase_subset[:5]}..." if len(phase_subset) > 5 else f"Phases: {phase_subset}")
        
        try:
            # CPU
            cpu_result = equilibrium(dbf, comps, phase_subset, conditions,
                                    calc_opts={'pdens': 50},
                                    gpu=False, verbose=False)
            cpu_gm = cpu_result.GM.values.item()
            
            # GPU
            gpu_result = equilibrium(dbf, comps, phase_subset, conditions,
                                    calc_opts={'pdens': 50},
                                    gpu=True, verbose=False)
            gpu_gm = gpu_result.GM.values.item()
            
            diff = abs(gpu_gm - cpu_gm)
            
            # Count active phases
            gpu_np = gpu_result.NP.values.flatten()
            n_active = sum(1 for i in range(len(gpu_np)) if i < len(phase_subset) and gpu_np[i] > 0.001)
            
            print(f"  CPU GM: {cpu_gm:.1f}")
            print(f"  GPU GM: {gpu_gm:.1f}")
            print(f"  Difference: {diff:.1f} J/mol")
            print(f"  Active phases in GPU result: {n_active}")
            
            if diff > 100:
                print(f"  ✗ DIVERGENCE with {n_phases} phases!")
            else:
                print(f"  ✓ Results match with {n_phases} phases")
                
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)[:100]}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS:")
    print("-" * 80)
    
    print("The GPU appears to handle >8 phases by:")
    print("  1. Possibly truncating to first 8 phases")
    print("  2. Or selecting most relevant phases")
    print("  3. Or using a different algorithm for large phase sets")
    
    # Now test which specific phases cause issues
    print("\n" + "-" * 80)
    print("Testing specific phase combinations:")
    print("-" * 80)
    
    # Test without BCC_B2
    phases_no_b2 = [p for p in all_phases[:8] if p != 'BCC_B2']
    
    print(f"\nWithout BCC_B2 ({len(phases_no_b2)} phases):")
    
    cpu_result = equilibrium(dbf, comps, phases_no_b2, conditions,
                            calc_opts={'pdens': 50},
                            gpu=False, verbose=False)
    gpu_result = equilibrium(dbf, comps, phases_no_b2, conditions,
                            calc_opts={'pdens': 50},
                            gpu=True, verbose=False)
    
    cpu_gm = cpu_result.GM.values.item()
    gpu_gm = gpu_result.GM.values.item()
    diff = abs(gpu_gm - cpu_gm)
    
    print(f"  CPU GM: {cpu_gm:.1f}, GPU GM: {gpu_gm:.1f}, Diff: {diff:.1f}")
    
    if diff < 100:
        print("  ✓ Removing BCC_B2 fixes the divergence!")
    else:
        print("  ✗ Still diverges without BCC_B2")
    
    print("=" * 80)

if __name__ == "__main__":
    main()