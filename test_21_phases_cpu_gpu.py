#!/usr/bin/env python
"""
Test equilibrium calculation with 21 phases comparing CPU vs GPU results.
"""

import numpy as np
from pycalphad import Database, equilibrium
import pycalphad.variables as v
import time

def run_equilibrium_comparison():
    """Run equilibrium with 21 phases on both CPU and GPU and compare results."""
    
    # Load Al-Cu-Fe database with 21 phases
    dbf = Database('Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = list(dbf.phases.keys())  # All 21 phases
    
    print("=" * 80)
    print("TESTING EQUILIBRIUM WITH 21 PHASES: CPU vs GPU")
    print("=" * 80)
    print(f"Database: Al-Cu-Fe")
    print(f"Components: {comps}")
    print(f"Number of phases: {len(phases)}")
    print(f"Phases: {phases}")
    print()
    
    # Test multiple conditions to check consistency
    test_conditions = [
        # Condition 1: Original test
        {
            v.T: 800,
            v.P: 101325,
            v.X('AL'): 0.3,
            v.X('CU'): 0.3
        },
        # Condition 2: Different composition
        {
            v.T: 900,
            v.P: 101325,
            v.X('AL'): 0.4,
            v.X('CU'): 0.4
        },
        # Condition 3: High temperature
        {
            v.T: 1200,
            v.P: 101325,
            v.X('AL'): 0.5,
            v.X('CU'): 0.2
        },
        # Condition 4: Low temperature
        {
            v.T: 600,
            v.P: 101325,
            v.X('AL'): 0.2,
            v.X('CU'): 0.5
        },
    ]
    
    for i, conditions in enumerate(test_conditions, 1):
        print(f"\n{'=' * 60}")
        print(f"TEST CONDITION {i}")
        print(f"{'=' * 60}")
        print(f"T = {conditions[v.T]} K")
        print(f"P = {conditions[v.P]} Pa")
        print(f"X(AL) = {conditions[v.X('AL')]}")
        print(f"X(CU) = {conditions[v.X('CU')]}")
        print(f"X(FE) = {1 - conditions[v.X('AL')] - conditions[v.X('CU')]}")
        
        # Run CPU calculation
        print("\n--- Running CPU calculation ---")
        start_time = time.time()
        try:
            cpu_result = equilibrium(dbf, comps, phases, conditions, 
                                    verbose=False, gpu=False)
            cpu_time = time.time() - start_time
            print(f"CPU calculation completed in {cpu_time:.2f} seconds")
            
            # Extract CPU results
            cpu_gm = float(cpu_result.GM.values)
            cpu_phases = cpu_result.Phase.values[0, 0, 0, :]
            cpu_np = cpu_result.NP.values[0, 0, 0, :]
            
            # Filter out empty phases
            cpu_active_phases = [(p, np) for p, np in zip(cpu_phases, cpu_np) 
                                if p != '' and np > 1e-6]
            
            print(f"CPU GM: {cpu_gm:.6f} J/mol")
            print(f"CPU Active phases: {cpu_active_phases}")
            
        except Exception as e:
            print(f"CPU calculation failed: {e}")
            cpu_result = None
        
        # Run GPU calculation
        print("\n--- Running GPU calculation ---")
        start_time = time.time()
        try:
            gpu_result = equilibrium(dbf, comps, phases, conditions, 
                                    verbose=False, gpu=True)
            gpu_time = time.time() - start_time
            print(f"GPU calculation completed in {gpu_time:.2f} seconds")
            
            # Extract GPU results
            gpu_gm = float(gpu_result.GM.values)
            gpu_phases = gpu_result.Phase.values[0, 0, 0, :]
            gpu_np = gpu_result.NP.values[0, 0, 0, :]
            
            # Filter out empty phases
            gpu_active_phases = [(p, np) for p, np in zip(gpu_phases, gpu_np) 
                                if p != '' and np > 1e-6]
            
            print(f"GPU GM: {gpu_gm:.6f} J/mol")
            print(f"GPU Active phases: {gpu_active_phases}")
            
        except Exception as e:
            print(f"GPU calculation failed: {e}")
            import traceback
            traceback.print_exc()
            gpu_result = None
        
        # Compare results if both succeeded
        if cpu_result is not None and gpu_result is not None:
            print("\n--- Comparison ---")
            
            # Compare GM values
            gm_diff = abs(cpu_gm - gpu_gm)
            gm_rel_diff = gm_diff / abs(cpu_gm) * 100 if cpu_gm != 0 else 0
            
            print(f"GM difference: {gm_diff:.6f} J/mol ({gm_rel_diff:.4f}%)")
            
            if gm_diff < 1.0:  # Within 1 J/mol
                print("✓ GM values match within tolerance")
            else:
                print("✗ GM values differ significantly")
            
            # Compare phase assemblages
            cpu_phase_set = set(p for p, _ in cpu_active_phases)
            gpu_phase_set = set(p for p, _ in gpu_active_phases)
            
            if cpu_phase_set == gpu_phase_set:
                print("✓ Same phases in equilibrium")
                
                # Compare phase fractions
                cpu_phase_dict = dict(cpu_active_phases)
                gpu_phase_dict = dict(gpu_active_phases)
                
                max_np_diff = 0
                for phase in cpu_phase_set:
                    cpu_np = cpu_phase_dict[phase]
                    gpu_np = gpu_phase_dict[phase]
                    np_diff = abs(cpu_np - gpu_np)
                    max_np_diff = max(max_np_diff, np_diff)
                    if np_diff > 0.01:  # More than 1% difference
                        print(f"  Phase {phase}: CPU={cpu_np:.4f}, GPU={gpu_np:.4f}, diff={np_diff:.4f}")
                
                if max_np_diff < 0.01:
                    print("✓ Phase fractions match within tolerance")
                else:
                    print(f"✗ Maximum phase fraction difference: {max_np_diff:.4f}")
            else:
                print("✗ Different phases in equilibrium")
                print(f"  CPU phases: {cpu_phase_set}")
                print(f"  GPU phases: {gpu_phase_set}")
                print(f"  Only in CPU: {cpu_phase_set - gpu_phase_set}")
                print(f"  Only in GPU: {gpu_phase_set - cpu_phase_set}")
            
            # Speed comparison
            if cpu_time > 0:
                speedup = cpu_time / gpu_time
                print(f"\nSpeedup: {speedup:.2f}x")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_equilibrium_comparison()