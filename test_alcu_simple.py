#!/usr/bin/env python
"""
Simple test for Al-Cu-Fe system focusing on ALCU_ZETA phase
"""

import pycalphad as cp
import numpy as np

def test_alcu_simple():
    print("Testing Al-Cu-Fe system with ALCU_ZETA phase")
    print("=" * 60)
    
    # Load database
    db = cp.Database('Al-Cu-Fe.tdb')
    components = ['AL', 'CU', 'FE', 'VA']
    
    # Test with just a few phases including ALCU_ZETA
    phases = ['LIQUID', 'FCC_A1', 'ALCU_ZETA']
    
    # Original problematic condition: X(AL)=0.7, X(CU)=0.2, T=600K
    conditions = {
        cp.v.X('AL'): 0.7,
        cp.v.X('CU'): 0.2,
        cp.v.T: 600,
        cp.v.P: 101325
    }
    
    print(f"\nPhases: {phases}")
    print(f"Conditions: X(AL)={conditions[cp.v.X('AL')]}, X(CU)={conditions[cp.v.X('CU')]}, T={conditions[cp.v.T]}K")
    print(f"X(FE) = {1 - conditions[cp.v.X('AL')] - conditions[cp.v.X('CU')]}")
    
    # CPU calculation
    print("\nCPU calculation...")
    try:
        eq_cpu = cp.equilibrium(db, components, phases, conditions, verbose=False)
        
        # Extract results more carefully
        if hasattr(eq_cpu, 'GM') and hasattr(eq_cpu.GM, 'values'):
            gm_values = eq_cpu.GM.values
            if len(gm_values.shape) > 0 and gm_values.shape[0] > 0:
                cpu_gm = float(gm_values.flat[0])  # Use flat to handle any shape
                print(f"  CPU GM = {cpu_gm:.6f} J/mol")
                
                # Check stable phases
                if hasattr(eq_cpu, 'NP') and hasattr(eq_cpu.NP, 'values'):
                    np_values = eq_cpu.NP.values
                    phase_values = eq_cpu.Phase.values
                    
                    print("  CPU stable phases:")
                    for i in range(len(phase_values.flat)):
                        if i < len(np_values.flat) and np_values.flat[i] > 1e-6:
                            print(f"    {phase_values.flat[i]}: {np_values.flat[i]:.6f}")
            else:
                print("  CPU calculation returned empty results")
                return
        else:
            print("  CPU calculation did not return expected format")
            return
            
    except Exception as e:
        print(f"  CPU Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # GPU calculation  
    print("\nGPU calculation...")
    try:
        eq_gpu = cp.equilibrium(db, components, phases, conditions, gpu=True, verbose=False)
        
        # Extract results
        if hasattr(eq_gpu, 'GM') and hasattr(eq_gpu.GM, 'values'):
            gm_values = eq_gpu.GM.values
            if len(gm_values.shape) > 0 and gm_values.shape[0] > 0:
                gpu_gm = float(gm_values.flat[0])
                print(f"  GPU GM = {gpu_gm:.6f} J/mol")
                
                # Check stable phases
                if hasattr(eq_gpu, 'NP') and hasattr(eq_gpu.NP, 'values'):
                    np_values = eq_gpu.NP.values
                    phase_values = eq_gpu.Phase.values
                    
                    print("  GPU stable phases:")
                    for i in range(len(phase_values.flat)):
                        if i < len(np_values.flat) and np_values.flat[i] > 1e-6:
                            print(f"    {phase_values.flat[i]}: {np_values.flat[i]:.6f}")
                            
                # Calculate error
                error = abs(cpu_gm - gpu_gm)
                print(f"\nError: {error:.6f} J/mol")
                
                if error < 1.0:
                    print("✅ PASS: Error < 1 J/mol")
                else:
                    print(f"❌ FAIL: Error = {error:.3f} J/mol")
                    
                # Check improvement from 806 J/mol
                if error < 806:
                    improvement = 806 / error if error > 0 else float('inf')
                    print(f"Improvement from original 806 J/mol error: {improvement:.1f}x")
                    
            else:
                print("  GPU calculation returned empty results")
        else:
            print("  GPU calculation did not return expected format")
            
    except Exception as e:
        print(f"  GPU Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_alcu_simple()