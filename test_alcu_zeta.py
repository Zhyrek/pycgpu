#!/usr/bin/env python
"""
Test script for ALCU_ZETA multi-sublattice phase CPU vs GPU comparison
This tests the fixes for the 806 J/mol error reported in the conversation summary
"""

import pycalphad as cp
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def test_alcu_zeta_phase():
    """Test CPU vs GPU with ALCU_ZETA multi-sublattice phase"""
    
    print("="*70)
    print("Testing Al-Cu-Fe System with ALCU_ZETA Multi-Sublattice Phase")
    print("="*70)
    
    # Load the Al-Cu-Fe database from current directory
    try:
        db = cp.Database('Al-Cu-Fe.tdb')
        print(f"✓ Loaded Al-Cu-Fe.tdb database")
        print(f"  Elements: {db.elements}")
        print(f"  Total phases: {len(db.phases)}")
    except Exception as e:
        print(f"✗ Error loading database: {e}")
        return
    
    # Verify ALCU_ZETA phase exists and check its properties
    if 'ALCU_ZETA' in db.phases:
        phase = db.phases['ALCU_ZETA']
        print(f"\n✓ ALCU_ZETA phase found!")
        print(f"  Sublattices: {phase.sublattices}")
        print(f"  Site ratio sum: {sum(phase.sublattices)}")
        print(f"  Constituents: {phase.constituents}")
    else:
        print("✗ ALCU_ZETA phase not found in database!")
        return
    
    # Set up the system
    components = ['AL', 'CU', 'FE', 'VA']
    
    # Test different phase combinations and conditions
    test_cases = [
        {
            'name': 'Simple two-phase (baseline)',
            'phases': ['LIQUID', 'FCC_A1'],
            'conditions': {cp.v.X('AL'): 0.7, cp.v.X('CU'): 0.2, cp.v.T: 800, cp.v.P: 101325}
        },
        {
            'name': 'With ALCU_ZETA at high temperature',
            'phases': ['LIQUID', 'FCC_A1', 'ALCU_ZETA'],
            'conditions': {cp.v.X('AL'): 0.7, cp.v.X('CU'): 0.2, cp.v.T: 800, cp.v.P: 101325}
        },
        {
            'name': 'With ALCU_ZETA at medium temperature',
            'phases': ['LIQUID', 'FCC_A1', 'ALCU_ZETA'],
            'conditions': {cp.v.X('AL'): 0.5, cp.v.X('CU'): 0.4, cp.v.T: 600, cp.v.P: 101325}
        },
        {
            'name': 'Original problematic condition (from summary)',
            'phases': ['LIQUID', 'FCC_A1', 'ALCU_ZETA'],
            'conditions': {cp.v.X('AL'): 0.7, cp.v.X('CU'): 0.2, cp.v.T: 600, cp.v.P: 101325},
            'note': 'This is the exact condition that gave 806 J/mol error'
        }
    ]
    
    results = []
    
    for i, test in enumerate(test_cases):
        print(f"\n{'='*70}")
        print(f"Test Case {i+1}: {test['name']}")
        print(f"{'='*70}")
        print(f"Phases: {test['phases']}")
        
        # Extract condition values for display
        x_al = test['conditions'][cp.v.X('AL')]
        x_cu = test['conditions'][cp.v.X('CU')]
        x_fe = 1.0 - x_al - x_cu  # Fe makes up the rest
        temp = test['conditions'][cp.v.T]
        
        print(f"Conditions: X(AL)={x_al:.2f}, X(CU)={x_cu:.2f}, X(FE)={x_fe:.2f}, T={temp}K")
        
        if 'note' in test:
            print(f"Note: {test['note']}")
        
        try:
            # CPU calculation
            print("\nRunning CPU calculation...")
            eq_cpu = cp.equilibrium(db, components, test['phases'], test['conditions'], 
                                   verbose=False, debug=False)
            cpu_gm = float(eq_cpu.GM.values[0])
            
            # Check which phases are stable
            cpu_stable_phases = []
            cpu_phase_amounts = []
            for j in range(len(eq_cpu.Phase.values[0])):
                if eq_cpu.NP.values[0][j] > 1e-6:
                    cpu_stable_phases.append(eq_cpu.Phase.values[0][j])
                    cpu_phase_amounts.append(eq_cpu.NP.values[0][j])
            
            print(f"✓ CPU Success:")
            print(f"  GM = {cpu_gm:.3f} J/mol")
            print(f"  Stable phases: {cpu_stable_phases}")
            print(f"  Phase amounts: {[f'{amt:.3f}' for amt in cpu_phase_amounts]}")
            
            # GPU calculation
            print("\nRunning GPU calculation...")
            try:
                eq_gpu = cp.equilibrium(db, components, test['phases'], test['conditions'], 
                                       gpu=True, verbose=False, debug=False)
                gpu_gm = float(eq_gpu.GM.values[0])
                
                # Check which phases are stable
                gpu_stable_phases = []
                gpu_phase_amounts = []
                for j in range(len(eq_gpu.Phase.values[0])):
                    if eq_gpu.NP.values[0][j] > 1e-6:
                        gpu_stable_phases.append(eq_gpu.Phase.values[0][j])
                        gpu_phase_amounts.append(eq_gpu.NP.values[0][j])
                
                print(f"✓ GPU Success:")
                print(f"  GM = {gpu_gm:.3f} J/mol")
                print(f"  Stable phases: {gpu_stable_phases}")
                print(f"  Phase amounts: {[f'{amt:.3f}' for amt in gpu_phase_amounts]}")
                
                # Calculate error
                error = abs(cpu_gm - gpu_gm)
                
                print(f"\nError Analysis:")
                print(f"  CPU GM: {cpu_gm:.6f} J/mol")
                print(f"  GPU GM: {gpu_gm:.6f} J/mol")
                print(f"  Absolute error: {error:.6f} J/mol")
                
                if error < 1.0:
                    print(f"  ✅ PASS: Error within 1 J/mol tolerance!")
                elif error < 10.0:
                    print(f"  ⚠️  ACCEPTABLE: Error < 10 J/mol")
                else:
                    print(f"  ❌ FAIL: Error > 10 J/mol")
                
                # Check if ALCU_ZETA is stable in both
                if 'ALCU_ZETA' in test['phases']:
                    cpu_has_zeta = 'ALCU_ZETA' in cpu_stable_phases
                    gpu_has_zeta = 'ALCU_ZETA' in gpu_stable_phases
                    
                    if cpu_has_zeta and gpu_has_zeta:
                        print(f"  ✓ ALCU_ZETA is stable in both CPU and GPU")
                    elif cpu_has_zeta and not gpu_has_zeta:
                        print(f"  ⚠️  ALCU_ZETA stable in CPU but not GPU")
                    elif not cpu_has_zeta and gpu_has_zeta:
                        print(f"  ⚠️  ALCU_ZETA stable in GPU but not CPU")
                    else:
                        print(f"  - ALCU_ZETA not stable in either calculation")
                
                results.append({
                    'test': test['name'],
                    'error': error,
                    'cpu_gm': cpu_gm,
                    'gpu_gm': gpu_gm,
                    'passed': error < 1.0
                })
                
            except Exception as e:
                print(f"✗ GPU Error: {e}")
                results.append({
                    'test': test['name'],
                    'error': None,
                    'cpu_gm': cpu_gm,
                    'gpu_gm': None,
                    'passed': False
                })
                
        except Exception as e:
            print(f"✗ CPU Error: {e}")
            results.append({
                'test': test['name'],
                'error': None,
                'cpu_gm': None,
                'gpu_gm': None,
                'passed': False
            })
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*70}")
    
    print("\nTest Results:")
    for r in results:
        if r['error'] is not None:
            status = "PASS" if r['passed'] else "FAIL"
            print(f"  {r['test']}: {r['error']:.3f} J/mol - {status}")
        else:
            print(f"  {r['test']}: ERROR")
    
    print("\nKey Findings:")
    print("1. Energy calculation fix: Changed pr->obj() to pr->formulaobj()")
    print("   - This fixes normalization by site ratios for multi-sublattice phases")
    print("2. Constraint counting fix: Use Gibbs phase rule formula")
    print("   - This ensures consistent matrix dimensions between CPU and GPU")
    
    print("\nOriginal Issue:")
    print("  - 806 J/mol error for ALCU_ZETA phase with site ratios (9.0, 11.0)")
    print("  - Site ratio sum = 20.0 was causing incorrect normalization")
    
    # Check if we tested the original problematic case
    for r in results:
        if 'Original problematic' in r['test'] and r['error'] is not None:
            print(f"\nCurrent Status:")
            print(f"  - Original error: 806 J/mol")
            print(f"  - Current error: {r['error']:.3f} J/mol")
            if r['error'] < 806:
                improvement = 806 / r['error']
                print(f"  - Improvement: {improvement:.1f}x better!")

if __name__ == "__main__":
    test_alcu_zeta_phase()