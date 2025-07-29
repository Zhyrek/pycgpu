#!/usr/bin/env python
"""Test AlCu system with dgelsd implementation to check multi-sublattice handling."""

import os
os.environ['PYCALPHAD_DEBUG_EQSOLVER'] = ''  # Disable debug output initially

from pycalphad import Database, equilibrium, variables as v
import numpy as np

# Load AlCuFe database
dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']
phases = ['LIQUID', 'ALCU_ZETA']  # ALCU_ZETA has 2 sublattices with site ratios (9.0, 11.0)

print("Testing AlCu system with dgelsd_device implementation")
print("="*70)
print("ALCU_ZETA has 2 sublattices with site ratios (9.0, 11.0)")
print("Site ratio sum = 20.0 (vs 1.0 for LIQUID)")
print("="*70)

# Test conditions from previous analysis
# Only specify 2 of 3 mole fractions to avoid overdetermination
test_conditions = [
    # (name, X(AL), X(CU), T) - X(FE) will be 1 - X(AL) - X(CU)
    ("Low T, Al-rich", 0.7, 0.2, 600),    # X(FE) = 0.1
    ("Med T, balanced", 0.6, 0.3, 900),   # X(FE) = 0.1 - This was the closest before
    ("High T, Cu-rich", 0.3, 0.6, 1200),  # X(FE) = 0.1
    ("Med T, more Cu", 0.4, 0.5, 900),    # X(FE) = 0.1
    ("Med T, more Al", 0.5, 0.4, 900),    # X(FE) = 0.1
]

results = []

for name, x_al, x_cu, T in test_conditions:
    x_fe = 1.0 - x_al - x_cu
    print(f"\n{name}: X(AL)={x_al}, X(CU)={x_cu}, X(FE)={x_fe}, T={T}K")
    print("-" * 50)
    
    conditions = {
        v.T: T, 
        v.P: 101325, 
        v.N: 1, 
        v.X('AL'): x_al,
        v.X('CU'): x_cu
    }
    
    try:
        # CPU calculation
        cpu_result = equilibrium(dbf, comps, phases, conditions, 
                               calc_opts={'pdens': 100}, verbose=False)
        cpu_gm = float(cpu_result.GM.values.item())
        cpu_phases = cpu_result.Phase.values.squeeze()
        cpu_phase_fracs = cpu_result.NP.values.squeeze()
        
        # GPU calculation
        gpu_result = equilibrium(dbf, comps, phases, conditions, 
                               calc_opts={'pdens': 100}, verbose=False, gpu=True)
        gpu_gm = float(gpu_result.GM.values.item())
        gpu_phases = gpu_result.Phase.values.squeeze()
        gpu_phase_fracs = gpu_result.NP.values.squeeze()
        
        # Compare results
        diff = abs(cpu_gm - gpu_gm)
        
        print(f"CPU GM: {cpu_gm:.6f} J/mol")
        print(f"GPU GM: {gpu_gm:.6f} J/mol")
        print(f"Difference: {diff:.2e} J/mol")
        
        # Check phase presence
        print(f"\nPhase presence:")
        if isinstance(cpu_phases, np.ndarray):
            for i, (phase, frac) in enumerate(zip(cpu_phases, cpu_phase_fracs)):
                if not pd.isna(phase) and frac > 1e-6:
                    print(f"  CPU Phase {i}: {phase} ({frac:.4f})")
        else:
            print(f"  CPU: {cpu_phases} ({cpu_phase_fracs:.4f})")
            
        if isinstance(gpu_phases, np.ndarray):
            for i, (phase, frac) in enumerate(zip(gpu_phases, gpu_phase_fracs)):
                if not pd.isna(phase) and frac > 1e-6:
                    print(f"  GPU Phase {i}: {phase} ({frac:.4f})")
        else:
            print(f"  GPU: {gpu_phases} ({gpu_phase_fracs:.4f})")
        
        # Check if phases match
        cpu_has_both = False
        gpu_has_both = False
        
        if isinstance(cpu_phases, np.ndarray) and len(cpu_phases) > 1:
            cpu_has_both = 'LIQUID' in cpu_phases and 'ALCU_ZETA' in cpu_phases
        if isinstance(gpu_phases, np.ndarray) and len(gpu_phases) > 1:
            gpu_has_both = 'LIQUID' in gpu_phases and 'ALCU_ZETA' in gpu_phases
            
        phase_match = cpu_has_both == gpu_has_both
        
        results.append({
            'name': name,
            'diff': diff,
            'cpu_has_both': cpu_has_both,
            'gpu_has_both': gpu_has_both,
            'phase_match': phase_match
        })
        
        if not phase_match:
            print("\n⚠️  PHASE MISMATCH: CPU and GPU have different phases!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        results.append({
            'name': name,
            'diff': float('inf'),
            'error': str(e)
        })

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

if results:
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        diffs = [r['diff'] for r in valid_results]
        avg_diff = np.mean(diffs)
        max_diff = np.max(diffs)
        max_case = next(r for r in valid_results if r['diff'] == max_diff)
        
        print(f"Valid test cases: {len(valid_results)}/{len(results)}")
        print(f"Average difference: {avg_diff:.2e} J/mol")
        print(f"Maximum difference: {max_diff:.2e} J/mol (at {max_case['name']})")
        
        # Check phase matching
        phase_matches = sum(1 for r in valid_results if r.get('phase_match', False))
        print(f"\nPhase matching: {phase_matches}/{len(valid_results)} cases")
        
        # Compare to previous results
        print(f"\nComparison to previous results:")
        print(f"Previous 'Med T, balanced' error: ~6.94 J/mol")
        print(f"Current 'Med T, balanced' error: ", end="")
        med_t_result = next((r for r in valid_results if r['name'] == "Med T, balanced"), None)
        if med_t_result:
            print(f"{med_t_result['diff']:.2e} J/mol")
            if med_t_result['diff'] < 6.94:
                improvement = 6.94 / med_t_result['diff']
                print(f"✓ IMPROVED by {improvement:.1f}x!")
            else:
                print("✗ No improvement")
                
        # Check if multi-sublattice issue is resolved
        print(f"\nMulti-sublattice phase handling:")
        both_phases_cases = sum(1 for r in valid_results if r.get('cpu_has_both', False))
        gpu_matches = sum(1 for r in valid_results if r.get('cpu_has_both', False) and r.get('gpu_has_both', False))
        print(f"Cases where CPU has both phases: {both_phases_cases}")
        print(f"Cases where GPU also has both phases: {gpu_matches}/{both_phases_cases}")
        
        if gpu_matches < both_phases_cases:
            print("\n⚠️  GPU still removing phases that CPU keeps!")
            print("The matrix conditioning improvement alone may not be sufficient.")
            print("Need to investigate other sources of divergence.")

# Import pandas only if needed
try:
    import pandas as pd
except ImportError:
    pd = None