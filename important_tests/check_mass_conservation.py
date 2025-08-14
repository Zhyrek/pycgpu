#!/usr/bin/env python
"""Check mass conservation in CPU vs GPU calculations."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def check_mass_balance(phases_info, bulk_composition):
    """Calculate overall composition from phase amounts and compositions."""
    
    total_al = 0
    total_cu = 0
    total_fe = 0
    total_amount = 0
    
    for phase in phases_info:
        amount = phase['amount']
        comp = phase['composition']
        
        total_al += amount * comp['AL']
        total_cu += amount * comp['CU']
        total_fe += amount * comp['FE']
        total_amount += amount
    
    # Normalize if not equal to 1
    if abs(total_amount - 1.0) > 0.001:
        print(f"  WARNING: Total phase amount = {total_amount:.6f} (should be 1.0)")
    
    return {
        'AL': total_al,
        'CU': total_cu,
        'FE': total_fe,
        'total': total_al + total_cu + total_fe
    }

def extract_phase_data(result):
    """Extract phase data with proper indexing."""
    phase_labels = result.Phase.values.flatten()
    np_vals = result.NP.values.flatten()
    x_vals = result.X.values
    
    active_phases = []
    n_comps = 3  # AL, CU, FE
    
    for i, (phase, amount) in enumerate(zip(phase_labels, np_vals)):
        if not np.isnan(amount) and amount > 0.001:
            # Extract composition - X is (conditions, phases, components)
            x_reshaped = x_vals.reshape(-1, n_comps)
            if i < len(x_reshaped):
                comp = {
                    'AL': x_reshaped[i][0],
                    'CU': x_reshaped[i][1],
                    'FE': x_reshaped[i][2]
                }
                active_phases.append({
                    'name': phase,
                    'amount': amount,
                    'composition': comp
                })
    
    return active_phases

def main():
    """Check mass conservation for the problematic case."""
    
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2', 'L12', 'ALCU_THETA', 'AL13FE4', 'AL5FE2']
    
    print("=" * 80)
    print("MASS CONSERVATION CHECK")
    print("=" * 80)
    
    # Test the problematic case and some others
    test_cases = [
        (0.60, 0.10, 600, "Main failure (706 J/mol diff)"),
        (0.59, 0.10, 600, "Neighbor (0 J/mol diff)"),
        (0.61, 0.10, 600, "Neighbor (0 J/mol diff)"),
        (0.40, 0.40, 600, "Another failure (41 J/mol diff)"),
        (0.30, 0.30, 900, "Working case for comparison")
    ]
    
    for x_al, x_cu, temp, desc in test_cases:
        x_fe = 1.0 - x_al - x_cu
        
        print(f"\n{desc}")
        print(f"Bulk composition: X(AL)={x_al:.3f}, X(CU)={x_cu:.3f}, X(FE)={x_fe:.3f}")
        print("-" * 80)
        
        conditions = {
            v.X('AL'): x_al,
            v.X('CU'): x_cu,
            v.T: temp,
            v.P: 101325
        }
        
        bulk_comp = {'AL': x_al, 'CU': x_cu, 'FE': x_fe}
        
        # CPU calculation
        cpu_result = equilibrium(dbf, comps, phases, conditions,
                                gpu=False, verbose=False)
        cpu_phases = extract_phase_data(cpu_result)
        cpu_gm = cpu_result.GM.values.item()
        
        # GPU calculation
        gpu_result = equilibrium(dbf, comps, phases, conditions,
                                gpu=True, verbose=False)
        gpu_phases = extract_phase_data(gpu_result)
        gpu_gm = gpu_result.GM.values.item()
        
        print("\nCPU RESULT:")
        print(f"  GM = {cpu_gm:.1f} J/mol")
        print("  Phases:")
        for p in cpu_phases:
            print(f"    {p['name']:12s}: {p['amount']:.4f} | X(AL)={p['composition']['AL']:.3f}, X(CU)={p['composition']['CU']:.3f}, X(FE)={p['composition']['FE']:.3f}")
        
        cpu_overall = check_mass_balance(cpu_phases, bulk_comp)
        print(f"  Overall composition from phases:")
        print(f"    X(AL)={cpu_overall['AL']:.6f} (target: {x_al:.6f}, error: {abs(cpu_overall['AL']-x_al):.6f})")
        print(f"    X(CU)={cpu_overall['CU']:.6f} (target: {x_cu:.6f}, error: {abs(cpu_overall['CU']-x_cu):.6f})")
        print(f"    X(FE)={cpu_overall['FE']:.6f} (target: {x_fe:.6f}, error: {abs(cpu_overall['FE']-x_fe):.6f})")
        
        cpu_mass_error = max(abs(cpu_overall['AL']-x_al), 
                            abs(cpu_overall['CU']-x_cu),
                            abs(cpu_overall['FE']-x_fe))
        
        print("\nGPU RESULT:")
        print(f"  GM = {gpu_gm:.1f} J/mol")
        print("  Phases:")
        for p in gpu_phases:
            print(f"    {p['name']:12s}: {p['amount']:.4f} | X(AL)={p['composition']['AL']:.3f}, X(CU)={p['composition']['CU']:.3f}, X(FE)={p['composition']['FE']:.3f}")
        
        gpu_overall = check_mass_balance(gpu_phases, bulk_comp)
        print(f"  Overall composition from phases:")
        print(f"    X(AL)={gpu_overall['AL']:.6f} (target: {x_al:.6f}, error: {abs(gpu_overall['AL']-x_al):.6f})")
        print(f"    X(CU)={gpu_overall['CU']:.6f} (target: {x_cu:.6f}, error: {abs(gpu_overall['CU']-x_cu):.6f})")
        print(f"    X(FE)={gpu_overall['FE']:.6f} (target: {x_fe:.6f}, error: {abs(gpu_overall['FE']-x_fe):.6f})")
        
        gpu_mass_error = max(abs(gpu_overall['AL']-x_al),
                            abs(gpu_overall['CU']-x_cu),
                            abs(gpu_overall['FE']-x_fe))
        
        print(f"\n  Energy difference: {abs(cpu_gm - gpu_gm):.1f} J/mol")
        print(f"  Max mass conservation error: CPU={cpu_mass_error:.6f}, GPU={gpu_mass_error:.6f}")
        
        if gpu_mass_error > 0.001:
            print("  ⚠️ GPU FAILS MASS CONSERVATION!")
        if cpu_mass_error > 0.001:
            print("  ⚠️ CPU FAILS MASS CONSERVATION!")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
If GPU is failing mass conservation, the issue is likely in:
1. The equilibrium matrix constraint rows for multi-component systems
2. The way prescribed mole fractions are enforced
3. Indexing issues when there are 3+ components

The constraint rows in the equilibrium matrix should enforce:
- sum(NP[i] * X[i,j]) = X_target[j] for each component j
- This needs proper handling for AL, CU, and FE simultaneously
""")

if __name__ == "__main__":
    main()