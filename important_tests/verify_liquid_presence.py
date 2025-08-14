#!/usr/bin/env python
"""Carefully verify LIQUID phase presence and composition."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def analyze_phases_carefully(result, phases, label):
    """Carefully analyze which phases are present and their compositions."""
    print(f"\n{label} DETAILED ANALYSIS:")
    print("-" * 70)
    
    # Get phase amounts
    np_vals = result.NP.values.flatten()
    
    # Get GM
    gm = result.GM.values.item()
    print(f"Gibbs energy: {gm:.2f} J/mol")
    
    # Check each phase
    print("\nPhases:")
    liquid_found = False
    liquid_data = {}
    
    for i, phase in enumerate(phases):
        if i < len(np_vals) and not np.isnan(np_vals[i]):
            amount = np_vals[i]
            if amount > 0.001:  # Phase is present
                print(f"  {phase}: {amount:.6f} (active)")
                
                if phase == 'LIQUID':
                    liquid_found = True
                    liquid_data['amount'] = amount
                    
                    # Try to extract composition
                    try:
                        x_vals = result.X.values
                        # Get component values for this phase
                        for comp_idx, comp in enumerate(['AL', 'CU', 'FE']):
                            # Use xarray selection
                            comp_val = result.X.sel(component=comp).values.flatten()[i]
                            if not np.isnan(comp_val):
                                liquid_data[f'X_{comp}'] = comp_val
                    except Exception as e:
                        print(f"    ERROR extracting composition: {e}")
            else:
                print(f"  {phase}: {amount:.6f} (inactive, < 0.001)")
        else:
            print(f"  {phase}: not present or NaN")
    
    if liquid_found:
        print(f"\nLIQUID phase composition:")
        if 'X_AL' in liquid_data:
            print(f"  X(AL) = {liquid_data.get('X_AL', 'ERROR'):.6f}")
        if 'X_CU' in liquid_data:
            print(f"  X(CU) = {liquid_data.get('X_CU', 'ERROR'):.6f}")
        if 'X_FE' in liquid_data:
            print(f"  X(FE) = {liquid_data.get('X_FE', 'ERROR'):.6f}")
        
        # Verify sum
        if all(k in liquid_data for k in ['X_AL', 'X_CU', 'X_FE']):
            total = liquid_data['X_AL'] + liquid_data['X_CU'] + liquid_data['X_FE']
            print(f"  Sum = {total:.6f} (should be ~1.0)")
    else:
        print("\n⚠ NO LIQUID PHASE FOUND (amount < 0.001)")
    
    return liquid_found, liquid_data

def main():
    """Verify LIQUID presence at key conditions."""
    
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2']
    
    print("=" * 80)
    print("VERIFICATION OF LIQUID PHASE PRESENCE AND COMPOSITION")
    print("=" * 80)
    
    # Focus on the critical X(CU)=0.50 cases
    critical_conditions = [
        (0.20, 0.49, 900, "BEFORE: X(AL)=0.20, X(CU)=0.49"),
        (0.20, 0.50, 900, "AT SINGULARITY: X(AL)=0.20, X(CU)=0.50"),
        (0.20, 0.51, 900, "AFTER: X(AL)=0.20, X(CU)=0.51"),
    ]
    
    for x_al, x_cu, temp, desc in critical_conditions:
        x_fe = 1.0 - x_al - x_cu
        
        print("\n" + "=" * 80)
        print(desc)
        print(f"Bulk: X(AL)={x_al:.2f}, X(CU)={x_cu:.2f}, X(FE)={x_fe:.2f}, T={temp}K")
        print("=" * 80)
        
        conditions = {
            v.X('AL'): x_al,
            v.X('CU'): x_cu,
            v.T: temp,
            v.P: 101325
        }
        
        # CPU calculation
        cpu_result = equilibrium(dbf, comps, phases, conditions,
                                calc_opts={'pdens': 50},
                                gpu=False, verbose=False)
        
        cpu_liquid_found, cpu_liquid_data = analyze_phases_carefully(cpu_result, phases, "CPU")
        
        # GPU calculation
        gpu_result = equilibrium(dbf, comps, phases, conditions,
                                calc_opts={'pdens': 50},
                                gpu=True, verbose=False)
        
        gpu_liquid_found, gpu_liquid_data = analyze_phases_carefully(gpu_result, phases, "GPU")
        
        # Compare
        print("\n" + "-" * 70)
        print("COMPARISON:")
        
        if cpu_liquid_found and gpu_liquid_found:
            cpu_cu = cpu_liquid_data.get('X_CU', 'ERROR')
            gpu_cu = gpu_liquid_data.get('X_CU', 'ERROR')
            
            if isinstance(cpu_cu, float) and isinstance(gpu_cu, float):
                print(f"LIQUID copper content:")
                print(f"  CPU: X(CU) = {cpu_cu:.6f}")
                print(f"  GPU: X(CU) = {gpu_cu:.6f}")
                print(f"  Difference: {abs(cpu_cu - gpu_cu):.6f}")
                
                if cpu_cu < 0.001 and gpu_cu > 0.01:
                    print("  ⚠ CPU shows near-zero copper while GPU shows significant copper")
                elif gpu_cu < 0.001 and cpu_cu > 0.01:
                    print("  ⚠ GPU shows near-zero copper while CPU shows significant copper")
            else:
                print("ERROR: Could not extract copper values")
        elif cpu_liquid_found and not gpu_liquid_found:
            print("⚠ CPU has LIQUID phase but GPU does not")
        elif gpu_liquid_found and not cpu_liquid_found:
            print("⚠ GPU has LIQUID phase but CPU does not")
        else:
            print("Neither CPU nor GPU has LIQUID phase")
        
        # GM comparison
        cpu_gm = cpu_result.GM.values.item()
        gpu_gm = gpu_result.GM.values.item()
        print(f"\nGibbs energy difference: {abs(cpu_gm - gpu_gm):.2f} J/mol")
        if gpu_gm < cpu_gm:
            print(f"  GPU finds lower energy (more stable)")
        elif cpu_gm < gpu_gm:
            print(f"  CPU finds lower energy (more stable)")
    
    # Now check the other X(CU)=0.50 cases
    print("\n" + "=" * 80)
    print("OTHER X(CU)=0.50 CASES")
    print("=" * 80)
    
    other_cases = [
        (0.10, 0.50, 900, "X(AL)=0.10, X(CU)=0.50"),
        (0.30, 0.50, 900, "X(AL)=0.30, X(CU)=0.50"),
    ]
    
    for x_al, x_cu, temp, desc in other_cases:
        print(f"\n{desc}:")
        
        conditions = {
            v.X('AL'): x_al,
            v.X('CU'): x_cu,
            v.T: temp,
            v.P: 101325
        }
        
        cpu_result = equilibrium(dbf, comps, phases, conditions,
                                calc_opts={'pdens': 50},
                                gpu=False, verbose=False)
        gpu_result = equilibrium(dbf, comps, phases, conditions,
                                calc_opts={'pdens': 50},
                                gpu=True, verbose=False)
        
        # Check LIQUID presence
        cpu_np = cpu_result.NP.values.flatten()
        gpu_np = gpu_result.NP.values.flatten()
        
        liquid_idx = phases.index('LIQUID')
        cpu_liquid_amt = cpu_np[liquid_idx] if liquid_idx < len(cpu_np) else 0
        gpu_liquid_amt = gpu_np[liquid_idx] if liquid_idx < len(gpu_np) else 0
        
        print(f"  CPU LIQUID amount: {cpu_liquid_amt:.6f}")
        print(f"  GPU LIQUID amount: {gpu_liquid_amt:.6f}")
        
        if cpu_liquid_amt > 0.001 or gpu_liquid_amt > 0.001:
            print("  → LIQUID phase is present, need to check composition carefully")
        else:
            print("  → No significant LIQUID phase")
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
Key questions answered:
1. Is LIQUID phase actually present at X(CU)=0.50?
2. If present, does it really have zero copper?
3. Or is this an extraction error?
""")

if __name__ == "__main__":
    main()