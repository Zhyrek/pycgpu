#!/usr/bin/env python
"""Check phase compositions for failing conditions."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def get_liquid_compositions(result, phases):
    """Extract LIQUID phase composition if present."""
    np_vals = result.NP.values.flatten()
    x_vals = result.X.values
    
    # Find LIQUID phase index
    liquid_idx = -1
    for i, phase in enumerate(phases):
        if phase == 'LIQUID' and i < len(np_vals) and np_vals[i] > 0.001:
            liquid_idx = i
            break
    
    if liquid_idx >= 0:
        # Extract composition
        try:
            x_flat = x_vals.flatten()
            n_comps = 3  # AL, CU, FE
            comps = {}
            for c_idx, comp in enumerate(['AL', 'CU', 'FE']):
                idx = liquid_idx * n_comps + c_idx
                if idx < len(x_flat):
                    comps[comp] = x_flat[idx]
            return comps
        except:
            return None
    return None

def main():
    """Check compositions at failure points."""
    
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2', 'L12', 'ALCU_THETA', 'AL13FE4', 'AL5FE2']
    
    print("=" * 80)
    print("LIQUID PHASE COMPOSITION ANALYSIS AT FAILURE POINTS")
    print("=" * 80)
    
    # Focus on X(CU)=0.50 failures and neighbors
    test_cases = [
        # Group 1: X(CU) = 0.50 at 900K
        (0.10, 0.49, 900, "Low Al, CU=0.49"),
        (0.10, 0.50, 900, "Low Al, CU=0.50 *"),
        (0.10, 0.51, 900, "Low Al, CU=0.51"),
        
        (0.20, 0.49, 900, "Med Al, CU=0.49"),
        (0.20, 0.50, 900, "Med Al, CU=0.50 ***"),
        (0.20, 0.51, 900, "Med Al, CU=0.51"),
        
        (0.30, 0.49, 900, "High Al, CU=0.49"),
        (0.30, 0.50, 900, "High Al, CU=0.50 *"),
        (0.30, 0.51, 900, "High Al, CU=0.51"),
        
        # Group 2: X(CU) = 0.40 variations
        (0.39, 0.40, 600, "600K, CU=0.40"),
        (0.40, 0.40, 600, "600K, Equal *"),
        (0.41, 0.40, 600, "600K, CU=0.40"),
        
        (0.49, 0.40, 600, "600K, High Al"),
        (0.50, 0.40, 600, "600K, Very High Al *"),
        (0.51, 0.40, 600, "600K, Very High Al"),
    ]
    
    print("\n(* = known failure point)")
    
    for group_start in [0, 3, 6, 9, 12]:
        if group_start == 0:
            print("\n" + "=" * 80)
            print("GROUP: X(AL)=0.10, varying X(CU) around 0.50, T=900K")
        elif group_start == 3:
            print("\n" + "=" * 80)
            print("GROUP: X(AL)=0.20, varying X(CU) around 0.50, T=900K")
        elif group_start == 6:
            print("\n" + "=" * 80)
            print("GROUP: X(AL)=0.30, varying X(CU) around 0.50, T=900K")
        elif group_start == 9:
            print("\n" + "=" * 80)
            print("GROUP: Equal Al-Cu around 0.40, T=600K")
        else:
            print("\n" + "=" * 80)
            print("GROUP: High Al around 0.50, X(CU)=0.40, T=600K")
        
        print("-" * 80)
        print("Condition          | CPU LIQUID X(CU) | GPU LIQUID X(CU) | Diff  | GM Diff")
        print("-------------------|------------------|------------------|-------|--------")
        
        for i in range(group_start, min(group_start + 3, len(test_cases))):
            if i >= len(test_cases):
                break
                
            x_al, x_cu, temp, desc = test_cases[i]
            
            conditions = {
                v.X('AL'): x_al,
                v.X('CU'): x_cu,
                v.T: temp,
                v.P: 101325
            }
            
            # Calculate
            cpu_result = equilibrium(dbf, comps, phases, conditions,
                                    calc_opts={'pdens': 50},
                                    gpu=False, verbose=False)
            gpu_result = equilibrium(dbf, comps, phases, conditions,
                                    calc_opts={'pdens': 50},
                                    gpu=True, verbose=False)
            
            cpu_gm = cpu_result.GM.values.item()
            gpu_gm = gpu_result.GM.values.item()
            gm_diff = abs(gpu_gm - cpu_gm)
            
            # Get LIQUID compositions
            cpu_liquid = get_liquid_compositions(cpu_result, phases)
            gpu_liquid = get_liquid_compositions(gpu_result, phases)
            
            if cpu_liquid and gpu_liquid:
                cpu_cu = cpu_liquid.get('CU', 0.0)
                gpu_cu = gpu_liquid.get('CU', 0.0)
                cu_diff = abs(gpu_cu - cpu_cu)
                
                marker = "***" if gm_diff > 100 else "*" if gm_diff > 30 else ""
                
                print(f"{desc:18s} | {cpu_cu:16.4f} | {gpu_cu:16.4f} | {cu_diff:5.3f} | {gm_diff:6.1f} {marker}")
            else:
                print(f"{desc:18s} | No LIQUID phase   | No LIQUID phase   |   -   | {gm_diff:6.1f}")
        
        # Check for discontinuities
        if group_start < 9:  # For the X(CU)=0.50 groups
            print("\nContinuity check:")
            
            # Get the three values for this group
            vals = []
            for i in range(group_start, min(group_start + 3, len(test_cases))):
                x_al, x_cu, temp, desc = test_cases[i]
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
                
                cpu_liquid = get_liquid_compositions(cpu_result, phases)
                gpu_liquid = get_liquid_compositions(gpu_result, phases)
                
                if cpu_liquid and gpu_liquid:
                    vals.append((cpu_liquid.get('CU', 0.0), gpu_liquid.get('CU', 0.0)))
            
            if len(vals) == 3:
                # Check CPU continuity
                cpu_vals = [v[0] for v in vals]
                cpu_jump = abs(cpu_vals[1] - cpu_vals[0]) + abs(cpu_vals[2] - cpu_vals[1])
                cpu_smooth = abs(cpu_vals[2] - cpu_vals[0])
                
                # Check GPU continuity  
                gpu_vals = [v[1] for v in vals]
                gpu_jump = abs(gpu_vals[1] - gpu_vals[0]) + abs(gpu_vals[2] - gpu_vals[1])
                gpu_smooth = abs(gpu_vals[2] - gpu_vals[0])
                
                if cpu_jump > 2 * cpu_smooth and cpu_vals[1] < 0.01:
                    print(f"  ⚠ CPU shows discontinuity: {cpu_vals[0]:.3f} → {cpu_vals[1]:.3f} → {cpu_vals[2]:.3f}")
                    print(f"  ✓ GPU is continuous:      {gpu_vals[0]:.3f} → {gpu_vals[1]:.3f} → {gpu_vals[2]:.3f}")
                elif gpu_jump > 2 * gpu_smooth:
                    print(f"  ✓ CPU is continuous:      {cpu_vals[0]:.3f} → {cpu_vals[1]:.3f} → {cpu_vals[2]:.3f}")
                    print(f"  ⚠ GPU shows discontinuity: {gpu_vals[0]:.3f} → {gpu_vals[1]:.3f} → {gpu_vals[2]:.3f}")
                else:
                    print(f"  Both show smooth variation")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Key observations:
1. Major divergence (180 J/mol) at X(AL)=0.20, X(CU)=0.50
2. Minor divergences (< 40 J/mol) at other X(CU)=0.50 points
3. Look for discontinuous copper content in LIQUID phase
4. GPU typically finds lower energy states (more negative GM)
""")

if __name__ == "__main__":
    main()