#!/usr/bin/env python
"""Analyze the 7 failing conditions without pdens."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def test_condition_and_neighbors(dbf, comps, phases, x_al, x_cu, temp, desc):
    """Test a condition and its neighbors to check for discontinuities."""
    
    print("\n" + "=" * 80)
    print(f"{desc}")
    print(f"Center: X(AL)={x_al:.2f}, X(CU)={x_cu:.2f}, X(FE)={1-x_al-x_cu:.2f}, T={temp}K")
    print("=" * 80)
    
    # Test center and nearby points
    deltas = [
        (-0.01, 0, "AL-0.01"),
        (0, -0.01, "CU-0.01"),
        (0, 0, "CENTER"),
        (0, +0.01, "CU+0.01"),
        (+0.01, 0, "AL+0.01"),
    ]
    
    results = []
    
    print("\nPoint      | X(AL) | X(CU) | CPU GM    | GPU GM    | Diff   | CPU Phases | GPU Phases")
    print("-----------|-------|-------|-----------|-----------|--------|------------|------------")
    
    for dal, dcu, label in deltas:
        x_al_test = x_al + dal
        x_cu_test = x_cu + dcu
        
        # Skip invalid compositions
        if x_al_test < 0 or x_cu_test < 0 or x_al_test + x_cu_test >= 1.0:
            continue
        
        conditions = {
            v.X('AL'): x_al_test,
            v.X('CU'): x_cu_test,
            v.T: temp,
            v.P: 101325
        }
        
        try:
            # CPU calculation
            cpu_result = equilibrium(dbf, comps, phases, conditions,
                                    gpu=False, verbose=False)
            cpu_gm = cpu_result.GM.values.item()
            cpu_phases = cpu_result.Phase.values.flatten()
            cpu_np = cpu_result.NP.values.flatten()
            
            # GPU calculation
            gpu_result = equilibrium(dbf, comps, phases, conditions,
                                    gpu=True, verbose=False)
            gpu_gm = gpu_result.GM.values.item()
            gpu_phases = gpu_result.Phase.values.flatten()
            gpu_np = gpu_result.NP.values.flatten()
            
            diff = abs(gpu_gm - cpu_gm)
            
            # Get active phases
            cpu_active = []
            gpu_active = []
            for i, (cp, gp, cn, gn) in enumerate(zip(cpu_phases, gpu_phases, cpu_np, gpu_np)):
                if not np.isnan(cn) and cn > 0.001:
                    cpu_active.append(f"{cp}({cn:.2f})")
                if not np.isnan(gn) and gn > 0.001:
                    gpu_active.append(f"{gp}({gn:.2f})")
            
            cpu_str = "+".join(cpu_active) if cpu_active else "NONE"
            gpu_str = "+".join(gpu_active) if gpu_active else "NONE"
            
            results.append({
                'label': label,
                'x_al': x_al_test,
                'x_cu': x_cu_test,
                'cpu_gm': cpu_gm,
                'gpu_gm': gpu_gm,
                'diff': diff,
                'cpu_phases': cpu_str,
                'gpu_phases': gpu_str,
                'cpu_np': cpu_np.copy(),
                'gpu_np': gpu_np.copy()
            })
            
            marker = "*" if diff > 100 else ""
            print(f"{label:10s} | {x_al_test:.2f}  | {x_cu_test:.2f}  | {cpu_gm:9.1f} | {gpu_gm:9.1f} | {diff:6.1f}{marker} | {cpu_str[:11]:11s} | {gpu_str[:11]:11s}")
            
        except Exception as e:
            print(f"{label:10s} | {x_al_test:.2f}  | {x_cu_test:.2f}  | ERROR: {str(e)[:40]}")
    
    # Check for discontinuities
    if len(results) >= 3:
        print("\nDiscontinuity Analysis:")
        center_idx = next((i for i, r in enumerate(results) if r['label'] == 'CENTER'), None)
        
        if center_idx is not None:
            center = results[center_idx]
            
            # Check phase amounts for jumps
            for i, r in enumerate(results):
                if r['label'] != 'CENTER':
                    # Compare phase amounts
                    cpu_jump = np.max(np.abs(r['cpu_np'] - center['cpu_np'][0:len(r['cpu_np'])]))
                    gpu_jump = np.max(np.abs(r['gpu_np'] - center['gpu_np'][0:len(r['gpu_np'])]))
                    
                    if cpu_jump > 0.2 and gpu_jump < 0.1:
                        print(f"  ⚠ CPU shows jump from {r['label']}: max phase change = {cpu_jump:.3f}")
                    elif gpu_jump > 0.2 and cpu_jump < 0.1:
                        print(f"  ⚠ GPU shows jump from {r['label']}: max phase change = {gpu_jump:.3f}")
            
            # Check if CPU and GPU have different phase assemblages at center
            if center['cpu_phases'] != center['gpu_phases']:
                print(f"  ⚠ Different phases at center: CPU={center['cpu_phases']}, GPU={center['gpu_phases']}")

def main():
    """Analyze all 7 failing conditions."""
    
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2', 'L12', 'ALCU_THETA', 'AL13FE4', 'AL5FE2']
    
    print("=" * 80)
    print("ANALYSIS OF 7 FAILING CONDITIONS (NO PDENS)")
    print("=" * 80)
    
    # The 7 failing conditions
    failures = [
        (0.40, 0.40, 600, "Failure 1: Equal Al-Cu at 600K (41 J/mol)"),
        (0.50, 0.40, 600, "Failure 2: High Al at 600K (20 J/mol)"),
        (0.60, 0.10, 600, "Failure 3: Very high Al at 600K (706 J/mol) ***"),
        (0.10, 0.50, 900, "Failure 4: High Cu at 900K (37 J/mol)"),
        (0.30, 0.50, 900, "Failure 5: High Cu at 900K (6 J/mol)"),
        (0.70, 0.20, 900, "Failure 6: Very high Al at 900K (30 J/mol)"),
        (0.70, 0.10, 1200, "Failure 7: Very high Al at 1200K (33 J/mol)")
    ]
    
    for x_al, x_cu, temp, desc in failures:
        test_condition_and_neighbors(dbf, comps, phases, x_al, x_cu, temp, desc)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Look for:
1. Discontinuous jumps in phase amounts between neighbors
2. Different phase assemblages between CPU and GPU at center
3. Large energy differences (> 100 J/mol)
""")

if __name__ == "__main__":
    main()