#!/usr/bin/env python
"""Investigate all failing conditions for patterns."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def test_condition(dbf, comps, phases, x_al, x_cu, temp, label=""):
    """Test a single condition and return results."""
    conditions = {
        v.X('AL'): x_al,
        v.X('CU'): x_cu,
        v.T: temp,
        v.P: 101325
    }
    
    try:
        # CPU calculation
        cpu_result = equilibrium(dbf, comps, phases, conditions,
                                calc_opts={'pdens': 50},
                                gpu=False, verbose=False)
        cpu_gm = cpu_result.GM.values.item()
        cpu_np = cpu_result.NP.values.flatten()
        
        # GPU calculation
        gpu_result = equilibrium(dbf, comps, phases, conditions,
                                calc_opts={'pdens': 50},
                                gpu=True, verbose=False)
        gpu_gm = gpu_result.GM.values.item()
        gpu_np = gpu_result.NP.values.flatten()
        
        diff = abs(gpu_gm - cpu_gm)
        
        # Get active phases and compositions
        cpu_active = []
        gpu_active = []
        
        for i, phase in enumerate(phases):
            if i < len(cpu_np) and cpu_np[i] > 0.001:
                cpu_active.append(f"{phase}({cpu_np[i]:.3f})")
            if i < len(gpu_np) and gpu_np[i] > 0.001:
                gpu_active.append(f"{phase}({gpu_np[i]:.3f})")
        
        return {
            'cpu_gm': cpu_gm,
            'gpu_gm': gpu_gm,
            'diff': diff,
            'cpu_active': cpu_active,
            'gpu_active': gpu_active,
            'cpu_np': cpu_np,
            'gpu_np': gpu_np,
            'status': 'PASS' if diff < 100 else 'FAIL'
        }
    except Exception as e:
        return {
            'error': str(e),
            'status': 'ERROR'
        }

def investigate_failure_point(dbf, comps, phases, x_al_center, x_cu_center, temp, point_name):
    """Investigate a failure point and its neighbors."""
    
    print("\n" + "=" * 80)
    print(f"INVESTIGATING: {point_name}")
    print(f"Center point: X(AL)={x_al_center:.2f}, X(CU)={x_cu_center:.2f}, T={temp}K")
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
    
    print("\nTesting neighborhood:")
    print("-" * 80)
    print("Point      | X(AL) | X(CU) | X(FE) | CPU GM    | GPU GM    | Diff   | Status")
    print("-----------|-------|-------|-------|-----------|-----------|--------|-------")
    
    for dal, dcu, label in deltas:
        x_al = x_al_center + dal
        x_cu = x_cu_center + dcu
        
        # Skip invalid compositions
        if x_al < 0 or x_cu < 0 or x_al + x_cu >= 1.0:
            continue
            
        x_fe = 1.0 - x_al - x_cu
        
        result = test_condition(dbf, comps, phases, x_al, x_cu, temp, label)
        results.append((label, x_al, x_cu, result))
        
        if 'error' not in result:
            print(f"{label:10s} | {x_al:.2f}  | {x_cu:.2f}  | {x_fe:.2f}  | {result['cpu_gm']:9.1f} | {result['gpu_gm']:9.1f} | {result['diff']:6.1f} | {result['status']}")
    
    # Check for discontinuities
    print("\nPhase stability analysis:")
    print("-" * 80)
    
    center_idx = -1
    for i, (label, _, _, _) in enumerate(results):
        if label == "CENTER":
            center_idx = i
            break
    
    if center_idx >= 0 and 'error' not in results[center_idx][3]:
        center_result = results[center_idx][3]
        print(f"CENTER - CPU: {', '.join(center_result['cpu_active'])}")
        print(f"         GPU: {', '.join(center_result['gpu_active'])}")
        
        # Check neighbors for discontinuities
        discontinuous = False
        for label, x_al, x_cu, result in results:
            if label != "CENTER" and 'error' not in result:
                # Check if phase amounts change dramatically
                if result['status'] == 'PASS' and center_result['status'] == 'FAIL':
                    print(f"\n{label} - CPU: {', '.join(result['cpu_active'])}")
                    print(f"        GPU: {', '.join(result['gpu_active'])}")
                    
                    # Check for large jumps in phase amounts
                    for i in range(min(len(result['cpu_np']), len(center_result['cpu_np']))):
                        cpu_jump = abs(result['cpu_np'][i] - center_result['cpu_np'][i])
                        gpu_jump = abs(result['gpu_np'][i] - center_result['gpu_np'][i])
                        
                        if cpu_jump > 0.1 and gpu_jump < 0.05:
                            discontinuous = True
                            print(f"  ⚠ CPU shows discontinuous jump in phase {i}: {cpu_jump:.3f}")
                        elif gpu_jump > 0.1 and cpu_jump < 0.05:
                            discontinuous = True
                            print(f"  ⚠ GPU shows discontinuous jump in phase {i}: {gpu_jump:.3f}")
        
        if discontinuous:
            print("\n⚠ DISCONTINUOUS BEHAVIOR DETECTED")
        else:
            print("\n✓ Phase amounts vary smoothly")
    
    return results

def main():
    """Investigate all failing conditions."""
    
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2', 'L12', 'ALCU_THETA', 'AL13FE4', 'AL5FE2']
    
    print("=" * 80)
    print("COMPREHENSIVE FAILURE INVESTIGATION")
    print("=" * 80)
    
    # Known failing conditions from the test results
    failing_conditions = [
        (0.10, 0.50, 900, "Low Al, High Cu"),
        (0.20, 0.50, 900, "Med Al, High Cu (known singular)"),
        (0.30, 0.50, 900, "High Al, High Cu"),
        (0.40, 0.40, 600, "Equal Al-Cu at 600K"),
        (0.50, 0.40, 600, "High Al at 600K"),
        (0.70, 0.20, 900, "Very High Al"),
        (0.20, 0.40, 1200, "High temp case 1"),
        (0.30, 0.60, 1200, "High temp case 2"),
        (0.70, 0.10, 1200, "High temp case 3"),
    ]
    
    # First, quick overview
    print("\nQuick overview of all failures:")
    print("-" * 80)
    print("X(AL) | X(CU) | T(K) | Diff (J/mol) | Description")
    print("------|-------|------|--------------|-------------")
    
    for x_al, x_cu, temp, desc in failing_conditions:
        result = test_condition(dbf, comps, phases, x_al, x_cu, temp)
        if 'error' not in result:
            print(f"{x_al:.2f}  | {x_cu:.2f}  | {temp:4d} | {result['diff']:12.1f} | {desc}")
    
    # Now investigate each in detail
    print("\n" + "=" * 80)
    print("DETAILED INVESTIGATION OF EACH FAILURE")
    print("=" * 80)
    
    # Group by temperature
    for temp in [600, 900, 1200]:
        print(f"\n{'='*80}")
        print(f"TEMPERATURE: {temp}K")
        print(f"{'='*80}")
        
        temp_failures = [(x_al, x_cu, t, desc) for x_al, x_cu, t, desc in failing_conditions if t == temp]
        
        for x_al, x_cu, _, desc in temp_failures:
            investigate_failure_point(dbf, comps, phases, x_al, x_cu, temp, desc)
    
    print("\n" + "=" * 80)
    print("SUMMARY OF FINDINGS")
    print("=" * 80)
    
    print("""
Patterns to look for:
1. Do failures occur at specific composition ratios (e.g., X(CU)=0.5)?
2. Are there discontinuous jumps in phase amounts?
3. Does one solver consistently find lower energy states?
4. Do neighboring points show smooth transitions?
""")

if __name__ == "__main__":
    main()