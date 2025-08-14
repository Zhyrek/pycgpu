#!/usr/bin/env python
"""Check which phases are stable in equilibrium and correlate with divergence."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def analyze_equilibrium(dbf, comps, phases, conditions, label=""):
    """Analyze equilibrium results for both CPU and GPU."""
    
    # CPU calculation
    cpu_result = equilibrium(dbf, comps, phases, conditions,
                            calc_opts={'pdens': 50},
                            gpu=False, verbose=False)
    cpu_gm = cpu_result.GM.values.item()
    
    # GPU calculation
    gpu_result = equilibrium(dbf, comps, phases, conditions,
                            calc_opts={'pdens': 50},
                            gpu=True, verbose=False)
    gpu_gm = gpu_result.GM.values.item()
    
    # Get phase amounts
    cpu_np = cpu_result.NP.values.flatten()
    gpu_np = gpu_result.NP.values.flatten()
    
    # Identify stable phases (amount > 0.001)
    cpu_stable = {}
    gpu_stable = {}
    
    for i, phase in enumerate(phases):
        if i < len(cpu_np) and cpu_np[i] > 0.001:
            cpu_stable[phase] = cpu_np[i]
        if i < len(gpu_np) and gpu_np[i] > 0.001:
            gpu_stable[phase] = gpu_np[i]
    
    diff = abs(gpu_gm - cpu_gm)
    
    return {
        'cpu_gm': cpu_gm,
        'gpu_gm': gpu_gm,
        'diff': diff,
        'cpu_stable': cpu_stable,
        'gpu_stable': gpu_stable,
        'cpu_np': cpu_np,
        'gpu_np': gpu_np
    }

def main():
    """Test phase stability correlation with divergence."""
    
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    all_phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2', 'L12', 'ALCU_THETA', 'AL13FE4', 'AL5FE2']
    
    print("=" * 100)
    print("PHASE STABILITY ANALYSIS - CORRELATION WITH DIVERGENCE")
    print("=" * 100)
    
    # Test various conditions
    test_conditions = [
        # Original failing condition
        (0.2, 0.5, 900, "High Cu, Low Al - FAILS"),
        # Working conditions from previous test
        (0.3, 0.3, 900, "Equal Al-Cu - WORKS"),
        (0.1, 0.1, 900, "Low Al-Cu - WORKS"),
        (0.4, 0.4, 900, "High Al-Cu - WORKS"),
        # Additional test points
        (0.1, 0.5, 900, "High Cu, Very Low Al"),
        (0.3, 0.5, 900, "High Cu, Med Al"),
        (0.2, 0.3, 900, "Med Cu, Low Al"),
        (0.2, 0.4, 900, "Med-High Cu, Low Al"),
        (0.2, 0.6, 900, "Very High Cu, Low Al"),
        (0.2, 0.7, 900, "Extreme Cu, Low Al"),
    ]
    
    print("\nTesting with all 8 phases:")
    print("-" * 100)
    print("X(AL) | X(CU) | X(FE) | CPU Stable Phases           | GPU Stable Phases           | Diff   | B2?")
    print("------|-------|-------|----------------------------|----------------------------|--------|----")
    
    for x_al, x_cu, temp, desc in test_conditions:
        x_fe = 1.0 - x_al - x_cu
        
        conditions = {
            v.X('AL'): x_al,
            v.X('CU'): x_cu,
            v.T: temp,
            v.P: 101325
        }
        
        result = analyze_equilibrium(dbf, comps, all_phases, conditions)
        
        # Format stable phases
        cpu_phases_str = ', '.join([f"{p}({amt:.3f})" for p, amt in result['cpu_stable'].items()])
        gpu_phases_str = ', '.join([f"{p}({amt:.3f})" for p, amt in result['gpu_stable'].items()])
        
        # Check if BCC_B2 is stable
        b2_in_cpu = 'BCC_B2' in result['cpu_stable']
        b2_in_gpu = 'BCC_B2' in result['gpu_stable']
        b2_status = "CPU" if b2_in_cpu and not b2_in_gpu else "GPU" if b2_in_gpu and not b2_in_cpu else "Both" if b2_in_cpu and b2_in_gpu else "No"
        
        status = "✓" if result['diff'] < 100 else "✗"
        
        print(f" {x_al:.2f}  | {x_cu:.2f}  | {x_fe:.2f}  | {cpu_phases_str:27s} | {gpu_phases_str:27s} | {result['diff']:6.1f} | {b2_status:4s} {status}")
    
    # Now test with phases excluding BCC_B2
    print("\n" + "=" * 100)
    print("Testing WITHOUT BCC_B2 (7 phases):")
    print("-" * 100)
    
    phases_no_b2 = [p for p in all_phases if p != 'BCC_B2']
    
    print("X(AL) | X(CU) | CPU GM    | GPU GM    | Diff    | Status")
    print("------|-------|-----------|-----------|---------|--------")
    
    for x_al, x_cu, temp, desc in test_conditions[:5]:  # Just test first 5
        conditions = {
            v.X('AL'): x_al,
            v.X('CU'): x_cu,
            v.T: temp,
            v.P: 101325
        }
        
        result = analyze_equilibrium(dbf, comps, phases_no_b2, conditions)
        
        status = "✓" if result['diff'] < 100 else "✗"
        print(f" {x_al:.2f}  | {x_cu:.2f}  | {result['cpu_gm']:9.1f} | {result['gpu_gm']:9.1f} | {result['diff']:7.1f} | {status}")
    
    print("\n" + "=" * 100)
    print("ANALYSIS:")
    print("-" * 100)
    print("Key observations:")
    print("1. Check if BCC_B2 presence in stable assemblage correlates with divergence")
    print("2. Check if divergence only occurs when BCC_B2 is actually stable")
    print("3. Check if CPU and GPU disagree on whether BCC_B2 should be stable")
    print("=" * 100)

if __name__ == "__main__":
    main()