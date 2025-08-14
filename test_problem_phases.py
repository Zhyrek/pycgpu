#!/usr/bin/env python
"""Focus on the specific phase combination that causes divergence."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def main():
    dbf = Database('Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    
    # The problematic phase combination
    phases = ['FCC_A1', 'BCC_A2', 'BCC_B2', 'L12_FCC']
    
    print("=" * 80)
    print("TESTING PROBLEMATIC PHASE COMBINATION")
    print("=" * 80)
    print(f"Phases: {phases}")
    print()
    
    # Test multiple points around the failing condition
    test_points = [
        (0.39, 0.40, "Works"),
        (0.395, 0.40, "Test"),
        (0.398, 0.40, "Test"),
        (0.399, 0.40, "Test"),
        (0.40, 0.40, "FAILS"),
        (0.401, 0.40, "Test"),
        (0.402, 0.40, "Test"),
        (0.405, 0.40, "Test"),
        (0.41, 0.40, "Works"),
    ]
    
    results = []
    for x_al, x_cu, label in test_points:
        conditions = {
            v.X('AL'): x_al,
            v.X('CU'): x_cu,
            v.T: 600,
            v.P: 101325
        }
        
        # Run both CPU and GPU
        cpu_result = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
        gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
        
        cpu_gm = float(cpu_result.GM.values.squeeze())
        gpu_gm = float(gpu_result.GM.values.squeeze())
        diff = gpu_gm - cpu_gm
        
        # Get active phases
        cpu_active = []
        gpu_active = []
        for phase in phases:
            try:
                cpu_np = float(cpu_result.NP.sel(phase=phase).values.squeeze())
                if cpu_np > 1e-6:
                    cpu_active.append((phase, cpu_np))
            except:
                pass
            try:
                gpu_np = float(gpu_result.NP.sel(phase=phase).values.squeeze())
                if gpu_np > 1e-6:
                    gpu_active.append((phase, gpu_np))
            except:
                pass
        
        status = "MATCH" if abs(diff) < 1.0 else "***DIFF***"
        print(f"X(AL)={x_al:.3f}: CPU={cpu_gm:9.2f} GPU={gpu_gm:9.2f} Δ={diff:7.2f} [{status}] ({label})")
        
        if abs(diff) > 1.0:
            print(f"  CPU phases: {cpu_active}")
            print(f"  GPU phases: {gpu_active}")
        
        results.append((x_al, cpu_gm, gpu_gm, diff))
    
    # Analyze the pattern
    print("\n" + "=" * 80)
    print("PATTERN ANALYSIS")
    print("=" * 80)
    
    # Find where the jump occurs
    for i in range(1, len(results)):
        x_prev, cpu_prev, gpu_prev, _ = results[i-1]
        x_curr, cpu_curr, gpu_curr, _ = results[i]
        
        cpu_jump = abs(cpu_curr - cpu_prev)
        gpu_jump = abs(gpu_curr - gpu_prev)
        
        if cpu_jump > 100 or gpu_jump > 100:
            print(f"\nLarge jump between X(AL)={x_prev:.3f} and {x_curr:.3f}:")
            print(f"  CPU: {cpu_prev:.2f} -> {cpu_curr:.2f} (Δ={cpu_jump:.2f})")
            print(f"  GPU: {gpu_prev:.2f} -> {gpu_curr:.2f} (Δ={gpu_jump:.2f})")
    
    # Now test with verbose output for the exact failing point
    print("\n" + "=" * 80)
    print("VERBOSE OUTPUT FOR FAILING POINT")
    print("=" * 80)
    
    conditions = {
        v.X('AL'): 0.40,
        v.X('CU'): 0.40,
        v.T: 600,
        v.P: 101325
    }
    
    print("\nCPU VERBOSE:")
    print("-" * 60)
    cpu_result = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=True)
    
    print("\nGPU VERBOSE:")
    print("-" * 60)
    gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)

if __name__ == "__main__":
    main()