#!/usr/bin/env python
"""Test compositions near the failing point to determine which is correct."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def test_composition(dbf, comps, phases, x_al, x_cu, t, label=""):
    """Test a single composition and return results."""
    conditions = {
        v.X('AL'): x_al,
        v.X('CU'): x_cu,
        v.T: t,
        v.P: 101325
    }
    
    # Run CPU
    cpu_result = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
    cpu_gm = float(cpu_result.GM.values.squeeze())
    
    # Run GPU
    gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    gpu_gm = float(gpu_result.GM.values.squeeze())
    
    diff = gpu_gm - cpu_gm
    status = "MATCH" if abs(diff) < 1.0 else "DIFF"
    
    print(f"{label:15s} X(AL)={x_al:.2f} X(CU)={x_cu:.2f} X(FE)={1-x_al-x_cu:.2f} | "
          f"CPU={cpu_gm:8.1f} GPU={gpu_gm:8.1f} Δ={diff:7.2f} | {status}")
    
    return cpu_gm, gpu_gm, diff

def main():
    dbf = Database('Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2', 'L12_FCC', 
              'ALCU_THETA', 'AL13FE4_D03', 'AL5FE2_D82']
    
    print("=" * 80)
    print("TESTING COMPOSITIONS NEAR FAILING POINT")
    print("=" * 80)
    print("All tests at T=600K")
    print()
    
    # Store results
    results = []
    
    # Test the failing point
    print("ORIGINAL FAILING POINT:")
    cpu_fail, gpu_fail, diff_fail = test_composition(dbf, comps, phases, 0.40, 0.40, 600, "CENTER")
    results.append(('CENTER', 0.40, 0.40, cpu_fail, gpu_fail, diff_fail))
    print()
    
    # Test nearby points in a grid
    print("NEARBY COMPOSITIONS:")
    deltas = [-0.02, -0.01, 0.01, 0.02]
    
    for dal in deltas:
        for dcu in deltas:
            x_al = 0.40 + dal
            x_cu = 0.40 + dcu
            x_fe = 1 - x_al - x_cu
            
            # Skip if composition is invalid
            if x_al < 0 or x_cu < 0 or x_fe < 0 or x_al > 1 or x_cu > 1 or x_fe > 1:
                continue
            
            label = f"Δ({dal:+.2f},{dcu:+.2f})"
            cpu_gm, gpu_gm, diff = test_composition(dbf, comps, phases, x_al, x_cu, 600, label)
            results.append((label, x_al, x_cu, cpu_gm, gpu_gm, diff))
    
    # Also test along lines through the failing point
    print("\nALONG X(AL) = 0.40 LINE:")
    for x_cu in [0.35, 0.36, 0.37, 0.38, 0.39, 0.41, 0.42, 0.43, 0.44, 0.45]:
        x_al = 0.40
        x_fe = 1 - x_al - x_cu
        if x_fe < 0 or x_fe > 1:
            continue
        label = f"AL=0.40,CU={x_cu:.2f}"
        cpu_gm, gpu_gm, diff = test_composition(dbf, comps, phases, x_al, x_cu, 600, label)
        results.append((label, x_al, x_cu, cpu_gm, gpu_gm, diff))
    
    print("\nALONG X(CU) = 0.40 LINE:")
    for x_al in [0.35, 0.36, 0.37, 0.38, 0.39, 0.41, 0.42, 0.43, 0.44, 0.45]:
        x_cu = 0.40
        x_fe = 1 - x_al - x_cu
        if x_fe < 0 or x_fe > 1:
            continue
        label = f"AL={x_al:.2f},CU=0.40"
        cpu_gm, gpu_gm, diff = test_composition(dbf, comps, phases, x_al, x_cu, 600, label)
        results.append((label, x_al, x_cu, cpu_gm, gpu_gm, diff))
    
    # Analyze results
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    # Find matching results (where CPU and GPU agree)
    matching = [(label, x_al, x_cu, cpu_gm, gpu_gm) 
                for label, x_al, x_cu, cpu_gm, gpu_gm, diff in results 
                if abs(diff) < 1.0]
    
    if matching:
        print(f"\nFound {len(matching)} matching points out of {len(results)} total")
        
        # Calculate average GM for matching points
        avg_gm_matching = np.mean([cpu_gm for _, _, _, cpu_gm, _ in matching])
        print(f"Average GM for matching points: {avg_gm_matching:.1f} J/mol")
        
        # Check which failing result is closer to the matching average
        print(f"\nFailing point results:")
        print(f"  CPU: {cpu_fail:.1f} J/mol (Δ from avg = {cpu_fail - avg_gm_matching:.1f})")
        print(f"  GPU: {gpu_fail:.1f} J/mol (Δ from avg = {gpu_fail - avg_gm_matching:.1f})")
        
        # Also look at the gradient/smoothness
        print("\nGradient analysis (GM values along lines):")
        
        # Extract values along AL=0.40 line
        al_line = [(x_cu, cpu_gm, gpu_gm) for label, x_al, x_cu, cpu_gm, gpu_gm, diff in results 
                   if abs(x_al - 0.40) < 0.001 and label != 'CENTER']
        al_line.sort()
        
        if len(al_line) > 1:
            print("\n  Along X(AL)=0.40:")
            for x_cu, cpu_gm, gpu_gm in al_line:
                print(f"    X(CU)={x_cu:.2f}: CPU={cpu_gm:8.1f} GPU={gpu_gm:8.1f}")
            
            # Check smoothness by looking at second derivatives
            if len(al_line) >= 3:
                # Approximate second derivative for CPU
                cpu_gms = [gm for _, gm, _ in al_line]
                cpu_second_deriv = []
                for i in range(1, len(cpu_gms)-1):
                    d2g = cpu_gms[i+1] - 2*cpu_gms[i] + cpu_gms[i-1]
                    cpu_second_deriv.append(d2g)
                
                # Approximate second derivative for GPU  
                gpu_gms = [gm for _, _, gm in al_line]
                gpu_second_deriv = []
                for i in range(1, len(gpu_gms)-1):
                    d2g = gpu_gms[i+1] - 2*gpu_gms[i] + gpu_gms[i-1]
                    gpu_second_deriv.append(d2g)
                
                print(f"\n  Curvature (d²G/dx²) along X(AL)=0.40:")
                print(f"    CPU: max={max(cpu_second_deriv):.1f}, min={min(cpu_second_deriv):.1f}")
                print(f"    GPU: max={max(gpu_second_deriv):.1f}, min={min(gpu_second_deriv):.1f}")
        
        # Extract values along CU=0.40 line
        cu_line = [(x_al, cpu_gm, gpu_gm) for label, x_al, x_cu, cpu_gm, gpu_gm, diff in results 
                   if abs(x_cu - 0.40) < 0.001 and label != 'CENTER']
        cu_line.sort()
        
        if len(cu_line) > 1:
            print("\n  Along X(CU)=0.40:")
            for x_al, cpu_gm, gpu_gm in cu_line:
                print(f"    X(AL)={x_al:.2f}: CPU={cpu_gm:8.1f} GPU={gpu_gm:8.1f}")
            
            # Check smoothness
            if len(cu_line) >= 3:
                # Approximate second derivative for CPU
                cpu_gms = [gm for _, gm, _ in cu_line]
                cpu_second_deriv = []
                for i in range(1, len(cpu_gms)-1):
                    d2g = cpu_gms[i+1] - 2*cpu_gms[i] + cpu_gms[i-1]
                    cpu_second_deriv.append(d2g)
                
                # Approximate second derivative for GPU
                gpu_gms = [gm for _, _, gm in cu_line]
                gpu_second_deriv = []
                for i in range(1, len(gpu_gms)-1):
                    d2g = gpu_gms[i+1] - 2*gpu_gms[i] + gpu_gms[i-1]
                    gpu_second_deriv.append(d2g)
                
                print(f"\n  Curvature (d²G/dx²) along X(CU)=0.40:")
                print(f"    CPU: max={max(cpu_second_deriv):.1f}, min={min(cpu_second_deriv):.1f}")
                print(f"    GPU: max={max(gpu_second_deriv):.1f}, min={min(gpu_second_deriv):.1f}")
        
        # Determine which is likely correct
        print("\n" + "=" * 80)
        print("CONCLUSION")
        print("=" * 80)
        
        if abs(cpu_fail - avg_gm_matching) < abs(gpu_fail - avg_gm_matching):
            print("CPU result appears MORE CONSISTENT with neighboring points")
            print(f"CPU is {abs(gpu_fail - cpu_fail):.1f} J/mol LOWER (more stable)")
        else:
            print("GPU result appears MORE CONSISTENT with neighboring points")
            print(f"GPU is {abs(gpu_fail - cpu_fail):.1f} J/mol HIGHER (less stable)")
        
        # Check for discontinuities
        min_neighbor_gm = min([cpu_gm for label, _, _, cpu_gm, _, _ in results if label != 'CENTER'])
        if cpu_fail < min_neighbor_gm - 100:
            print("\n⚠ CPU result seems anomalously low compared to all neighbors")
        if gpu_fail < min_neighbor_gm - 100:
            print("\n⚠ GPU result seems anomalously low compared to all neighbors")
            
    else:
        print("No matching points found - CPU and GPU disagree everywhere!")

if __name__ == "__main__":
    main()