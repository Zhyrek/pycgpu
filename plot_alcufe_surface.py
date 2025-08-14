#!/usr/bin/env python
"""Plot the energy surface to visualize the discontinuity."""

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
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2', 'L12_FCC', 
              'ALCU_THETA', 'AL13FE4_D03', 'AL5FE2_D82']
    
    print("=" * 80)
    print("ENERGY SURFACE ALONG CRITICAL LINES")
    print("=" * 80)
    print()
    
    # Test along X(AL)=0.40 line with finer resolution
    print("Along X(AL) = 0.40 line (finer resolution):")
    print("-" * 60)
    print("X(CU)    CPU_GM     GPU_GM     Diff    Status")
    print("-" * 60)
    
    x_cu_values = np.linspace(0.35, 0.45, 21)
    cpu_gms = []
    gpu_gms = []
    
    for x_cu in x_cu_values:
        x_al = 0.40
        x_fe = 1 - x_al - x_cu
        
        conditions = {
            v.X('AL'): x_al,
            v.X('CU'): x_cu,
            v.T: 600,
            v.P: 101325
        }
        
        # Run CPU
        cpu_result = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
        cpu_gm = float(cpu_result.GM.values.squeeze())
        cpu_gms.append(cpu_gm)
        
        # Run GPU
        gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
        gpu_gm = float(gpu_result.GM.values.squeeze())
        gpu_gms.append(gpu_gm)
        
        diff = gpu_gm - cpu_gm
        status = "MATCH" if abs(diff) < 1.0 else "***DIFF***"
        
        print(f"{x_cu:.3f}  {cpu_gm:9.1f}  {gpu_gm:9.1f}  {diff:7.2f}  {status}")
    
    # Calculate derivatives
    print("\n" + "=" * 80)
    print("DERIVATIVE ANALYSIS")
    print("=" * 80)
    
    dx = x_cu_values[1] - x_cu_values[0]
    
    # First derivatives (gradient)
    cpu_grad = np.gradient(cpu_gms, dx)
    gpu_grad = np.gradient(gpu_gms, dx)
    
    # Second derivatives (curvature)
    cpu_curv = np.gradient(cpu_grad, dx)
    gpu_curv = np.gradient(gpu_grad, dx)
    
    print("\nAt critical point X(CU)=0.40:")
    idx_040 = np.argmin(np.abs(x_cu_values - 0.40))
    print(f"  CPU: GM={cpu_gms[idx_040]:.1f}, dG/dx={cpu_grad[idx_040]:.1f}, d²G/dx²={cpu_curv[idx_040]:.1f}")
    print(f"  GPU: GM={gpu_gms[idx_040]:.1f}, dG/dx={gpu_grad[idx_040]:.1f}, d²G/dx²={gpu_curv[idx_040]:.1f}")
    
    # Find where derivatives are most different
    grad_diff = np.abs(cpu_grad - gpu_grad)
    curv_diff = np.abs(cpu_curv - gpu_curv)
    
    max_grad_idx = np.argmax(grad_diff)
    max_curv_idx = np.argmax(curv_diff)
    
    print(f"\nMaximum gradient difference at X(CU)={x_cu_values[max_grad_idx]:.3f}:")
    print(f"  CPU gradient: {cpu_grad[max_grad_idx]:.1f}")
    print(f"  GPU gradient: {gpu_grad[max_grad_idx]:.1f}")
    print(f"  Difference: {grad_diff[max_grad_idx]:.1f}")
    
    print(f"\nMaximum curvature difference at X(CU)={x_cu_values[max_curv_idx]:.3f}:")
    print(f"  CPU curvature: {cpu_curv[max_curv_idx]:.1f}")
    print(f"  GPU curvature: {gpu_curv[max_curv_idx]:.1f}")
    print(f"  Difference: {curv_diff[max_curv_idx]:.1f}")
    
    # ASCII plot
    print("\n" + "=" * 80)
    print("ASCII PLOT OF GM ALONG X(AL)=0.40")
    print("=" * 80)
    
    # Normalize for plotting
    min_gm = min(min(cpu_gms), min(gpu_gms))
    max_gm = max(max(cpu_gms), max(gpu_gms))
    range_gm = max_gm - min_gm
    
    height = 20
    width = 60
    
    # Create plot
    plot = [[' ' for _ in range(width)] for _ in range(height)]
    
    for i, x_cu in enumerate(x_cu_values):
        x_pos = int((i / (len(x_cu_values) - 1)) * (width - 1))
        
        # CPU point
        cpu_y = int((1 - (cpu_gms[i] - min_gm) / range_gm) * (height - 1))
        if 0 <= cpu_y < height:
            plot[cpu_y][x_pos] = 'C' if plot[cpu_y][x_pos] == ' ' else '*'
        
        # GPU point
        gpu_y = int((1 - (gpu_gms[i] - min_gm) / range_gm) * (height - 1))
        if 0 <= gpu_y < height:
            plot[gpu_y][x_pos] = 'G' if plot[gpu_y][x_pos] == ' ' else '*'
    
    # Print plot
    print(f"GM (J/mol)")
    print(f"{max_gm:8.0f} |" + "-" * width)
    for row in plot:
        print("         |" + ''.join(row))
    print(f"{min_gm:8.0f} |" + "-" * width)
    print(f"         X(CU): 0.35" + " " * (width - 20) + "0.45")
    print("\nLegend: C=CPU, G=GPU, *=Both agree")
    
    # Conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    
    if np.max(np.abs(cpu_curv)) > 2 * np.max(np.abs(gpu_curv)):
        print("CPU shows much higher curvature - likely hitting a spurious local minimum")
        print("GPU result appears more physically reasonable (smoother energy surface)")
    elif np.max(np.abs(gpu_curv)) > 2 * np.max(np.abs(cpu_curv)):
        print("GPU shows much higher curvature - likely hitting a spurious local minimum")
        print("CPU result appears more physically reasonable (smoother energy surface)")
    else:
        print("Both show similar curvature - difference may be due to different convergence paths")

if __name__ == "__main__":
    main()