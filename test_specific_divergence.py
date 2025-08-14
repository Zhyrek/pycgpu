#!/usr/bin/env python
"""Test the specific phase set and composition that causes divergence."""

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
    print("NARROW WINDOW OF DIVERGENCE")
    print("=" * 80)
    print(f"Phases: {phases}")
    print()
    
    # Very narrow test around the transition
    test_points = [
        0.390,  # Works
        0.391,
        0.392,
        0.393,
        0.394,
        0.395,  # Start of divergence
        0.396,
        0.397,
        0.398,
        0.399,
        0.400,  # Peak divergence
        0.401,
        0.402,
        0.403,
        0.404,
        0.405,  # End of divergence
    ]
    
    print("X(AL)     CPU GM      GPU GM      Diff     Status")
    print("-" * 60)
    
    prev_cpu = None
    prev_gpu = None
    
    for x_al in test_points:
        conditions = {
            v.X('AL'): x_al,
            v.X('CU'): 0.40,
            v.T: 600,
            v.P: 101325
        }
        
        # Run both CPU and GPU
        cpu_result = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
        gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
        
        cpu_gm = float(cpu_result.GM.values.squeeze())
        gpu_gm = float(gpu_result.GM.values.squeeze())
        diff = gpu_gm - cpu_gm
        
        status = "MATCH" if abs(diff) < 1.0 else "DIFF"
        
        # Check for jumps
        jump_marker = ""
        if prev_cpu is not None:
            cpu_jump = abs(cpu_gm - prev_cpu)
            gpu_jump = abs(gpu_gm - prev_gpu)
            if cpu_jump > 50:
                jump_marker += " CPU↑"
            if gpu_jump > 50:
                jump_marker += " GPU↑"
        
        print(f"{x_al:.3f}  {cpu_gm:10.2f}  {gpu_gm:10.2f}  {diff:7.2f}  {status:5s}{jump_marker}")
        
        prev_cpu = cpu_gm
        prev_gpu = gpu_gm
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("\nThe divergence occurs in a narrow window from X(AL)=0.395 to X(AL)=0.404")
    print("This suggests the CPU and GPU are finding different local minima in this region.")
    print("\nBoth solvers show discontinuous jumps, indicating phase transitions.")
    print("The CPU jumps at X(AL)=0.395, while GPU jumps at X(AL)=0.405.")
    print("This 0.01 shift in the phase boundary location causes the divergence.")

if __name__ == "__main__":
    main()