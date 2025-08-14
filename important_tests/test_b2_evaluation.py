#!/usr/bin/env python
"""Test how BCC_B2 is evaluated during the solver iterations."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def main():
    """Test BCC_B2 evaluation during solving."""
    
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    
    print("=" * 80)
    print("BCC_B2 EVALUATION DURING SOLVER")
    print("=" * 80)
    
    # The problematic condition
    conditions = {
        v.X('AL'): 0.2,
        v.X('CU'): 0.5,
        v.T: 900,
        v.P: 101325
    }
    
    print("\nCondition: X(AL)=0.2, X(CU)=0.5, X(FE)=0.3, T=900K")
    print("\nBCC_B2 has 3 sublattices with site fractions (0.5, 0.5, 3.0)")
    print("This means:")
    print("  - Sublattice 1: 0.5 sites")
    print("  - Sublattice 2: 0.5 sites")
    print("  - Sublattice 3: 3.0 sites")
    print("  - Total: 4.0 sites")
    
    print("\n" + "-" * 80)
    print("Testing solver behavior with different phase lists:")
    print("-" * 80)
    
    # Test 1: Just LIQUID and FCC_A1 (baseline)
    phases_base = ['LIQUID', 'FCC_A1']
    print("\n1. LIQUID + FCC_A1 only:")
    
    cpu_result = equilibrium(dbf, comps, phases_base, conditions,
                            calc_opts={'pdens': 50},
                            gpu=False, verbose=False)
    gpu_result = equilibrium(dbf, comps, phases_base, conditions,
                            calc_opts={'pdens': 50},
                            gpu=True, verbose=False)
    
    cpu_gm = cpu_result.GM.values.item()
    gpu_gm = gpu_result.GM.values.item()
    diff = abs(gpu_gm - cpu_gm)
    
    print(f"   CPU GM: {cpu_gm:.1f} J/mol")
    print(f"   GPU GM: {gpu_gm:.1f} J/mol")
    print(f"   Difference: {diff:.1f} J/mol")
    
    # Test 2: Add BCC_A2 (simpler ordered phase)
    phases_with_a2 = ['LIQUID', 'FCC_A1', 'BCC_A2']
    print("\n2. LIQUID + FCC_A1 + BCC_A2:")
    
    cpu_result = equilibrium(dbf, comps, phases_with_a2, conditions,
                            calc_opts={'pdens': 50},
                            gpu=False, verbose=False)
    gpu_result = equilibrium(dbf, comps, phases_with_a2, conditions,
                            calc_opts={'pdens': 50},
                            gpu=True, verbose=False)
    
    cpu_gm = cpu_result.GM.values.item()
    gpu_gm = gpu_result.GM.values.item()
    diff = abs(gpu_gm - cpu_gm)
    
    print(f"   CPU GM: {cpu_gm:.1f} J/mol")
    print(f"   GPU GM: {gpu_gm:.1f} J/mol")
    print(f"   Difference: {diff:.1f} J/mol")
    
    # Test 3: Add BCC_B2 (problematic phase)
    phases_with_b2 = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2']
    print("\n3. LIQUID + FCC_A1 + BCC_A2 + BCC_B2:")
    
    cpu_result = equilibrium(dbf, comps, phases_with_b2, conditions,
                            calc_opts={'pdens': 50},
                            gpu=False, verbose=False)
    gpu_result = equilibrium(dbf, comps, phases_with_b2, conditions,
                            calc_opts={'pdens': 50},
                            gpu=True, verbose=False)
    
    cpu_gm = cpu_result.GM.values.item()
    gpu_gm = gpu_result.GM.values.item()
    diff = abs(gpu_gm - cpu_gm)
    
    cpu_np = cpu_result.NP.values.flatten()
    gpu_np = gpu_result.NP.values.flatten()
    
    print(f"   CPU GM: {cpu_gm:.1f} J/mol")
    print(f"   GPU GM: {gpu_gm:.1f} J/mol")
    print(f"   Difference: {diff:.1f} J/mol {'✗ DIVERGED' if diff > 100 else ''}")
    
    print("\n   Phase amounts:")
    for i, phase in enumerate(phases_with_b2):
        if i < len(cpu_np):
            print(f"     {phase:10s}: CPU={cpu_np[i]:.4f}, GPU={gpu_np[i]:.4f}")
    
    # Test 4: Check if the issue is specific to the composition
    print("\n" + "=" * 80)
    print("Testing if divergence is composition-specific with BCC_B2:")
    print("-" * 80)
    
    test_points = [
        (0.2, 0.5, "Original failing"),
        (0.2, 0.49, "Slightly less Cu"),
        (0.2, 0.51, "Slightly more Cu"),
        (0.19, 0.5, "Slightly less Al"),
        (0.21, 0.5, "Slightly more Al"),
    ]
    
    phases_with_b2 = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2']
    
    print("\nX(AL) | X(CU) | CPU GM    | GPU GM    | Diff    | Description")
    print("------|-------|-----------|-----------|---------|-------------")
    
    for x_al, x_cu, desc in test_points:
        conditions = {
            v.X('AL'): x_al,
            v.X('CU'): x_cu,
            v.T: 900,
            v.P: 101325
        }
        
        cpu_result = equilibrium(dbf, comps, phases_with_b2, conditions,
                                calc_opts={'pdens': 50},
                                gpu=False, verbose=False)
        gpu_result = equilibrium(dbf, comps, phases_with_b2, conditions,
                                calc_opts={'pdens': 50},
                                gpu=True, verbose=False)
        
        cpu_gm = cpu_result.GM.values.item()
        gpu_gm = gpu_result.GM.values.item()
        diff = abs(gpu_gm - cpu_gm)
        
        status = "✓" if diff < 100 else "✗"
        print(f" {x_al:.2f}  | {x_cu:.2f}  | {cpu_gm:9.1f} | {gpu_gm:9.1f} | {diff:7.1f} | {desc} {status}")
    
    print("\n" + "=" * 80)
    print("HYPOTHESIS:")
    print("The BCC_B2 phase with its 3 sublattices and fractional site occupancies")
    print("causes numerical issues in the GPU solver at specific compositions.")
    print("Even though BCC_B2 is not stable, the solver still evaluates it during")
    print("the search, and errors in its gradient/Hessian calculations affect the")
    print("overall convergence path.")
    print("=" * 80)

if __name__ == "__main__":
    main()