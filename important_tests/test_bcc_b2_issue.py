#!/usr/bin/env python
"""Test BCC_B2 phase specifically to understand the issue."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def main():
    """Test BCC_B2 phase issue."""
    
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    
    print("=" * 80)
    print("BCC_B2 PHASE INVESTIGATION")
    print("=" * 80)
    
    print("\nPhase information from database:")
    print(f"BCC_B2 sublattice model: {dbf.phases['BCC_B2'].sublattices}")
    print(f"BCC_A2 sublattice model: {dbf.phases['BCC_A2'].sublattices}")
    print(f"FCC_A1 sublattice model: {dbf.phases['FCC_A1'].sublattices}")
    print(f"LIQUID sublattice model: {dbf.phases['LIQUID'].sublattices}")
    
    # Test at different compositions
    test_conditions = [
        (0.2, 0.5, 900, "Original failing condition"),
        (0.3, 0.3, 900, "Equal Al-Cu"),
        (0.1, 0.1, 900, "Low Al-Cu"),
        (0.4, 0.4, 900, "High Al-Cu"),
    ]
    
    print("\n" + "=" * 80)
    print("Testing with LIQUID + FCC_A1 + BCC_A2 + BCC_B2:")
    print("-" * 80)
    print("X(AL) | X(CU) | T(K) | CPU GM    | GPU GM    | Diff    | Status")
    print("------|-------|------|-----------|-----------|---------|--------")
    
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2']
    
    for x_al, x_cu, temp, desc in test_conditions:
        conditions = {
            v.X('AL'): x_al,
            v.X('CU'): x_cu,
            v.T: temp,
            v.P: 101325
        }
        
        # CPU
        cpu_result = equilibrium(dbf, comps, phases, conditions,
                                calc_opts={'pdens': 50},
                                gpu=False, verbose=False)
        cpu_gm = cpu_result.GM.values.item()
        
        # GPU
        gpu_result = equilibrium(dbf, comps, phases, conditions,
                                calc_opts={'pdens': 50},
                                gpu=True, verbose=False)
        gpu_gm = gpu_result.GM.values.item()
        
        diff = abs(gpu_gm - cpu_gm)
        status = "✓" if diff < 100 else "✗"
        
        print(f" {x_al:.2f}  | {x_cu:.2f}  | {temp:4d} | {cpu_gm:9.1f} | {gpu_gm:9.1f} | {diff:7.1f} | {status}")
    
    # Test with just BCC_B2
    print("\n" + "=" * 80)
    print("Testing with ONLY BCC_B2 phase:")
    print("-" * 80)
    
    phases_only_bcc_b2 = ['BCC_B2']
    
    for x_al, x_cu, temp, desc in test_conditions[:2]:  # Just test a couple
        conditions = {
            v.X('AL'): x_al,
            v.X('CU'): x_cu,
            v.T: temp,
            v.P: 101325
        }
        
        try:
            # CPU
            cpu_result = equilibrium(dbf, comps, phases_only_bcc_b2, conditions,
                                    calc_opts={'pdens': 50},
                                    gpu=False, verbose=False)
            cpu_gm = cpu_result.GM.values.item()
            
            # GPU
            gpu_result = equilibrium(dbf, comps, phases_only_bcc_b2, conditions,
                                    calc_opts={'pdens': 50},
                                    gpu=True, verbose=False)
            gpu_gm = gpu_result.GM.values.item()
            
            diff = abs(gpu_gm - cpu_gm)
            status = "✓" if diff < 100 else "✗"
            
            print(f"X(AL)={x_al:.2f}, X(CU)={x_cu:.2f}: CPU={cpu_gm:.1f}, GPU={gpu_gm:.1f}, Diff={diff:.1f} {status}")
        except Exception as e:
            print(f"X(AL)={x_al:.2f}, X(CU)={x_cu:.2f}: ERROR - {str(e)[:50]}")
    
    # Compare with simpler ordered phases
    print("\n" + "=" * 80)
    print("Comparing ordered phases (each with LIQUID):")
    print("-" * 80)
    
    ordered_phases = [
        ('BCC_A2', "2 sublattices (1 + 3 sites)"),
        ('FCC_A1', "2 sublattices (1 + 1 sites)"),
        ('BCC_B2', "3 sublattices (0.5 + 0.5 + 3 sites)"),
        ('L12', "L12 ordered phase"),
    ]
    
    conditions = {
        v.X('AL'): 0.2,
        v.X('CU'): 0.5,
        v.T: 900,
        v.P: 101325
    }
    
    for phase, desc in ordered_phases:
        phases = ['LIQUID', phase]
        
        # CPU
        cpu_result = equilibrium(dbf, comps, phases, conditions,
                                calc_opts={'pdens': 50},
                                gpu=False, verbose=False)
        cpu_gm = cpu_result.GM.values.item()
        
        # GPU
        gpu_result = equilibrium(dbf, comps, phases, conditions,
                                calc_opts={'pdens': 50},
                                gpu=True, verbose=False)
        gpu_gm = gpu_result.GM.values.item()
        
        diff = abs(gpu_gm - cpu_gm)
        status = "✓" if diff < 100 else "✗"
        
        print(f"{phase:10s} ({desc}): Diff={diff:.1f} {status}")
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("BCC_B2 is a complex ordered phase with 3 sublattices and fractional")
    print("site occupancies (0.5, 0.5, 3). This complexity appears to cause")
    print("divergence in the GPU solver, possibly due to:")
    print("  1. Numerical precision issues with fractional sites")
    print("  2. Different handling of multi-sublattice constraints")
    print("  3. Issues in gradient/Hessian calculations for complex phases")
    print("=" * 80)

if __name__ == "__main__":
    main()