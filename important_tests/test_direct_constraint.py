#!/usr/bin/env python
"""Direct test of constraint enforcement."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def check_solution(phases, amounts, compositions):
    """Check if a solution satisfies the constraints."""
    print("\nChecking solution:")
    print(f"  Phases: {phases}")
    print(f"  Amounts: {amounts}")
    print(f"  Compositions (X_AL, X_CU, X_FE): {compositions}")
    
    # Calculate bulk composition
    bulk_al = sum(amt * comp[0] for amt, comp in zip(amounts, compositions))
    bulk_cu = sum(amt * comp[1] for amt, comp in zip(amounts, compositions))
    bulk_fe = sum(amt * comp[2] for amt, comp in zip(amounts, compositions))
    
    print(f"\nBulk composition:")
    print(f"  X(AL) = {bulk_al:.6f} (target: 0.600000)")
    print(f"  X(CU) = {bulk_cu:.6f} (target: 0.100000)")
    print(f"  X(FE) = {bulk_fe:.6f} (target: 0.300000)")
    
    # Check mass conservation
    total = bulk_al + bulk_cu + bulk_fe
    print(f"  Total = {total:.6f} (should be 1.0)")
    
    # Check phase amounts
    total_amt = sum(amounts)
    print(f"\nTotal phase amount = {total_amt:.6f} (should be 1.0)")
    
    return abs(bulk_al - 0.60) < 0.001 and abs(bulk_cu - 0.10) < 0.001

def main():
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['BCC_B2', 'AL5FE2']
    
    conditions = {
        v.X('AL'): 0.60,
        v.X('CU'): 0.10,
        v.T: 600,
        v.P: 101325
    }
    
    print("=" * 80)
    print("DIRECT CONSTRAINT TEST")
    print("=" * 80)
    
    print("\nTarget conditions:")
    print("  X(AL) = 0.60")
    print("  X(CU) = 0.10")
    print("  X(FE) = 0.30 (by difference)")
    print("  T = 600 K")
    
    # CPU solution
    print("\n" + "=" * 80)
    print("CPU SOLUTION")
    print("=" * 80)
    
    cpu_result = equilibrium(dbf, comps, phases, conditions, gpu=False)
    
    cpu_phases = []
    cpu_amounts = []
    cpu_compositions = []
    
    for i in range(len(cpu_result.Phase.values.flatten())):
        phase = cpu_result.Phase.values.flatten()[i]
        amount = cpu_result.NP.values.flatten()[i]
        if not np.isnan(amount) and amount > 0.001:
            cpu_phases.append(phase)
            cpu_amounts.append(amount)
            x_vals = cpu_result.X.values.reshape(-1, 3)
            cpu_compositions.append(x_vals[i])
    
    cpu_ok = check_solution(cpu_phases, cpu_amounts, cpu_compositions)
    print(f"\nCPU solution {'✓ PASSES' if cpu_ok else '✗ FAILS'} constraints")
    
    # GPU solution
    print("\n" + "=" * 80)
    print("GPU SOLUTION")
    print("=" * 80)
    
    gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True)
    
    gpu_phases = []
    gpu_amounts = []
    gpu_compositions = []
    
    for i in range(len(gpu_result.Phase.values.flatten())):
        phase = gpu_result.Phase.values.flatten()[i]
        amount = gpu_result.NP.values.flatten()[i]
        if not np.isnan(amount) and amount > 0.001:
            gpu_phases.append(phase)
            gpu_amounts.append(amount)
            x_vals = gpu_result.X.values.reshape(-1, 3)
            gpu_compositions.append(x_vals[i])
    
    gpu_ok = check_solution(gpu_phases, gpu_amounts, gpu_compositions)
    print(f"\nGPU solution {'✓ PASSES' if gpu_ok else '✗ FAILS'} constraints")
    
    # Compare energies
    print("\n" + "=" * 80)
    print("ENERGY COMPARISON")
    print("=" * 80)
    
    cpu_gm = float(cpu_result.GM.values.flatten()[0])
    gpu_gm = float(gpu_result.GM.values.flatten()[0])
    
    print(f"CPU Gibbs energy: {cpu_gm:.2f} J/mol")
    print(f"GPU Gibbs energy: {gpu_gm:.2f} J/mol")
    print(f"Difference: {gpu_gm - cpu_gm:.2f} J/mol")
    
    if gpu_gm < cpu_gm:
        print("\nGPU found lower energy, but violates constraints!")
        print("This suggests the constraint enforcement is not working.")
    elif gpu_gm > cpu_gm:
        print("\nGPU found higher energy and violates constraints.")
        print("This suggests both the optimization and constraints are wrong.")

if __name__ == "__main__":
    main()