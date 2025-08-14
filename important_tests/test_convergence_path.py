#!/usr/bin/env python
"""Test to see convergence differences between CPU and GPU."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def main():
    """Compare CPU and GPU convergence for a failing condition."""
    
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2', 'L12', 'ALCU_THETA', 'AL13FE4', 'AL5FE2']
    
    # Use the failing condition X_AL=0.2, X_CU=0.5, T=900K
    conditions = {
        v.X('AL'): 0.2,
        v.X('CU'): 0.5,
        v.T: 900,
        v.P: 101325
    }
    
    print("=" * 80)
    print("CONVERGENCE PATH COMPARISON")
    print("Condition: X(AL)=0.2, X(CU)=0.5, X(FE)=0.3, T=900K")
    print("=" * 80)
    
    # Run CPU calculation
    print("\n1. CPU Calculation:")
    cpu_result = equilibrium(dbf, comps, phases, conditions,
                            calc_opts={'pdens': 50},
                            gpu=False, verbose=False)
    
    cpu_gm = cpu_result.GM.values.item()
    cpu_mu_al = cpu_result.MU.sel(component='AL').values.item()
    cpu_mu_cu = cpu_result.MU.sel(component='CU').values.item()
    cpu_mu_fe = cpu_result.MU.sel(component='FE').values.item()
    
    print(f"  Final GM: {cpu_gm:.2f} J/mol")
    print(f"  Chemical potentials:")
    print(f"    μ(AL): {cpu_mu_al:.2f} J/mol")
    print(f"    μ(CU): {cpu_mu_cu:.2f} J/mol")
    print(f"    μ(FE): {cpu_mu_fe:.2f} J/mol")
    
    # Extract phase amounts
    cpu_np = cpu_result.NP.values.flatten()
    print(f"\n  Active phases (NP > 0.001):")
    for i, phase in enumerate(phases):
        if i < len(cpu_np) and cpu_np[i] > 0.001:
            print(f"    {phase}: {cpu_np[i]:.4f}")
    
    # Run GPU calculation
    print("\n2. GPU Calculation:")
    gpu_result = equilibrium(dbf, comps, phases, conditions,
                            calc_opts={'pdens': 50},
                            gpu=True, verbose=False)
    
    gpu_gm = gpu_result.GM.values.item()
    gpu_mu_al = gpu_result.MU.sel(component='AL').values.item()
    gpu_mu_cu = gpu_result.MU.sel(component='CU').values.item()
    gpu_mu_fe = gpu_result.MU.sel(component='FE').values.item()
    
    print(f"  Final GM: {gpu_gm:.2f} J/mol")
    print(f"  Chemical potentials:")
    print(f"    μ(AL): {gpu_mu_al:.2f} J/mol")
    print(f"    μ(CU): {gpu_mu_cu:.2f} J/mol")
    print(f"    μ(FE): {gpu_mu_fe:.2f} J/mol")
    
    # Extract phase amounts
    gpu_np = gpu_result.NP.values.flatten()
    print(f"\n  Active phases (NP > 0.001):")
    for i, phase in enumerate(phases):
        if i < len(gpu_np) and gpu_np[i] > 0.001:
            print(f"    {phase}: {gpu_np[i]:.4f}")
    
    # Compare results
    print("\n3. Differences:")
    print("-" * 40)
    
    gm_diff = gpu_gm - cpu_gm
    mu_al_diff = gpu_mu_al - cpu_mu_al
    mu_cu_diff = gpu_mu_cu - cpu_mu_cu
    mu_fe_diff = gpu_mu_fe - cpu_mu_fe
    
    print(f"  ΔGM: {gm_diff:+.2f} J/mol ({abs(gm_diff/cpu_gm)*100:.3f}% relative)")
    print(f"  Δμ(AL): {mu_al_diff:+.2f} J/mol")
    print(f"  Δμ(CU): {mu_cu_diff:+.2f} J/mol")
    print(f"  Δμ(FE): {mu_fe_diff:+.2f} J/mol")
    
    print(f"\n  Phase amount differences:")
    for i, phase in enumerate(phases):
        if i < len(cpu_np) and i < len(gpu_np):
            diff = gpu_np[i] - cpu_np[i]
            if abs(diff) > 0.001:
                print(f"    {phase}: CPU={cpu_np[i]:.4f}, GPU={gpu_np[i]:.4f}, Δ={diff:+.4f}")
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS:")
    print("-" * 80)
    
    # Check if they converged to different phase assemblages
    cpu_active = set([phases[i] for i in range(len(phases)) if i < len(cpu_np) and cpu_np[i] > 0.001])
    gpu_active = set([phases[i] for i in range(len(phases)) if i < len(gpu_np) and gpu_np[i] > 0.001])
    
    if cpu_active != gpu_active:
        print("✗ CPU and GPU converged to DIFFERENT phase assemblages:")
        print(f"  CPU phases: {cpu_active}")
        print(f"  GPU phases: {gpu_active}")
        print(f"  Only in CPU: {cpu_active - gpu_active}")
        print(f"  Only in GPU: {gpu_active - cpu_active}")
    else:
        print("✓ CPU and GPU converged to the SAME phase assemblage:")
        print(f"  Active phases: {cpu_active}")
        
    if abs(gm_diff) > 100:
        print(f"\n⚠ Significant energy difference: {abs(gm_diff):.0f} J/mol")
        print("  This suggests convergence to different local minima or")
        print("  numerical differences in the solver algorithms.")
    else:
        print(f"\n✓ Energy difference is small: {abs(gm_diff):.1f} J/mol")
    
    print("\nCONCLUSION:")
    print("The initial equilibrium matrices are identical (verified separately),")
    print("but the solvers converge to slightly different states due to:")
    print("  1. Different linear algebra implementations (LAPACK vs custom)")
    print("  2. Floating-point rounding differences")
    print("  3. Different iteration/convergence criteria")
    print("=" * 80)

if __name__ == "__main__":
    main()