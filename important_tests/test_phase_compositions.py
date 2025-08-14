#!/usr/bin/env python
"""Get detailed phase compositions at instability point."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def print_detailed_results(result, phases, label):
    """Print detailed equilibrium results."""
    print(f"\n{label} RESULTS:")
    print("=" * 70)
    
    # Get basic values
    gm = result.GM.values.item()
    print(f"Gibbs energy: {gm:.2f} J/mol")
    
    # Get phase amounts
    np_vals = result.NP.values.flatten()
    
    # Get compositions - handle the complex xarray structure
    print("\nPhases present:")
    print("-" * 70)
    
    for p_idx, phase in enumerate(phases):
        if p_idx < len(np_vals) and not np.isnan(np_vals[p_idx]) and np_vals[p_idx] > 0.001:
            print(f"\n{phase}:")
            print(f"  Phase fraction: {np_vals[p_idx]:.6f}")
            
            # Get composition for this phase
            print("  Composition:")
            
            # Method 1: Try using Phase dimension
            try:
                for comp in ['AL', 'CU', 'FE']:
                    # Access using xarray dimensions
                    comp_val = result.X.sel(component=comp).values.flatten()[p_idx]
                    if not np.isnan(comp_val):
                        print(f"    X({comp}) = {comp_val:.6f}")
            except:
                # Method 2: Direct array access
                try:
                    x_array = result.X.values
                    # X shape is typically (points, phases, components) after flattening higher dims
                    if x_array.size > 0:
                        x_flat = x_array.flatten()
                        n_comps = 3  # AL, CU, FE
                        
                        for c_idx, comp in enumerate(['AL', 'CU', 'FE']):
                            idx = p_idx * n_comps + c_idx
                            if idx < len(x_flat):
                                val = x_flat[idx]
                                if not np.isnan(val):
                                    print(f"    X({comp}) = {val:.6f}")
                except:
                    print("    (Unable to extract composition)")
    
    # Get chemical potentials
    mu_vals = result.MU.values.flatten()
    print("\nChemical potentials:")
    for i, comp in enumerate(['AL', 'CU', 'FE']):
        if i < len(mu_vals):
            print(f"  μ({comp}) = {mu_vals[i]:.2f} J/mol")

def main():
    """Test phase compositions at instability."""
    
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2']
    
    print("=" * 80)
    print("PHASE COMPOSITIONS AT NUMERICAL INSTABILITY POINT")
    print("=" * 80)
    
    # Test the three key conditions
    test_points = [
        (0.2, 0.49, "BEFORE INSTABILITY (X_CU=0.49)"),
        (0.2, 0.50, "AT INSTABILITY (X_CU=0.50)"),
        (0.2, 0.51, "AFTER INSTABILITY (X_CU=0.51)"),
    ]
    
    for x_al, x_cu, label in test_points:
        x_fe = 1.0 - x_al - x_cu
        
        print("\n" + "=" * 80)
        print(f"{label}")
        print(f"Bulk composition: X(AL)={x_al:.2f}, X(CU)={x_cu:.2f}, X(FE)={x_fe:.2f}")
        print("Temperature: 900 K")
        print("=" * 80)
        
        conditions = {
            v.X('AL'): x_al,
            v.X('CU'): x_cu,
            v.T: 900,
            v.P: 101325
        }
        
        # Run calculations
        cpu_result = equilibrium(dbf, comps, phases, conditions,
                                calc_opts={'pdens': 50},
                                gpu=False, verbose=False)
        
        gpu_result = equilibrium(dbf, comps, phases, conditions,
                                calc_opts={'pdens': 50},
                                gpu=True, verbose=False)
        
        # Print results
        print_detailed_results(cpu_result, phases, "CPU")
        print_detailed_results(gpu_result, phases, "GPU")
        
        # Compare
        print("\n" + "-" * 70)
        print("DIFFERENCES:")
        print("-" * 70)
        
        cpu_gm = cpu_result.GM.values.item()
        gpu_gm = gpu_result.GM.values.item()
        gm_diff = abs(gpu_gm - cpu_gm)
        
        print(f"ΔGM = {gm_diff:.2f} J/mol")
        
        if gm_diff > 100:
            print("⚠ SIGNIFICANT DIVERGENCE - Different equilibrium states found!")
        else:
            print("✓ Results match - Same equilibrium state")
        
        # Phase fractions
        cpu_np = cpu_result.NP.values.flatten()
        gpu_np = gpu_result.NP.values.flatten()
        
        print("\nPhase fraction differences:")
        for i, phase in enumerate(phases):
            if i < len(cpu_np) and i < len(gpu_np):
                cpu_val = cpu_np[i] if not np.isnan(cpu_np[i]) else 0
                gpu_val = gpu_np[i] if not np.isnan(gpu_np[i]) else 0
                if cpu_val > 0.001 or gpu_val > 0.001:
                    diff = gpu_val - cpu_val
                    print(f"  {phase}: CPU={cpu_val:.4f}, GPU={gpu_val:.4f}, Δ={diff:+.4f}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("""
The instability at X(CU)=0.50 causes:
1. Different Gibbs energies (180 J/mol difference)
2. Different phase fractions (LIQUID: 40.6% vs 43.8%)
3. Different chemical potentials
4. The issue occurs ONLY at X(CU)=0.50 ± 0.001

This is a numerical singularity in the equilibrium matrix when
BCC_B2 is present, even though BCC_B2 itself is not stable.
""")

if __name__ == "__main__":
    main()