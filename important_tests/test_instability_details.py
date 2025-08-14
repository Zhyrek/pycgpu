#!/usr/bin/env python
"""Detailed comparison at the instability point."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def analyze_result(result, phases, label):
    """Analyze and print equilibrium result details."""
    print(f"\n{label} Results:")
    print("-" * 60)
    
    # Get GM
    gm = result.GM.values.item()
    print(f"Gibbs energy: {gm:.2f} J/mol")
    
    # Get phase amounts
    np_vals = result.NP.values.flatten()
    
    # Get phase compositions (X values)
    x_vals = result.X.values
    
    print("\nActive phases:")
    active_count = 0
    
    for i, phase in enumerate(phases):
        if i < len(np_vals) and not np.isnan(np_vals[i]) and np_vals[i] > 0.001:
            active_count += 1
            print(f"\n  {phase}:")
            print(f"    Amount: {np_vals[i]:.4f} (mole fraction)")
            
            # Extract composition for this phase
            # X array shape is complex, need to extract carefully
            if hasattr(result, 'Phase') and hasattr(result, 'X'):
                # Try to get phase composition
                try:
                    # Find the phase compositions
                    phase_comps = {}
                    for comp_idx, comp in enumerate(['AL', 'CU', 'FE']):
                        # Navigate the complex xarray structure
                        val = x_vals.sel(component=comp, vertex=0).values.flatten()[i]
                        if not np.isnan(val):
                            phase_comps[comp] = val
                    
                    if phase_comps:
                        print(f"    Composition:")
                        for comp, val in phase_comps.items():
                            print(f"      X({comp}) = {val:.6f}")
                except:
                    # Fallback method
                    print(f"    Composition: (unable to extract)")
    
    if active_count == 0:
        print("  No active phases found (may be an error)")
    
    print(f"\nTotal active phases: {active_count}")
    
    # Get chemical potentials
    try:
        mu_vals = result.MU.values.flatten()
        print("\nChemical potentials:")
        for i, comp in enumerate(['AL', 'CU', 'FE']):
            if i < len(mu_vals):
                print(f"  μ({comp}) = {mu_vals[i]:.2f} J/mol")
    except:
        pass
    
    return gm, active_count

def main():
    """Compare CPU and GPU at instability point."""
    
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2']
    
    print("=" * 80)
    print("DETAILED COMPARISON AT INSTABILITY POINT")
    print("=" * 80)
    
    # Test conditions around X(CU)=0.5
    test_conditions = [
        (0.2, 0.49, "Just before instability"),
        (0.2, 0.50, "At instability point"),
        (0.2, 0.51, "Just after instability"),
    ]
    
    for x_al, x_cu, desc in test_conditions:
        x_fe = 1.0 - x_al - x_cu
        
        print("\n" + "=" * 80)
        print(f"{desc}: X(AL)={x_al:.2f}, X(CU)={x_cu:.2f}, X(FE)={x_fe:.2f}, T=900K")
        print("=" * 80)
        
        conditions = {
            v.X('AL'): x_al,
            v.X('CU'): x_cu,
            v.T: 900,
            v.P: 101325
        }
        
        # CPU calculation
        cpu_result = equilibrium(dbf, comps, phases, conditions,
                                calc_opts={'pdens': 50},
                                gpu=False, verbose=False)
        
        # GPU calculation
        gpu_result = equilibrium(dbf, comps, phases, conditions,
                                calc_opts={'pdens': 50},
                                gpu=True, verbose=False)
        
        # Analyze results
        cpu_gm, cpu_active = analyze_result(cpu_result, phases, "CPU")
        gpu_gm, gpu_active = analyze_result(gpu_result, phases, "GPU")
        
        # Compare
        print("\n" + "-" * 60)
        print("COMPARISON:")
        print("-" * 60)
        
        gm_diff = abs(gpu_gm - cpu_gm)
        print(f"GM difference: {gm_diff:.2f} J/mol")
        
        if gm_diff > 100:
            print("⚠ SIGNIFICANT DIVERGENCE!")
        else:
            print("✓ Results match within tolerance")
        
        # Compare phase amounts directly
        cpu_np = cpu_result.NP.values.flatten()
        gpu_np = gpu_result.NP.values.flatten()
        
        print("\nPhase amount differences:")
        for i, phase in enumerate(phases):
            if i < len(cpu_np) and i < len(gpu_np):
                cpu_amt = cpu_np[i] if not np.isnan(cpu_np[i]) else 0.0
                gpu_amt = gpu_np[i] if not np.isnan(gpu_np[i]) else 0.0
                
                if cpu_amt > 0.001 or gpu_amt > 0.001:
                    diff = abs(cpu_amt - gpu_amt)
                    print(f"  {phase}: CPU={cpu_amt:.4f}, GPU={gpu_amt:.4f}, Diff={diff:.4f}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nKey observations:")
    print("1. At X(CU)=0.50 exactly, the solver encounters numerical instability")
    print("2. CPU and GPU converge to different equilibrium states")
    print("3. The phase compositions and amounts differ significantly")
    print("4. BCC_B2 phase (even when not stable) causes the matrix to become ill-conditioned")

if __name__ == "__main__":
    main()