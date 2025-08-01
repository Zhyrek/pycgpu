#!/usr/bin/env python3
"""Debug script to trace vacancy handling in equilibrium calculations"""

import numpy as np
from pycalphad import Database, equilibrium, Model
import pycalphad.variables as v

def analyze_vacancy_in_equilibrium():
    """Analyze how vacancy is handled in equilibrium calculations for Au-Bi system"""
    
    # Initialize database and system
    db = Database('AuBi-07Wan.tdb')
    phases = ['LIQUID', 'FCC_A1']  # FCC has vacancy
    comps = ['AU', 'BI', 'VA']
    
    # Test a specific condition
    T = 600
    X_BI = 0.3
    
    print(f"\n{'='*80}")
    print(f"Testing Au-Bi system at T={T}K, X(BI)={X_BI}")
    print(f"{'='*80}\n")
    
    # Check phase model for FCC_A1
    print("FCC_A1 Phase Model:")
    mod_fcc = Model(db, comps, 'FCC_A1')
    print(f"  Sublattices: {mod_fcc.site_fractions}")
    print(f"  Site ratios: {mod_fcc.site_ratios}")
    print(f"  Components: {mod_fcc.components}")
    print(f"  Nonvacant elements: {mod_fcc.nonvacant_elements}")
    
    # Run CPU equilibrium with debug output
    print("\n" + "="*40 + " CPU Equilibrium " + "="*40)
    
    # Run with verbose mode to get debug output
    result_cpu = equilibrium(
        db, comps, phases,
        {v.X('BI'): X_BI, v.T: T},
        output='GM', 
        calc_opts={'pdens': 50},
        verbose=True,
        to_xarray=False
    )
    
    # Extract CPU results
    cpu_gm = float(result_cpu.GM.squeeze())
    phase_names = result_cpu.Phase.squeeze().values
    phase_amounts = result_cpu.NP.squeeze().values
    
    print(f"\nCPU Results:")
    print(f"  Total GM: {cpu_gm:.6f} J/mol")
    print(f"  Phases present:")
    
    for i, (phase, amount) in enumerate(zip(phase_names, phase_amounts)):
        if phase != '' and not np.isnan(amount) and amount > 1e-6:
            print(f"    {phase}: {amount:.6f}")
            # Get the X values for this phase
            x_values = result_cpu.X.squeeze()[i]
            print(f"      X(AU): {x_values[0]:.6f}")
            print(f"      X(BI): {x_values[1]:.6f}")
            print(f"      X(VA): {x_values[2]:.6f}")
            print(f"      Sum X: {np.sum(x_values):.6f}")
    
    # Run GPU equilibrium
    print("\n" + "="*40 + " GPU Equilibrium " + "="*40)
    
    result_gpu = equilibrium(
        db, comps, phases,
        {v.X('BI'): X_BI, v.T: T},
        output='GM',
        calc_opts={'pdens': 50},
        verbose=True,
        to_xarray=False,
        gpu=True
    )
    
    # Extract GPU results
    gpu_gm = float(result_gpu.GM.squeeze())
    gpu_phase_names = result_gpu.Phase.squeeze().values
    gpu_phase_amounts = result_gpu.NP.squeeze().values
    
    print(f"\nGPU Results:")
    print(f"  Total GM: {gpu_gm:.6f} J/mol")
    print(f"  Phases present:")
    
    for i, (phase, amount) in enumerate(zip(gpu_phase_names, gpu_phase_amounts)):
        if phase != '' and not np.isnan(amount) and amount > 1e-6:
            print(f"    {phase}: {amount:.6f}")
            x_values = result_gpu.X.squeeze()[i]
            print(f"      X(AU): {x_values[0]:.6f}")
            print(f"      X(BI): {x_values[1]:.6f}")
            print(f"      X(VA): {x_values[2]:.6f}")
            print(f"      Sum X: {np.sum(x_values):.6f}")
    
    print(f"\n{'='*80}")
    print(f"Differences:")
    print(f"  GM difference: {gpu_gm - cpu_gm:.6f} J/mol")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    analyze_vacancy_in_equilibrium()