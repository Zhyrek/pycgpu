#!/usr/bin/env python
"""
Test if GPU solver is converging properly.
"""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings('ignore')

def test_gpu_convergence():
    """Test if GPU solver is actually running iterations"""
    print("="*60)
    print("GPU CONVERGENCE TEST")
    print("="*60)
    
    dbf = Database("NbTi.tdb")
    
    # Use pure NB case that should be simple
    conditions = {v.X("TI"): 1e-10, v.T: 600, v.P: 101325}
    
    print(f"Testing: Pure NB at 600K")
    print(f"Expected: Single BCC_A2 phase with specific chemical potentials")
    
    # Run with verbose output to see if we get convergence info
    print(f"\n1. CPU EQUILIBRIUM (reference):")
    cpu_result = equilibrium(dbf, ['NB', 'TI', 'VA'], ['LIQUID', 'BCC_A2'], 
                           conditions, gpu=False, verbose=False)
    
    cpu_gm = float(cpu_result.GM.values[0,0,0,0])
    cpu_mu_nb = float(cpu_result.MU.values[0,0,0,0,0])
    cpu_mu_ti = float(cpu_result.MU.values[0,0,0,0,1])
    
    print(f"  GM: {cpu_gm:.6f}")
    print(f"  MU_NB: {cpu_mu_nb:.6f}")
    print(f"  MU_TI: {cpu_mu_ti:.6f}")
    
    print(f"\n2. GPU EQUILIBRIUM (test):")
    # Try to get more debug info
    gpu_result = equilibrium(dbf, ['NB', 'TI', 'VA'], ['LIQUID', 'BCC_A2'], 
                           conditions, gpu=True, verbose=True)  # Enable verbose
    
    gpu_gm = float(gpu_result.GM.values[0,0,0,0])
    gpu_mu_nb = float(gpu_result.MU.values[0,0,0,0,0])
    gpu_mu_ti = float(gpu_result.MU.values[0,0,0,0,1])
    
    print(f"  GM: {gpu_gm:.6f}")
    print(f"  MU_NB: {gpu_mu_nb:.6f}")
    print(f"  MU_TI: {gpu_mu_ti:.6f}")
    
    print(f"\n3. ANALYSIS:")
    gm_diff = abs(cpu_gm - gpu_gm)
    nb_diff = abs(cpu_mu_nb - gpu_mu_nb)
    ti_diff = abs(cpu_mu_ti - gpu_mu_ti)
    
    print(f"  Differences:")
    print(f"    GM: {gm_diff:.6e}")
    print(f"    MU_NB: {nb_diff:.6e}")
    print(f"    MU_TI: {ti_diff:.6e}")
    
    if ti_diff > 10000:
        print(f"\n  ❌ CONCLUSION: GPU solver is NOT converging properly")
        print(f"     TI chemical potential error of {ti_diff:.0f} indicates")
        print(f"     the solver is stuck at an intermediate state")
        print(f"     and not reaching true equilibrium.")
    elif max(gm_diff, nb_diff, ti_diff) < 0.001:
        print(f"\n  ✅ CONCLUSION: GPU solver is working correctly")
    else:
        print(f"\n  ⚠️  CONCLUSION: GPU solver has moderate errors")

if __name__ == "__main__":
    test_gpu_convergence()