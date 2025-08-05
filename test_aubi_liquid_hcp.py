#!/usr/bin/env python
"""Test AuBi system with LIQUID and HCP_A3 phases specifically."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import time

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

def test_liquid_hcp():
    """Test LIQUID and HCP_A3 phase combination."""
    
    # Load database
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    phases = ['LIQUID', 'HCP_A3']  # Just LIQUID and HCP_A3
    
    print(f"Testing phases: {phases}")
    
    # Single condition test first
    conditions = {
        v.X('BI'): 0.1,
        v.T: 400,
        v.P: 101325
    }
    
    print(f"\n=== Single condition test: X(BI)=0.1, T=400K ===")
    
    try:
        # CPU calculation
        print("CPU calculation...")
        result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
        cpu_gm = result_cpu.GM.values[0,0,0,0]
        cpu_phases = result_cpu.Phase.values[0,0,0,0]
        cpu_np = result_cpu.NP.values[0,0,0,0]
        cpu_mu_au = result_cpu.MU.sel(component='AU').values[0,0,0,0]
        cpu_mu_bi = result_cpu.MU.sel(component='BI').values[0,0,0,0]
        
        print(f"  CPU GM: {cpu_gm:.6f}")
        print(f"  CPU MU: AU={cpu_mu_au:.6f}, BI={cpu_mu_bi:.6f}")
        print(f"  CPU active phases:")
        for i, (phase, amount) in enumerate(zip(cpu_phases, cpu_np)):
            if amount > 1e-8 and phase != '':
                print(f"    {phase}: {amount:.6f}")
        
        # GPU calculation
        print("GPU calculation...")
        result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
        gpu_gm = result_gpu.GM.values[0,0,0,0]
        gpu_phases = result_gpu.Phase.values[0,0,0,0]
        gpu_np = result_gpu.NP.values[0,0,0,0]
        gpu_mu_au = result_gpu.MU.sel(component='AU').values[0,0,0,0]
        gpu_mu_bi = result_gpu.MU.sel(component='BI').values[0,0,0,0]
        
        print(f"  GPU GM: {gpu_gm:.6f}")
        print(f"  GPU MU: AU={gpu_mu_au:.6f}, BI={gpu_mu_bi:.6f}")
        print(f"  GPU active phases:")
        for i, (phase, amount) in enumerate(zip(gpu_phases, gpu_np)):
            if amount > 1e-8 and phase != '':
                print(f"    {phase}: {amount:.6f}")
        
        # Compare
        gm_diff = abs(gpu_gm - cpu_gm)
        mu_au_diff = abs(gpu_mu_au - cpu_mu_au)
        mu_bi_diff = abs(gpu_mu_bi - cpu_mu_bi)
        
        print(f"\n  Differences:")
        print(f"    GM: {gm_diff:.2e}")
        print(f"    MU_AU: {mu_au_diff:.2e}")
        print(f"    MU_BI: {mu_bi_diff:.2e}")
        
        if gm_diff < 1.0 and mu_au_diff < 1.0 and mu_bi_diff < 1.0:
            print(f"  ✓ SINGLE CONDITION PASS")
            single_pass = True
        else:
            print(f"  ✗ SINGLE CONDITION FAIL")
            single_pass = False
            
    except Exception as e:
        print(f"  ✗ SINGLE CONDITION ERROR: {str(e)}")
        single_pass = False
        
    if not single_pass:
        return False
    
    # Multi-condition test
    print(f"\n=== Multi-condition test ===")
    
    conditions_multi = {
        v.X('BI'): (0.1, 0.5, 0.1),  # 0.1, 0.2, 0.3, 0.4, 0.5
        v.T: (400, 600, 100),        # 400, 500, 600
        v.P: 101325
    }
    
    try:
        # CPU calculation
        print("Running multi-condition CPU calculation...")
        cpu_start = time.time()
        result_cpu = equilibrium(dbf, comps, phases, conditions_multi, gpu=False)
        cpu_time = time.time() - cpu_start
        print(f"CPU completed in {cpu_time:.1f} seconds")
        
        # GPU calculation
        print("Running multi-condition GPU calculation...")
        gpu_start = time.time()
        result_gpu = equilibrium(dbf, comps, phases, conditions_multi, gpu=True)
        gpu_time = time.time() - gpu_start
        print(f"GPU completed in {gpu_time:.1f} seconds")
        
        # Extract and compare results
        cpu_gm = result_cpu.GM.values.flatten()
        gpu_gm = result_gpu.GM.values.flatten()
        cpu_mu_au = result_cpu.MU.sel(component='AU').values.flatten()
        cpu_mu_bi = result_cpu.MU.sel(component='BI').values.flatten()
        gpu_mu_au = result_gpu.MU.sel(component='AU').values.flatten()
        gpu_mu_bi = result_gpu.MU.sel(component='BI').values.flatten()
        
        print(f"\nResult array shapes:")
        print(f"  CPU GM: {cpu_gm.shape}")
        print(f"  GPU GM: {gpu_gm.shape}")
        
        # Compare each condition
        passed = 0
        total = min(len(cpu_gm), len(gpu_gm))
        
        print(f"\nComparing {total} conditions:")
        
        for i in range(total):
            if np.isnan(cpu_gm[i]) or np.isnan(gpu_gm[i]):
                continue
                
            gm_diff = abs(gpu_gm[i] - cpu_gm[i])
            mu_au_diff = abs(gpu_mu_au[i] - cpu_mu_au[i])
            mu_bi_diff = abs(gpu_mu_bi[i] - cpu_mu_bi[i])
            
            tolerance = 1.0
            if gm_diff < tolerance and mu_au_diff < tolerance and mu_bi_diff < tolerance:
                passed += 1
            else:
                print(f"  Condition {i}: GM_diff={gm_diff:.2e}, MU_AU_diff={mu_au_diff:.2e}, MU_BI_diff={mu_bi_diff:.2e}")
        
        pass_rate = passed / total * 100
        print(f"\nMulti-condition results:")
        print(f"  Total conditions: {total}")
        print(f"  Passed: {passed}")
        print(f"  Pass rate: {pass_rate:.1f}%")
        print(f"  CPU time: {cpu_time:.1f}s")
        print(f"  GPU time: {gpu_time:.1f}s")
        
        if pass_rate >= 95.0:
            print(f"  ✓ MULTI-CONDITION PASS")
            return True
        else:
            print(f"  ✗ MULTI-CONDITION FAIL")
            return False
            
    except Exception as e:
        print(f"  ✗ MULTI-CONDITION ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_liquid_hcp()
    print(f"\n=== FINAL RESULT ===")
    if success:
        print("✓ LIQUID + HCP_A3 system works correctly")
    else:
        print("✗ LIQUID + HCP_A3 system has issues")