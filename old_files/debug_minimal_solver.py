#!/usr/bin/env python
"""
Minimal test to trigger GPU solver debugging.
"""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings('ignore')

def test_minimal_gpu_debug():
    """Test GPU debug with just 1 condition to trigger debug arrays"""
    print("="*60)
    print("MINIMAL GPU SOLVER DEBUG")
    print("="*60)
    
    dbf = Database("NbTi.tdb")
    
    # Use just 1 condition to trigger debug arrays
    conditions = {v.X("TI"): 1e-10, v.T: 600, v.P: 101325}
    
    print(f"Testing single condition: T=600K, X_TI=1e-10")
    print(f"Expected debug arrays to be enabled (<=10 conditions)")
    
    try:
        print(f"\nRunning GPU equilibrium with verbose=True...")
        gpu_result = equilibrium(dbf, ['NB', 'TI', 'VA'], ['LIQUID', 'BCC_A2'], 
                               conditions, gpu=True, verbose=True)
        
        print(f"\nGPU Result:")
        print(f"  GM: {gpu_result.GM.values}")
        print(f"  MU: {gpu_result.MU.values}")
        print(f"  NP: {gpu_result.NP.values}")
        print(f"  Phase: {gpu_result.Phase.values}")
        
        return gpu_result
        
    except Exception as e:
        print(f"GPU failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_with_cpu():
    """Compare with CPU for same condition"""
    print(f"\n" + "="*60)
    print("CPU COMPARISON")
    print("="*60)
    
    dbf = Database("NbTi.tdb")
    conditions = {v.X("TI"): 1e-10, v.T: 600, v.P: 101325}
    
    print(f"Running CPU equilibrium...")
    cpu_result = equilibrium(dbf, ['NB', 'TI', 'VA'], ['LIQUID', 'BCC_A2'], 
                           conditions, gpu=False, verbose=False)
    
    print(f"\nCPU Result:")
    print(f"  GM: {cpu_result.GM.values}")
    print(f"  MU: {cpu_result.MU.values}")
    print(f"  NP: {cpu_result.NP.values}")
    print(f"  Phase: {cpu_result.Phase.values}")
    
    return cpu_result

if __name__ == "__main__":
    gpu_result = test_minimal_gpu_debug()
    cpu_result = compare_with_cpu()
    
    if gpu_result is not None and cpu_result is not None:
        print(f"\n" + "="*60)
        print("COMPARISON")
        print("="*60)
        
        gpu_gm = float(gpu_result.GM.values)
        cpu_gm = float(cpu_result.GM.values)
        
        print(f"GM comparison:")
        print(f"  GPU: {gpu_gm:.6f}")
        print(f"  CPU: {cpu_gm:.6f}")
        print(f"  Difference: {abs(gpu_gm - cpu_gm):.6f}")
        
        if abs(gpu_gm - cpu_gm) < 0.001:
            print(f"  🚨 GPU returning starting point - solver not running!")
        else:
            print(f"  ✅ Values differ - solver appears to be working")