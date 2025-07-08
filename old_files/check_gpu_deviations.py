#!/usr/bin/env python
"""
Check if GPU results match CPU results or just return properties data.
"""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings('ignore')

def check_gpu_deviations():
    """Check if GPU deviations are greater than 0.001 in absolute terms"""
    print("="*60)
    print("GPU vs CPU DEVIATION CHECK")
    print("="*60)
    
    dbf = Database("NbTi.tdb")
    
    # Test a few representative conditions
    test_conditions = [
        {v.X("TI"): 1e-10, v.T: 600, v.P: 101325},  # Pure NB at 600K
        {v.X("TI"): 0.05, v.T: 600, v.P: 101325},   # 5% TI at 600K  
        {v.X("TI"): 0.5, v.T: 700, v.P: 101325},    # 50% TI at 700K
        {v.X("TI"): 0.95, v.T: 800, v.P: 101325},   # 95% TI at 800K
    ]
    
    max_deviation = 0.0
    all_match = True
    
    for i, conditions in enumerate(test_conditions):
        print(f"\nCondition {i+1}: T={conditions[v.T]}K, X_TI={conditions[v.X('TI')]}")
        
        # CPU calculation
        try:
            cpu_result = equilibrium(dbf, ['NB', 'TI', 'VA'], ['LIQUID', 'BCC_A2'], 
                                   conditions, gpu=False, verbose=False)
            cpu_gm = float(cpu_result.GM.values[0,0,0,0])
            cpu_mu = cpu_result.MU.values[0,0,0,0,:]
            print(f"  CPU: GM={cpu_gm:.6f}, MU={cpu_mu[:2]}")
        except Exception as e:
            print(f"  CPU failed: {e}")
            continue
            
        # GPU calculation
        try:
            gpu_result = equilibrium(dbf, ['NB', 'TI', 'VA'], ['LIQUID', 'BCC_A2'], 
                                   conditions, gpu=True, verbose=False)
            gpu_gm = float(gpu_result.GM.values[0,0,0,0])
            gpu_mu = gpu_result.MU.values[0,0,0,0,:]
            print(f"  GPU: GM={gpu_gm:.6f}, MU={gpu_mu[:2]}")
        except Exception as e:
            print(f"  GPU failed: {e}")
            continue
        
        # Calculate deviations
        gm_dev = abs(cpu_gm - gpu_gm)
        mu_dev = np.max(np.abs(cpu_mu[:2] - gpu_mu[:2]))
        
        print(f"  Deviations: GM={gm_dev:.6f}, MU_max={mu_dev:.6f}")
        
        # Track maximum deviation
        max_deviation = max(max_deviation, gm_dev, mu_dev)
        
        # Check if deviation exceeds threshold
        if gm_dev > 0.001 or mu_dev > 0.001:
            print(f"  ❌ DEVIATION > 0.001!")
            all_match = False
        else:
            print(f"  ✅ Within tolerance")
    
    print(f"\n" + "="*60)
    print(f"FINAL RESULT")
    print(f"="*60)
    print(f"Maximum deviation found: {max_deviation:.6f}")
    
    if max_deviation > 0.001:
        print(f"❌ GPU SOLVER IS NOT WORKING PROPERLY")
        print(f"   Largest deviation ({max_deviation:.6f}) exceeds 0.001 threshold")
        print(f"   GPU may be returning properties data instead of solving")
    else:
        print(f"✅ GPU SOLVER IS WORKING CORRECTLY")
        print(f"   All deviations are within 0.001 tolerance")
        print(f"   GPU is performing equilibrium calculations, not just returning properties")
    
    return max_deviation

if __name__ == "__main__":
    max_dev = check_gpu_deviations()