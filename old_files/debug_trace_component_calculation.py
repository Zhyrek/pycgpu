#!/usr/bin/env python
"""
Debug chemical potential calculation for trace components.
"""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings('ignore')

def debug_trace_component():
    """Debug chemical potential calculation for trace components"""
    print("="*60)
    print("TRACE COMPONENT CHEMICAL POTENTIAL DEBUG")
    print("="*60)
    
    dbf = Database("NbTi.tdb")
    
    # Test series: pure NB with increasing Ti content
    ti_fractions = [1e-10, 1e-8, 1e-6, 1e-4, 0.01, 0.05]
    
    print(f"Testing TI chemical potential calculation at various TI concentrations:")
    print(f"Temperature: 600K, Pressure: 101325 Pa")
    print()
    
    for ti_frac in ti_fractions:
        conditions = {v.X("TI"): ti_frac, v.T: 600, v.P: 101325}
        
        print(f"X_TI = {ti_frac:.2e}")
        
        # CPU calculation
        try:
            cpu_result = equilibrium(dbf, ['NB', 'TI', 'VA'], ['LIQUID', 'BCC_A2'], 
                                   conditions, gpu=False, verbose=False)
            cpu_mu_nb = float(cpu_result.MU.values[0,0,0,0,0])
            cpu_mu_ti = float(cpu_result.MU.values[0,0,0,0,1])
            print(f"  CPU - MU_NB: {cpu_mu_nb:.3f}, MU_TI: {cpu_mu_ti:.3f}")
            
        except Exception as e:
            print(f"  CPU failed: {e}")
            continue
            
        # GPU calculation
        try:
            gpu_result = equilibrium(dbf, ['NB', 'TI', 'VA'], ['LIQUID', 'BCC_A2'], 
                                   conditions, gpu=True, verbose=False)
            gpu_mu_nb = float(gpu_result.MU.values[0,0,0,0,0])
            gpu_mu_ti = float(gpu_result.MU.values[0,0,0,0,1])
            print(f"  GPU - MU_NB: {gpu_mu_nb:.3f}, MU_TI: {gpu_mu_ti:.3f}")
            
            # Calculate errors
            nb_error = abs(cpu_mu_nb - gpu_mu_nb)
            ti_error = abs(cpu_mu_ti - gpu_mu_ti)
            
            print(f"  Errors - NB: {nb_error:.3e}, TI: {ti_error:.3e}")
            
            # Check if error correlates with concentration
            if ti_error > 1000:
                print(f"  ❌ HUGE TI ERROR at low concentration!")
            elif ti_error > 1.0:
                print(f"  ⚠️  Large TI error")
            else:
                print(f"  ✅ TI error acceptable")
                
        except Exception as e:
            print(f"  GPU failed: {e}")
        
        print()
    
    print("="*60)
    print("ANALYSIS:")
    print("If TI chemical potential errors are huge at low concentrations,")
    print("this suggests a numerical precision issue in the GPU solver")
    print("when dealing with trace components.")

if __name__ == "__main__":
    debug_trace_component()