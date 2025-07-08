#!/usr/bin/env python
"""
Debug chemical potential calculation differences between CPU and GPU.
"""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings('ignore')

def debug_mu_calculation():
    """Debug chemical potential differences"""
    print("="*60)
    print("CHEMICAL POTENTIAL DEBUG")
    print("="*60)
    
    dbf = Database("NbTi.tdb")
    
    # Test multiple conditions
    test_conditions = [
        {v.X("TI"): 1e-10, v.T: 600, v.P: 101325},  # Pure NB
        {v.X("TI"): 0.05, v.T: 600, v.P: 101325},   # 5% TI
        {v.X("TI"): 0.5, v.T: 700, v.P: 101325},    # 50% TI  
        {v.X("TI"): 0.95, v.T: 800, v.P: 101325},   # 95% TI
    ]
    
    for i, conditions in enumerate(test_conditions):
        print(f"\nCondition {i+1}: T={conditions[v.T]}K, X_TI={conditions[v.X('TI')]}")
        
        # CPU calculation
        cpu_result = equilibrium(dbf, ['NB', 'TI', 'VA'], ['LIQUID', 'BCC_A2'], 
                               conditions, gpu=False, verbose=False)
        cpu_gm = float(cpu_result.GM.values[0,0,0,0])
        cpu_mu = cpu_result.MU.values[0,0,0,0,:]
        cpu_np = cpu_result.NP.values[0,0,0,0,:]
        cpu_phases = cpu_result.Phase.values[0,0,0,0,:]
        
        cpu_active_mask = cpu_np > 1e-10
        cpu_active_phases = cpu_phases[cpu_active_mask]
        cpu_active_amounts = cpu_np[cpu_active_mask]
        
        print(f"  CPU:")
        print(f"    GM: {cpu_gm:.6f}")
        print(f"    MU_NB: {cpu_mu[0]:.6f}")
        print(f"    MU_TI: {cpu_mu[1]:.6f}")
        print(f"    Active phases: {cpu_active_phases}")
        print(f"    Phase amounts: {cpu_active_amounts}")
        
        # GPU calculation
        try:
            gpu_result = equilibrium(dbf, ['NB', 'TI', 'VA'], ['LIQUID', 'BCC_A2'], 
                                   conditions, gpu=True, verbose=False)
            gpu_gm = float(gpu_result.GM.values[0,0,0,0])
            gpu_mu = gpu_result.MU.values[0,0,0,0,:]
            gpu_np = gpu_result.NP.values[0,0,0,0,:]
            gpu_phases = gpu_result.Phase.values[0,0,0,0,:]
            
            gpu_active_mask = gpu_np > 1e-10
            gpu_active_phases = gpu_phases[gpu_active_mask]
            gpu_active_amounts = gpu_np[gpu_active_mask]
            
            print(f"  GPU:")
            print(f"    GM: {gpu_gm:.6f}")
            print(f"    MU_NB: {gpu_mu[0]:.6f}")
            print(f"    MU_TI: {gpu_mu[1]:.6f}")
            print(f"    Active phases: {gpu_active_phases}")
            print(f"    Phase amounts: {gpu_active_amounts}")
            
            # Analyze differences
            print(f"  Differences:")
            print(f"    ΔGM: {abs(cpu_gm - gpu_gm):.6f}")
            print(f"    ΔMU_NB: {abs(cpu_mu[0] - gpu_mu[0]):.6f}")
            print(f"    ΔMU_TI: {abs(cpu_mu[1] - gpu_mu[1]):.6f}")
            
            # Check if phase configuration matches
            if len(cpu_active_phases) == len(gpu_active_phases):
                print(f"    Phase count: ✅ Both have {len(cpu_active_phases)} phases")
            else:
                print(f"    Phase count: ❌ CPU has {len(cpu_active_phases)}, GPU has {len(gpu_active_phases)}")
            
            # Check which component has the bigger error
            nb_error = abs(cpu_mu[0] - gpu_mu[0])
            ti_error = abs(cpu_mu[1] - gpu_mu[1])
            
            if nb_error > ti_error:
                print(f"    Primary error: NB chemical potential")
            else:
                print(f"    Primary error: TI chemical potential")
                
        except Exception as e:
            print(f"  GPU failed: {e}")
    
    print(f"\n" + "="*60)
    print(f"SUMMARY:")
    print(f"Looking for patterns in which chemical potentials are wrong...")

if __name__ == "__main__":
    debug_mu_calculation()