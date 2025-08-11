#!/usr/bin/env python
"""Quick GPU vs CPU comparison for Al-Cu-Fe system."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import time
import warnings
warnings.filterwarnings("ignore")

def run_test():
    """Run GPU vs CPU comparison test for Al-Cu-Fe system."""
    
    # Load database
    dbf = Database('Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = list(dbf.phases.keys())
    
    print(f"Testing Al-Cu-Fe with {len(phases)} phases")
    
    # Test with just 5 conditions for quick comparison
    n_conditions = 5
    
    x_cu_values = np.array([0.1, 0.2, 0.3, 0.33, 0.4])
    x_fe_values = np.array([0.1, 0.2, 0.3, 0.33, 0.2])
    temperatures = np.array([700, 800, 900, 1000, 1100])
    
    conditions = {
        v.T: temperatures,
        v.P: 101325 * np.ones(n_conditions),
        v.X('CU'): x_cu_values,
        v.X('FE'): x_fe_values
    }
    
    print(f"\nTesting {n_conditions} conditions:")
    for i in range(n_conditions):
        x_al = 1.0 - x_cu_values[i] - x_fe_values[i]
        print(f"  {i+1}: X(AL)={x_al:.2f}, X(CU)={x_cu_values[i]:.2f}, X(FE)={x_fe_values[i]:.2f}, T={temperatures[i]:.0f}K")
    
    # CPU calculation
    print("\nRunning CPU calculation...")
    cpu_start = time.time()
    result_cpu = equilibrium(dbf, comps, phases, conditions, 
                            gpu=False, verbose=False, 
                            calc_opts={'pdens': 60})
    cpu_time = time.time() - cpu_start
    
    # GPU calculation
    print("Running GPU calculation...")
    gpu_start = time.time()
    result_gpu = equilibrium(dbf, comps, phases, conditions, 
                            gpu=True, verbose=False,
                            calc_opts={'pdens': 60})
    gpu_time = time.time() - gpu_start
    
    # Extract and compare results
    cpu_gm = result_cpu.GM.values.flatten()
    gpu_gm = result_gpu.GM.values.flatten()
    
    cpu_mu_al = result_cpu.MU.sel(component='AL').values.flatten()
    cpu_mu_cu = result_cpu.MU.sel(component='CU').values.flatten()
    cpu_mu_fe = result_cpu.MU.sel(component='FE').values.flatten()
    
    gpu_mu_al = result_gpu.MU.sel(component='AL').values.flatten()
    gpu_mu_cu = result_gpu.MU.sel(component='CU').values.flatten()
    gpu_mu_fe = result_gpu.MU.sel(component='FE').values.flatten()
    
    print(f"\n" + "="*80)
    print(f"RESULTS COMPARISON")
    print(f"="*80)
    print(f"{'Cond':<5} {'X(AL)':<6} {'X(CU)':<6} {'X(FE)':<6} {'T(K)':<6} {'CPU GM':<12} {'GPU GM':<12} {'|Diff|':<10} {'Status':<8}")
    print("-" * 80)
    
    passed = 0
    tolerance = 1.0  # 1 J/mol tolerance
    
    for i in range(n_conditions):
        x_al = 1.0 - x_cu_values[i] - x_fe_values[i]
        gm_diff = abs(gpu_gm[i] - cpu_gm[i])
        mu_al_diff = abs(gpu_mu_al[i] - cpu_mu_al[i])
        mu_cu_diff = abs(gpu_mu_cu[i] - cpu_mu_cu[i])
        mu_fe_diff = abs(gpu_mu_fe[i] - cpu_mu_fe[i])
        
        status = "PASS" if (gm_diff < tolerance and 
                          mu_al_diff < tolerance and 
                          mu_cu_diff < tolerance and 
                          mu_fe_diff < tolerance) else "FAIL"
        
        if status == "PASS":
            passed += 1
        
        print(f"{i+1:<5} {x_al:<6.3f} {x_cu_values[i]:<6.3f} {x_fe_values[i]:<6.3f} {temperatures[i]:<6.0f} "
              f"{cpu_gm[i]:<12.2f} {gpu_gm[i]:<12.2f} {gm_diff:<10.6f} {status:<8}")
    
    print("-" * 80)
    print(f"\nChemical Potentials (J/mol):")
    print(f"{'Cond':<5} {'CPU μ(AL)':<12} {'GPU μ(AL)':<12} {'CPU μ(CU)':<12} {'GPU μ(CU)':<12} {'CPU μ(FE)':<12} {'GPU μ(FE)':<12}")
    print("-" * 80)
    
    for i in range(n_conditions):
        print(f"{i+1:<5} {cpu_mu_al[i]:<12.2f} {gpu_mu_al[i]:<12.2f} "
              f"{cpu_mu_cu[i]:<12.2f} {gpu_mu_cu[i]:<12.2f} "
              f"{cpu_mu_fe[i]:<12.2f} {gpu_mu_fe[i]:<12.2f}")
    
    print(f"\n" + "="*80)
    print(f"SUMMARY")
    print(f"="*80)
    print(f"Conditions tested: {n_conditions}")
    print(f"Passed: {passed}/{n_conditions} ({passed/n_conditions*100:.0f}%)")
    print(f"Failed: {n_conditions-passed}/{n_conditions} ({(n_conditions-passed)/n_conditions*100:.0f}%)")
    print(f"CPU time: {cpu_time:.2f} seconds")
    print(f"GPU time: {gpu_time:.2f} seconds")
    print(f"Speedup: {cpu_time/gpu_time:.1f}x")
    
    return passed == n_conditions

if __name__ == "__main__":
    success = run_test()
    exit(0 if success else 1)