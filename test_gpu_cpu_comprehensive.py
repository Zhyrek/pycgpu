#!/usr/bin/env python
"""Comprehensive GPU vs CPU comparison across multiple conditions."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import time

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Load database and set up calculation
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Define test conditions
x_ti_values = np.arange(0.1, 1.0, 0.1)  # 0.1 to 0.9 in 0.1 increments
temperatures = np.arange(500, 1001, 100)  # 500 to 1000 in 100 increments
pressure = 101325

print(f"Testing {len(x_ti_values)} compositions x {len(temperatures)} temperatures = {len(x_ti_values)*len(temperatures)} total conditions")
print("This may take a few minutes...")

# Open output file
with open('gpu_cpu_comparison_results.txt', 'w') as f:
    # Write header
    f.write("X(TI)\tT(K)\tCPU_GM\tGPU_GM\tGM_DIFF\tCPU_MU_NB\tCPU_MU_TI\tGPU_MU_NB\tGPU_MU_TI\tMU_NB_DIFF\tMU_TI_DIFF\tSTATUS\n")
    
    total_conditions = 0
    passed_conditions = 0
    failed_conditions = []
    
    start_time = time.time()
    
    for x_ti in x_ti_values:
        for temp in temperatures:
            total_conditions += 1
            conditions = {v.X('TI'): x_ti, v.T: temp, v.P: pressure}
            
            try:
                # CPU calculation
                result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
                cpu_gm = result_cpu.GM.values.flatten()[0]
                cpu_mu = result_cpu.MU.values.flatten()
                cpu_mu_nb = cpu_mu[0] if len(cpu_mu) > 0 else np.nan
                cpu_mu_ti = cpu_mu[1] if len(cpu_mu) > 1 else np.nan
                
                # GPU calculation
                result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
                gpu_gm = result_gpu.GM.values.flatten()[0]
                gpu_mu = result_gpu.MU.values.flatten()
                gpu_mu_nb = gpu_mu[0] if len(gpu_mu) > 0 else np.nan
                gpu_mu_ti = gpu_mu[1] if len(gpu_mu) > 1 else np.nan
                
                # Calculate differences
                gm_diff = abs(gpu_gm - cpu_gm)
                mu_nb_diff = abs(gpu_mu_nb - cpu_mu_nb)
                mu_ti_diff = abs(gpu_mu_ti - cpu_mu_ti)
                
                # Check tolerance (1 J/mol for energy)
                tolerance = 1.0
                status = "PASS" if (gm_diff < tolerance and mu_nb_diff < tolerance and mu_ti_diff < tolerance) else "FAIL"
                
                if status == "PASS":
                    passed_conditions += 1
                else:
                    failed_conditions.append((x_ti, temp))
                
                # Write results
                f.write(f"{x_ti:.1f}\t{temp}\t{cpu_gm:.6f}\t{gpu_gm:.6f}\t{gm_diff:.6f}\t")
                f.write(f"{cpu_mu_nb:.6f}\t{cpu_mu_ti:.6f}\t{gpu_mu_nb:.6f}\t{gpu_mu_ti:.6f}\t")
                f.write(f"{mu_nb_diff:.6f}\t{mu_ti_diff:.6f}\t{status}\n")
                
                # Progress indicator
                if total_conditions % 10 == 0:
                    print(f"Completed {total_conditions}/{len(x_ti_values)*len(temperatures)} conditions...")
                    
            except Exception as e:
                f.write(f"{x_ti:.1f}\t{temp}\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR: {str(e)}\n")
                failed_conditions.append((x_ti, temp))
    
    # Write summary
    elapsed_time = time.time() - start_time
    f.write(f"\n# SUMMARY\n")
    f.write(f"# Total conditions tested: {total_conditions}\n")
    f.write(f"# Passed: {passed_conditions}\n")
    f.write(f"# Failed: {total_conditions - passed_conditions}\n")
    f.write(f"# Pass rate: {passed_conditions/total_conditions*100:.1f}%\n")
    f.write(f"# Total time: {elapsed_time:.1f} seconds\n")
    
    if failed_conditions:
        f.write(f"\n# FAILED CONDITIONS:\n")
        for x_ti, temp in failed_conditions:
            f.write(f"# X(TI)={x_ti:.1f}, T={temp}K\n")

print(f"\nTest completed in {elapsed_time:.1f} seconds")
print(f"Results saved to gpu_cpu_comparison_results.txt")
print(f"\nSummary:")
print(f"  Total conditions: {total_conditions}")
print(f"  Passed: {passed_conditions}")
print(f"  Failed: {total_conditions - passed_conditions}")
print(f"  Pass rate: {passed_conditions/total_conditions*100:.1f}%")

if failed_conditions:
    print(f"\nFailed conditions:")
    for x_ti, temp in failed_conditions[:5]:  # Show first 5
        print(f"  X(TI)={x_ti:.1f}, T={temp}K")
    if len(failed_conditions) > 5:
        print(f"  ... and {len(failed_conditions)-5} more")