#!/usr/bin/env python
"""Test AuBi system with just FCC_A1 and LIQUID phases using multiple conditions."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import time

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

def run_two_phase_test():
    """Run GPU vs CPU comparison test for AuBi system with FCC_A1 and LIQUID only."""
    
    # Load database and set up calculation  
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    phases = ['FCC_A1', 'LIQUID']  # Just the two phases we know work
    
    print(f"Testing phases: {phases}")
    
    # Define test conditions
    conditions = {
        v.X('BI'): (0.1, 0.9, 0.1),  # 0.1 to 0.9 in 0.1 increments
        v.T: (400, 800, 100),        # 400 to 800 in 100 increments  
        v.P: 101325
    }
    
    x_bi_values = np.arange(0.1, 0.9, 0.1)  # [0.1, 0.2, ..., 0.8]
    temperatures = np.arange(400, 801, 100)  # [400, 500, 600, 700, 800]
    expected_conditions = 8 * 5  # 8 compositions * 5 temperatures = 40
    
    print(f"Testing {len(x_bi_values)} compositions x {len(temperatures)} temperatures = {expected_conditions} total conditions")
    
    start_time = time.time()
    
    try:
        # CPU calculation
        print("Running CPU calculation...")
        cpu_start = time.time()
        result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False)
        cpu_time = time.time() - cpu_start
        print(f"CPU calculation completed in {cpu_time:.1f} seconds")
        
        # GPU calculation
        print("Running GPU calculation...")
        gpu_start = time.time()
        result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True)
        gpu_time = time.time() - gpu_start
        print(f"GPU calculation completed in {gpu_time:.1f} seconds")
        
        # Extract results
        cpu_gm = result_cpu.GM.values.flatten()
        gpu_gm = result_gpu.GM.values.flatten()
        cpu_mu_au = result_cpu.MU.sel(component='AU').values.flatten()
        cpu_mu_bi = result_cpu.MU.sel(component='BI').values.flatten()
        gpu_mu_au = result_gpu.MU.sel(component='AU').values.flatten()
        gpu_mu_bi = result_gpu.MU.sel(component='BI').values.flatten()
        
        print(f"\nResult shapes:")
        print(f"  CPU GM shape: {cpu_gm.shape}")
        print(f"  GPU GM shape: {gpu_gm.shape}")
        
        # Get coordinates
        temp_coords = result_cpu.coords['T'].values if 'T' in result_cpu.coords else temperatures
        x_bi_coords = result_cpu.coords['X_BI'].values if 'X_BI' in result_cpu.coords else x_bi_values
        
        # Build condition arrays
        x_bi_flat = []
        temp_flat = []
        
        for t_val in temp_coords:
            for x_val in x_bi_coords:
                temp_flat.append(t_val)
                x_bi_flat.append(x_val)
        
        x_bi_flat = np.array(x_bi_flat)
        temp_flat = np.array(temp_flat)
        
        # Truncate to match result length
        min_len = min(len(cpu_gm), len(x_bi_flat))
        x_bi_flat = x_bi_flat[:min_len]
        temp_flat = temp_flat[:min_len]
        cpu_gm = cpu_gm[:min_len]
        gpu_gm = gpu_gm[:min_len]
        cpu_mu_au = cpu_mu_au[:min_len]
        cpu_mu_bi = cpu_mu_bi[:min_len]
        gpu_mu_au = gpu_mu_au[:min_len]
        gpu_mu_bi = gpu_mu_bi[:min_len]
        
        # Compare results
        passed_conditions = 0
        failed_conditions = []
        
        with open('gpu_cpu_aubi_two_phases_results_multi.txt', 'w') as f:
            f.write("X(BI)\tT(K)\tCPU_GM\tGPU_GM\tGM_DIFF\tCPU_MU_AU\tCPU_MU_BI\tGPU_MU_AU\tGPU_MU_BI\tMU_AU_DIFF\tMU_BI_DIFF\tSTATUS\n")
            
            for i in range(min_len):
                x_bi = x_bi_flat[i]
                temp = temp_flat[i]
                
                # Skip NaN values
                if np.isnan(cpu_gm[i]) or np.isnan(gpu_gm[i]):
                    continue
                
                # Calculate differences
                gm_diff = abs(gpu_gm[i] - cpu_gm[i])
                mu_au_diff = abs(gpu_mu_au[i] - cpu_mu_au[i])
                mu_bi_diff = abs(gpu_mu_bi[i] - cpu_mu_bi[i])
                
                # Check tolerance (1 J/mol for energy)
                tolerance = 1.0
                status = "PASS" if (gm_diff < tolerance and mu_au_diff < tolerance and mu_bi_diff < tolerance) else "FAIL"
                
                if status == "PASS":
                    passed_conditions += 1
                else:
                    failed_conditions.append((x_bi, temp))
                
                # Write results
                f.write(f"{x_bi:.1f}\t{temp:.0f}\t{cpu_gm[i]:.12f}\t{gpu_gm[i]:.12f}\t{gm_diff:.12f}\t")
                f.write(f"{cpu_mu_au[i]:.12f}\t{cpu_mu_bi[i]:.12f}\t{gpu_mu_au[i]:.12f}\t{gpu_mu_bi[i]:.12f}\t")
                f.write(f"{mu_au_diff:.12f}\t{mu_bi_diff:.12f}\t{status}\n")
            
            # Write summary
            elapsed_time = time.time() - start_time
            f.write(f"\n# SUMMARY\n")
            f.write(f"# Total conditions tested: {min_len}\n")
            f.write(f"# Passed: {passed_conditions}\n")
            f.write(f"# Failed: {min_len - passed_conditions}\n")
            f.write(f"# Pass rate: {passed_conditions/min_len*100:.1f}%\n")
            f.write(f"# CPU time: {cpu_time:.1f} seconds\n")
            f.write(f"# GPU time: {gpu_time:.1f} seconds\n")
            f.write(f"# Speedup: {cpu_time/gpu_time:.1f}x\n")
            f.write(f"# Total time: {elapsed_time:.1f} seconds\n")
        
        print(f"\nTest completed in {elapsed_time:.1f} seconds")
        print(f"Results saved to gpu_cpu_aubi_two_phases_results_multi.txt")
        print(f"\nSummary:")
        print(f"  Total conditions: {min_len}")
        print(f"  Passed: {passed_conditions}")
        print(f"  Failed: {min_len - passed_conditions}")
        print(f"  Pass rate: {passed_conditions/min_len*100:.1f}%")
        print(f"\nPerformance:")
        print(f"  CPU time: {cpu_time:.1f} seconds")
        print(f"  GPU time: {gpu_time:.1f} seconds")
        print(f"  Speedup: {cpu_time/gpu_time:.1f}x")
        
        if failed_conditions:
            print(f"\nFailed conditions:")
            for x_bi, temp in failed_conditions[:5]:  # Show first 5
                print(f"  X(BI)={x_bi:.1f}, T={temp:.0f}K")
            if len(failed_conditions) > 5:
                print(f"  ... and {len(failed_conditions)-5} more")
                
    except Exception as e:
        print(f"\nError during test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_two_phase_test()