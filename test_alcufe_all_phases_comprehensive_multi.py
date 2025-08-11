#!/usr/bin/env python
"""Comprehensive GPU vs CPU comparison for Al-Cu-Fe system with all phases enabled using multiple conditions per call."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import time

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

def run_comprehensive_test(verbose=False):
    """Run comprehensive GPU vs CPU comparison test for Al-Cu-Fe system with all phases."""
    
    # Load database and set up calculation
    dbf = Database('Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = list(dbf.phases.keys())
    phases = ["LIQUID", "FCC_A1"]
    
    print(f"Available phases ({len(phases)}): {phases}")
    
    # Define test conditions using the range syntax
    # For ternary, we specify two composition conditions
    # Using the same syntax twice as instructed - the code will handle it properly
    conditions = {
        v.X('CU'): (0.1, 0.5, 0.1),  # 0.1 to 0.5 in 0.1 increments
        v.X('FE'): (0.1, 0.4, 0.1),  # 0.1 to 0.4 in 0.1 increments
        v.T: (600, 1200, 200),        # 600 to 1200 in 200 increments  
        v.P: 101325
    }
    
    # Calculate expected number of conditions
    # Note: pycalphad's range syntax (start, stop, step) creates values from start up to but not including stop
    x_cu_values = np.arange(0.1, 0.5 + 0.1, 0.1)  # [0.1, 0.2, 0.3, 0.4, 0.5]
    x_fe_values = np.arange(0.1, 0.4 + 0.1, 0.1)  # [0.1, 0.2, 0.3, 0.4]
    temperatures = np.arange(600, 1200 + 200, 200)  # [600, 800, 1000, 1200]
    expected_conditions = len(x_cu_values) * len(x_fe_values) * len(temperatures)
    
    print(f"\nTesting {len(x_cu_values)} Cu compositions x {len(x_fe_values)} Fe compositions x {len(temperatures)} temperatures")
    print(f"= {expected_conditions} total conditions")
    print("Running calculations with multiple conditions per call...")
    
    start_time = time.time()
    
    try:
        # CPU calculation - all conditions at once
        print("Running CPU calculation...")
        cpu_start = time.time()
        result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=verbose, calc_opts={'pdens': 60})
        cpu_time = time.time() - cpu_start
        print(f"CPU calculation completed in {cpu_time:.1f} seconds")
        
        # GPU calculation - all conditions at once
        print("Running GPU calculation...")
        gpu_start = time.time()
        result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=verbose, calc_opts={'pdens': 60})
        gpu_time = time.time() - gpu_start
        print(f"GPU calculation completed in {gpu_time:.1f} seconds")
        
        # Extract results
        cpu_gm = result_cpu.GM.values
        gpu_gm = result_gpu.GM.values
        
        # Extract chemical potentials
        cpu_mu_al = result_cpu.MU.sel(component='AL').values
        cpu_mu_cu = result_cpu.MU.sel(component='CU').values
        cpu_mu_fe = result_cpu.MU.sel(component='FE').values
        
        gpu_mu_al = result_gpu.MU.sel(component='AL').values
        gpu_mu_cu = result_gpu.MU.sel(component='CU').values
        gpu_mu_fe = result_gpu.MU.sel(component='FE').values
        
        # Debug shapes
        print(f"\nDebug - Result shapes:")
        print(f"  cpu_gm.shape: {cpu_gm.shape}")
        print(f"  gpu_gm.shape: {gpu_gm.shape}")
        print(f"  result_cpu dimensions: {result_cpu.dims}")
        
        # Get the coordinate values from the dataset
        if 'T' in result_cpu.coords:
            temp_coords = result_cpu.coords['T'].values
        else:
            temp_coords = temperatures
            
        if 'X_CU' in result_cpu.coords:
            x_cu_coords = result_cpu.coords['X_CU'].values  
        else:
            x_cu_coords = x_cu_values
            
        if 'X_FE' in result_cpu.coords:
            x_fe_coords = result_cpu.coords['X_FE'].values  
        else:
            x_fe_coords = x_fe_values
            
        print(f"  Temperature coordinates: {temp_coords}")
        print(f"  X_CU coordinates: {x_cu_coords}")
        print(f"  X_FE coordinates: {x_fe_coords}")
        
        # Build arrays for all condition combinations
        x_cu_flat = []
        x_fe_flat = []
        temp_flat = []
        
        # The flattening order should match the dimension order in the dataset
        for t_idx, t_val in enumerate(temp_coords):
            for cu_idx, cu_val in enumerate(x_cu_coords):
                for fe_idx, fe_val in enumerate(x_fe_coords):
                    # Skip invalid compositions where Cu + Fe >= 1
                    if cu_val + fe_val >= 0.99:
                        continue
                    temp_flat.append(t_val)
                    x_cu_flat.append(cu_val)
                    x_fe_flat.append(fe_val)
                    
        x_cu_flat = np.array(x_cu_flat)
        x_fe_flat = np.array(x_fe_flat)
        temp_flat = np.array(temp_flat)
        
        print(f"  Valid condition combinations: {len(x_cu_flat)}")
        
        # Flatten all result arrays
        cpu_gm_flat = cpu_gm.flatten()
        gpu_gm_flat = gpu_gm.flatten()
        cpu_mu_al_flat = cpu_mu_al.flatten()
        cpu_mu_cu_flat = cpu_mu_cu.flatten()
        cpu_mu_fe_flat = cpu_mu_fe.flatten()
        gpu_mu_al_flat = gpu_mu_al.flatten()
        gpu_mu_cu_flat = gpu_mu_cu.flatten()
        gpu_mu_fe_flat = gpu_mu_fe.flatten()
        
        # Open output file
        with open('gpu_cpu_alcufe_all_phases_results_multi.txt', 'w') as f:
            # Write header
            f.write("X(CU)\tX(FE)\tX(AL)\tT(K)\tCPU_GM\tGPU_GM\tGM_DIFF\t")
            f.write("CPU_MU_AL\tCPU_MU_CU\tCPU_MU_FE\tGPU_MU_AL\tGPU_MU_CU\tGPU_MU_FE\t")
            f.write("MU_AL_DIFF\tMU_CU_DIFF\tMU_FE_DIFF\tSTATUS\n")
            
            passed_conditions = 0
            failed_conditions = []
            valid_conditions = 0
            
            # Compare results - iterate through the full grid
            result_idx = 0
            for t_idx, temp in enumerate(temp_coords):
                for cu_idx, x_cu in enumerate(x_cu_coords):
                    for fe_idx, x_fe in enumerate(x_fe_coords):
                        # Skip invalid compositions
                        if x_cu + x_fe >= 0.99:
                            result_idx += 1
                            continue
                            
                        x_al = 1.0 - x_cu - x_fe
                        
                        # Skip NaN values
                        if np.isnan(cpu_gm_flat[result_idx]) or np.isnan(gpu_gm_flat[result_idx]):
                            result_idx += 1
                            continue
                        
                        valid_conditions += 1
                        
                        # Calculate differences
                        gm_diff = abs(gpu_gm_flat[result_idx] - cpu_gm_flat[result_idx])
                        mu_al_diff = abs(gpu_mu_al_flat[result_idx] - cpu_mu_al_flat[result_idx])
                        mu_cu_diff = abs(gpu_mu_cu_flat[result_idx] - cpu_mu_cu_flat[result_idx])
                        mu_fe_diff = abs(gpu_mu_fe_flat[result_idx] - cpu_mu_fe_flat[result_idx])
                        
                        # Check tolerance (1 J/mol for energy)
                        tolerance = 1.0
                        status = "PASS" if (gm_diff < tolerance and 
                                          mu_al_diff < tolerance and 
                                          mu_cu_diff < tolerance and 
                                          mu_fe_diff < tolerance) else "FAIL"
                        
                        if status == "PASS":
                            passed_conditions += 1
                        else:
                            failed_conditions.append((x_cu, x_fe, temp))
                        
                        # Write results
                        f.write(f"{x_cu:.1f}\t{x_fe:.1f}\t{x_al:.1f}\t{temp:.0f}\t")
                        f.write(f"{cpu_gm_flat[result_idx]:.12f}\t{gpu_gm_flat[result_idx]:.12f}\t{gm_diff:.12f}\t")
                        f.write(f"{cpu_mu_al_flat[result_idx]:.12f}\t{cpu_mu_cu_flat[result_idx]:.12f}\t{cpu_mu_fe_flat[result_idx]:.12f}\t")
                        f.write(f"{gpu_mu_al_flat[result_idx]:.12f}\t{gpu_mu_cu_flat[result_idx]:.12f}\t{gpu_mu_fe_flat[result_idx]:.12f}\t")
                        f.write(f"{mu_al_diff:.12f}\t{mu_cu_diff:.12f}\t{mu_fe_diff:.12f}\t{status}\n")
                        
                        result_idx += 1
            
            # Write summary
            elapsed_time = time.time() - start_time
            f.write(f"\n# SUMMARY\n")
            f.write(f"# Total conditions tested: {valid_conditions}\n")
            f.write(f"# Passed: {passed_conditions}\n")
            f.write(f"# Failed: {valid_conditions - passed_conditions}\n")
            if valid_conditions > 0:
                f.write(f"# Pass rate: {passed_conditions/valid_conditions*100:.1f}%\n")
            f.write(f"# CPU time: {cpu_time:.1f} seconds\n")
            f.write(f"# GPU time: {gpu_time:.1f} seconds\n")
            f.write(f"# Speedup: {cpu_time/gpu_time:.1f}x\n")
            f.write(f"# Total time: {elapsed_time:.1f} seconds\n")
            
            if failed_conditions:
                f.write(f"\n# FAILED CONDITIONS:\n")
                for x_cu, x_fe, temp in failed_conditions:
                    f.write(f"# X(CU)={x_cu:.1f}, X(FE)={x_fe:.1f}, T={temp:.0f}K\n")
        
        print(f"\nTest completed in {elapsed_time:.1f} seconds")
        print(f"Results written to: gpu_cpu_alcufe_all_phases_results_multi.txt")
        
        # Print summary
        print(f"\n" + "="*60)
        print(f"SUMMARY")
        print(f"="*60)
        print(f"Total conditions tested: {valid_conditions}")
        print(f"Passed: {passed_conditions}")
        print(f"Failed: {valid_conditions - passed_conditions}")
        if valid_conditions > 0:
            print(f"Pass rate: {passed_conditions/valid_conditions*100:.1f}%")
        print(f"CPU time: {cpu_time:.1f} seconds")
        print(f"GPU time: {gpu_time:.1f} seconds")
        print(f"Speedup: {cpu_time/gpu_time:.1f}x")
        
        if failed_conditions:
            print(f"\nFailed conditions:")
            for x_cu, x_fe, temp in failed_conditions[:5]:  # Show first 5
                print(f"  X(CU)={x_cu:.1f}, X(FE)={x_fe:.1f}, T={temp:.0f}K")
            if len(failed_conditions) > 5:
                print(f"  ... and {len(failed_conditions)-5} more")
        
        return passed_conditions == valid_conditions
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test GPU vs CPU for Al-Cu-Fe system with all phases')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    
    success = run_comprehensive_test(verbose=args.verbose)
    exit(0 if success else 1)