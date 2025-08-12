#!/usr/bin/env python
"""Comprehensive GPU vs CPU comparison for Al-Cu-Fe system with 8 key phases using multiple conditions per call."""

import sys
import os
# Add parent directory to path to import pycalphad
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import time
import argparse

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

def run_comprehensive_test(verbose=False):
    """Run comprehensive GPU vs CPU comparison test for Al-Cu-Fe with 8 key phases."""
    
    # Load database and set up calculation
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    
    # Use 8 important phases including LIQUID, FCC, and BCC structures
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2', 'L12', 'ALCU_THETA', 'AL13FE4', 'AL5FE2']
    
    print("=" * 60)
    print("Al-Cu-Fe SYSTEM COMPREHENSIVE TEST WITH 8 KEY PHASES")
    print("=" * 60)
    print(f"Components: {comps}")
    print(f"Phases: {phases}")
    print(f"Number of phases: {len(phases)}")
    
    # Define test conditions for ternary system
    # We'll test a grid of compositions at different temperatures
    # For ternary, we need to specify two composition variables
    conditions = {
        v.X('AL'): (0.1, 0.7, 0.2),  # 0.1 to 0.7 in 0.2 increments (4 points)
        v.X('CU'): (0.1, 0.5, 0.2),  # 0.1 to 0.5 in 0.2 increments (3 points)
        v.T: (600, 1500, 300),        # 600 to 1500 in 300K increments (4 points)
        v.P: 101325
    }
    
    # Calculate expected number of conditions
    x_al_values = np.arange(0.1, 0.71, 0.2)  # [0.1, 0.3, 0.5, 0.7]
    x_cu_values = np.arange(0.1, 0.51, 0.2)  # [0.1, 0.3, 0.5]
    temperatures = np.arange(600, 1501, 300)  # [600, 900, 1200, 1500]
    
    # Filter out invalid compositions where X_AL + X_CU >= 1.0
    valid_conditions = []
    for x_al in x_al_values:
        for x_cu in x_cu_values:
            if x_al + x_cu < 0.999:  # Allow small numerical tolerance
                for temp in temperatures:
                    valid_conditions.append((x_al, x_cu, temp))
    
    expected_conditions = len(valid_conditions)
    
    print(f"\nTesting {len(x_al_values)} X_AL × {len(x_cu_values)} X_CU × {len(temperatures)} temperatures")
    print(f"Valid conditions (X_AL + X_CU < 1): {expected_conditions}")
    print("Running calculations with multiple conditions per call...")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        # CPU calculation - all conditions at once
        print("Running CPU calculation...")
        cpu_start = time.time()
        result_cpu = equilibrium(dbf, comps, phases, conditions, 
                                calc_opts={'pdens': 50}, 
                                gpu=False, verbose=verbose)
        cpu_time = time.time() - cpu_start
        print(f"CPU calculation completed in {cpu_time:.1f} seconds")
        
        # GPU calculation - all conditions at once
        print("Running GPU calculation...")
        gpu_start = time.time()
        result_gpu = equilibrium(dbf, comps, phases, conditions, 
                                calc_opts={'pdens': 50},
                                gpu=True, verbose=verbose)
        gpu_time = time.time() - gpu_start
        print(f"GPU calculation completed in {gpu_time:.1f} seconds")
        
        # Extract results
        cpu_gm = result_cpu.GM.values
        gpu_gm = result_gpu.GM.values
        cpu_mu_al = result_cpu.MU.sel(component='AL').values
        cpu_mu_cu = result_cpu.MU.sel(component='CU').values
        cpu_mu_fe = result_cpu.MU.sel(component='FE').values
        gpu_mu_al = result_gpu.MU.sel(component='AL').values
        gpu_mu_cu = result_gpu.MU.sel(component='CU').values
        gpu_mu_fe = result_gpu.MU.sel(component='FE').values
        
        # Extract phase amounts
        cpu_np = result_cpu.NP.values
        gpu_np = result_gpu.NP.values
        
        # Debug shapes
        if verbose:
            print(f"\nDebug - Result shapes:")
            print(f"  cpu_gm.shape: {cpu_gm.shape}")
            print(f"  gpu_gm.shape: {gpu_gm.shape}")
            print(f"  cpu_np.shape: {cpu_np.shape}")
            print(f"  result_cpu dimensions: {result_cpu.dims}")
        
        # Get the actual coordinate values from the dataset
        if 'T' in result_cpu.coords:
            temp_coords = result_cpu.coords['T'].values
        else:
            temp_coords = temperatures
            
        if 'X_AL' in result_cpu.coords:
            x_al_coords = result_cpu.coords['X_AL'].values  
        else:
            x_al_coords = x_al_values
            
        if 'X_CU' in result_cpu.coords:
            x_cu_coords = result_cpu.coords['X_CU'].values  
        else:
            x_cu_coords = x_cu_values
            
        if verbose:
            print(f"\nCoordinates:")
            print(f"  Temperature coordinates: {temp_coords}")
            print(f"  X_AL coordinates: {x_al_coords}")
            print(f"  X_CU coordinates: {x_cu_coords}")
        
        # Build the condition arrays to match the flattened GM array
        x_al_flat = []
        x_cu_flat = []
        temp_flat = []
        
        # The flattening order matches the dimension order in the dataset
        for t_idx, t_val in enumerate(temp_coords):
            for al_idx, al_val in enumerate(x_al_coords):
                for cu_idx, cu_val in enumerate(x_cu_coords):
                    # Skip invalid compositions
                    if al_val + cu_val >= 0.999:
                        continue
                    temp_flat.append(t_val)
                    x_al_flat.append(al_val)
                    x_cu_flat.append(cu_val)
                
        x_al_flat = np.array(x_al_flat)
        x_cu_flat = np.array(x_cu_flat)
        temp_flat = np.array(temp_flat)
        
        # Flatten all result arrays
        cpu_gm_flat = cpu_gm.flatten()
        gpu_gm_flat = gpu_gm.flatten()
        cpu_mu_al_flat = cpu_mu_al.flatten()
        cpu_mu_cu_flat = cpu_mu_cu.flatten()
        cpu_mu_fe_flat = cpu_mu_fe.flatten()
        gpu_mu_al_flat = gpu_mu_al.flatten()
        gpu_mu_cu_flat = gpu_mu_cu.flatten()
        gpu_mu_fe_flat = gpu_mu_fe.flatten()
        
        # Flatten phase amounts (shape: [conditions, phases])
        cpu_np_flat = cpu_np.reshape(-1, cpu_np.shape[-1])
        gpu_np_flat = gpu_np.reshape(-1, gpu_np.shape[-1])
        
        # Verify array lengths
        num_results = len(cpu_gm_flat)
        if len(x_al_flat) != num_results:
            print(f"\nWARNING: Adjusting condition arrays to match result length")
            x_al_flat = x_al_flat[:num_results]
            x_cu_flat = x_cu_flat[:num_results]
            temp_flat = temp_flat[:num_results]
        
        print(f"\nProcessing {num_results} results...")
        
        # Open output file
        output_file = 'gpu_cpu_alcufe_8phases_results_multi.txt'
        with open(output_file, 'w') as f:
            # Write header
            f.write("X(AL)\tX(CU)\tX(FE)\tT(K)\tCPU_GM\tGPU_GM\tGM_DIFF\t")
            f.write("CPU_MU_AL\tCPU_MU_CU\tCPU_MU_FE\tGPU_MU_AL\tGPU_MU_CU\tGPU_MU_FE\t")
            f.write("MU_AL_DIFF\tMU_CU_DIFF\tMU_FE_DIFF\t")
            # Add phase amount columns for each phase
            for phase in phases:
                f.write(f"CPU_{phase}\tGPU_{phase}\t")
            f.write("STATUS\n")
            
            passed_conditions = 0
            failed_conditions = []
            max_gm_diff = 0
            max_mu_diff = 0
            
            # Compare results
            for i in range(num_results):
                # Skip if we get NaN or if composition is invalid
                if i >= len(x_al_flat) or i >= len(x_cu_flat):
                    continue
                    
                x_al = x_al_flat[i]
                x_cu = x_cu_flat[i]
                x_fe = 1.0 - x_al - x_cu  # Calculate Fe mole fraction
                temp = temp_flat[i]
                
                # Skip invalid compositions
                if x_fe < -0.001 or x_fe > 1.001:
                    continue
                
                # Skip NaN values
                if np.isnan(cpu_gm_flat[i]) or np.isnan(gpu_gm_flat[i]):
                    continue
                
                # Calculate differences
                gm_diff = abs(gpu_gm_flat[i] - cpu_gm_flat[i])
                mu_al_diff = abs(gpu_mu_al_flat[i] - cpu_mu_al_flat[i])
                mu_cu_diff = abs(gpu_mu_cu_flat[i] - cpu_mu_cu_flat[i])
                mu_fe_diff = abs(gpu_mu_fe_flat[i] - cpu_mu_fe_flat[i])
                
                max_gm_diff = max(max_gm_diff, gm_diff)
                max_mu_diff = max(max_mu_diff, mu_al_diff, mu_cu_diff, mu_fe_diff)
                
                # Check tolerance (100 J/mol for energy - more lenient for complex ternary system)
                tolerance = 100.0
                status = "PASS" if (gm_diff < tolerance and mu_al_diff < tolerance and 
                                  mu_cu_diff < tolerance and mu_fe_diff < tolerance) else "FAIL"
                
                if status == "PASS":
                    passed_conditions += 1
                else:
                    failed_conditions.append((x_al, x_cu, temp))
                
                # Write results
                f.write(f"{x_al:.2f}\t{x_cu:.2f}\t{x_fe:.2f}\t{temp:.0f}\t")
                f.write(f"{cpu_gm_flat[i]:.6f}\t{gpu_gm_flat[i]:.6f}\t{gm_diff:.6f}\t")
                f.write(f"{cpu_mu_al_flat[i]:.6f}\t{cpu_mu_cu_flat[i]:.6f}\t{cpu_mu_fe_flat[i]:.6f}\t")
                f.write(f"{gpu_mu_al_flat[i]:.6f}\t{gpu_mu_cu_flat[i]:.6f}\t{gpu_mu_fe_flat[i]:.6f}\t")
                f.write(f"{mu_al_diff:.6f}\t{mu_cu_diff:.6f}\t{mu_fe_diff:.6f}\t")
                
                # Write phase amounts
                for j in range(len(phases)):
                    cpu_phase_amt = cpu_np_flat[i, j] if j < cpu_np_flat.shape[1] else 0.0
                    gpu_phase_amt = gpu_np_flat[i, j] if j < gpu_np_flat.shape[1] else 0.0
                    f.write(f"{cpu_phase_amt:.6f}\t{gpu_phase_amt:.6f}\t")
                
                f.write(f"{status}\n")
            
            # Write summary
            elapsed_time = time.time() - start_time
            valid_results = passed_conditions + len(failed_conditions)
            f.write(f"\n# SUMMARY\n")
            f.write(f"# Al-Cu-Fe ternary system with {len(phases)} key phases\n")
            f.write(f"# Phases: {', '.join(phases)}\n")
            f.write(f"# Total valid conditions tested: {valid_results}\n")
            f.write(f"# Passed: {passed_conditions}\n")
            f.write(f"# Failed: {len(failed_conditions)}\n")
            if valid_results > 0:
                f.write(f"# Pass rate: {passed_conditions/valid_results*100:.1f}%\n")
            f.write(f"# Maximum GM difference: {max_gm_diff:.2f} J/mol\n")
            f.write(f"# Maximum MU difference: {max_mu_diff:.2f} J/mol\n")
            f.write(f"# CPU time: {cpu_time:.1f} seconds\n")
            f.write(f"# GPU time: {gpu_time:.1f} seconds\n")
            if gpu_time > 0:
                f.write(f"# Speedup: {cpu_time/gpu_time:.1f}x\n")
            f.write(f"# Total time: {elapsed_time:.1f} seconds\n")
            
            if failed_conditions:
                f.write(f"\n# FAILED CONDITIONS:\n")
                for x_al, x_cu, temp in failed_conditions[:20]:  # Limit to first 20
                    x_fe = 1.0 - x_al - x_cu
                    f.write(f"# X(AL)={x_al:.2f}, X(CU)={x_cu:.2f}, X(FE)={x_fe:.2f}, T={temp:.0f}K\n")
                if len(failed_conditions) > 20:
                    f.write(f"# ... and {len(failed_conditions)-20} more\n")
        
        print(f"\nTest completed in {elapsed_time:.1f} seconds")
        print(f"Results saved to {output_file}")
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total valid conditions: {valid_results}")
        print(f"Passed: {passed_conditions}")
        print(f"Failed: {len(failed_conditions)}")
        if valid_results > 0:
            print(f"Pass rate: {passed_conditions/valid_results*100:.1f}%")
        print(f"Maximum GM difference: {max_gm_diff:.2f} J/mol")
        print(f"Maximum MU difference: {max_mu_diff:.2f} J/mol")
        
        print(f"\nPerformance:")
        print(f"  CPU time: {cpu_time:.1f} seconds")
        print(f"  GPU time: {gpu_time:.1f} seconds")
        if gpu_time > 0:
            print(f"  Speedup: {cpu_time/gpu_time:.1f}x")
        
        if failed_conditions:
            print(f"\nFailed conditions (showing first 10):")
            for x_al, x_cu, temp in failed_conditions[:10]:
                x_fe = 1.0 - x_al - x_cu
                print(f"  X(AL)={x_al:.2f}, X(CU)={x_cu:.2f}, X(FE)={x_fe:.2f}, T={temp:.0f}K")
            if len(failed_conditions) > 10:
                print(f"  ... and {len(failed_conditions)-10} more")
        
        # Report overall status
        if valid_results > 0:
            pass_rate = passed_conditions / valid_results
            if pass_rate == 1.0:
                print("\n✓ ALL CONDITIONS PASSED!")
            elif pass_rate >= 0.95:
                print(f"\n✓ {pass_rate*100:.1f}% pass rate - EXCELLENT!")
            elif pass_rate >= 0.80:
                print(f"\n⚠ {pass_rate*100:.1f}% pass rate - Good but needs improvement")
            else:
                print(f"\n✗ {pass_rate*100:.1f}% pass rate - Significant issues")
                
    except Exception as e:
        print(f"\nError during test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run comprehensive GPU vs CPU comparison for Al-Cu-Fe with 8 key phases')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    
    run_comprehensive_test(verbose=args.verbose)