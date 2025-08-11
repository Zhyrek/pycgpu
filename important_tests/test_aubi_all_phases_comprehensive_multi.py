#!/usr/bin/env python
"""Comprehensive GPU vs CPU comparison for Au-Bi system with ALL phases using multiple conditions per call."""

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
    """Run comprehensive GPU vs CPU comparison test for Au-Bi with all phases."""
    
    # Load database and set up calculation
    dbf = Database('AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    
    # Use ALL phases from the database
    phases = ['LIQUID', 'FCC_A1', 'HCP_A3', 'RHOMBOHEDRAL_A7', 'AU2BI_C15', 'BCC_A2']
    
    print("=" * 60)
    print("Au-Bi SYSTEM COMPREHENSIVE TEST WITH ALL PHASES")
    print("=" * 60)
    print(f"Components: {comps}")
    print(f"Phases: {phases}")
    print(f"Number of phases: {len(phases)}")
    
    # Define test conditions using the range syntax
    # Create a comprehensive grid covering the full composition and temperature range
    conditions = {
        v.X('BI'): (0.1, 0.9, 0.1),  # 0.1 to 0.9 in 0.1 increments (8 points)
        v.T: (300, 1200, 150),        # 300 to 1200 in 150K increments (6 points)
        v.P: 101325
    }
    
    # Calculate expected number of conditions
    x_bi_values = np.arange(0.1, 0.91, 0.1)  # [0.1, 0.2, ..., 0.9]
    temperatures = np.arange(300, 1201, 150)  # [300, 450, 600, 750, 900, 1050, 1200]
    expected_conditions = len(x_bi_values) * len(temperatures)
    
    print(f"\nTesting {len(x_bi_values)} compositions x {len(temperatures)} temperatures = {expected_conditions} total conditions")
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
        cpu_mu_au = result_cpu.MU.sel(component='AU').values
        cpu_mu_bi = result_cpu.MU.sel(component='BI').values
        gpu_mu_au = result_gpu.MU.sel(component='AU').values
        gpu_mu_bi = result_gpu.MU.sel(component='BI').values
        
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
        
        # Get the actual T and X_BI coordinate values from the dataset
        if 'T' in result_cpu.coords:
            temp_coords = result_cpu.coords['T'].values
        else:
            temp_coords = temperatures
            
        if 'X_BI' in result_cpu.coords:
            x_bi_coords = result_cpu.coords['X_BI'].values  
        else:
            x_bi_coords = x_bi_values
            
        if verbose:
            print(f"\nCoordinates:")
            print(f"  Temperature coordinates: {temp_coords}")
            print(f"  X_BI coordinates: {x_bi_coords}")
        
        # Build the condition arrays to match the flattened GM array
        x_bi_flat = []
        temp_flat = []
        
        # The flattening order matches the dimension order in the dataset
        for t_idx, t_val in enumerate(temp_coords):
            for x_idx, x_val in enumerate(x_bi_coords):
                temp_flat.append(t_val)
                x_bi_flat.append(x_val)
                
        x_bi_flat = np.array(x_bi_flat)
        temp_flat = np.array(temp_flat)
        
        # Flatten all result arrays
        cpu_gm_flat = cpu_gm.flatten()
        gpu_gm_flat = gpu_gm.flatten()
        cpu_mu_au_flat = cpu_mu_au.flatten()
        cpu_mu_bi_flat = cpu_mu_bi.flatten()
        gpu_mu_au_flat = gpu_mu_au.flatten()
        gpu_mu_bi_flat = gpu_mu_bi.flatten()
        
        # Flatten phase amounts (shape: [conditions, phases])
        cpu_np_flat = cpu_np.reshape(-1, cpu_np.shape[-1])
        gpu_np_flat = gpu_np.reshape(-1, gpu_np.shape[-1])
        
        # Verify array lengths
        num_results = len(cpu_gm_flat)
        if len(x_bi_flat) != num_results:
            print(f"\nWARNING: Adjusting condition arrays to match result length")
            x_bi_flat = x_bi_flat[:num_results]
            temp_flat = temp_flat[:num_results]
        
        print(f"\nProcessing {num_results} results...")
        
        # Open output file
        output_file = 'gpu_cpu_aubi_all_phases_results_multi.txt'
        with open(output_file, 'w') as f:
            # Write header
            f.write("X(BI)\tX(AU)\tT(K)\tCPU_GM\tGPU_GM\tGM_DIFF\t")
            f.write("CPU_MU_AU\tCPU_MU_BI\tGPU_MU_AU\tGPU_MU_BI\tMU_AU_DIFF\tMU_BI_DIFF\t")
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
                x_bi = x_bi_flat[i]
                x_au = 1.0 - x_bi  # Calculate Au mole fraction
                temp = temp_flat[i]
                
                # Skip NaN values
                if np.isnan(cpu_gm_flat[i]) or np.isnan(gpu_gm_flat[i]):
                    continue
                
                # Calculate differences
                gm_diff = abs(gpu_gm_flat[i] - cpu_gm_flat[i])
                mu_au_diff = abs(gpu_mu_au_flat[i] - cpu_mu_au_flat[i])
                mu_bi_diff = abs(gpu_mu_bi_flat[i] - cpu_mu_bi_flat[i])
                
                max_gm_diff = max(max_gm_diff, gm_diff)
                max_mu_diff = max(max_mu_diff, mu_au_diff, mu_bi_diff)
                
                # Check tolerance (100 J/mol for energy - more lenient for complex system)
                tolerance = 100.0
                status = "PASS" if (gm_diff < tolerance and mu_au_diff < tolerance and mu_bi_diff < tolerance) else "FAIL"
                
                if status == "PASS":
                    passed_conditions += 1
                else:
                    failed_conditions.append((x_bi, temp))
                
                # Write results
                f.write(f"{x_bi:.1f}\t{x_au:.1f}\t{temp:.0f}\t")
                f.write(f"{cpu_gm_flat[i]:.6f}\t{gpu_gm_flat[i]:.6f}\t{gm_diff:.6f}\t")
                f.write(f"{cpu_mu_au_flat[i]:.6f}\t{cpu_mu_bi_flat[i]:.6f}\t")
                f.write(f"{gpu_mu_au_flat[i]:.6f}\t{gpu_mu_bi_flat[i]:.6f}\t")
                f.write(f"{mu_au_diff:.6f}\t{mu_bi_diff:.6f}\t")
                
                # Write phase amounts
                for j in range(len(phases)):
                    cpu_phase_amt = cpu_np_flat[i, j] if j < cpu_np_flat.shape[1] else 0.0
                    gpu_phase_amt = gpu_np_flat[i, j] if j < gpu_np_flat.shape[1] else 0.0
                    f.write(f"{cpu_phase_amt:.6f}\t{gpu_phase_amt:.6f}\t")
                
                f.write(f"{status}\n")
            
            # Write summary
            elapsed_time = time.time() - start_time
            f.write(f"\n# SUMMARY\n")
            f.write(f"# Au-Bi system with ALL {len(phases)} phases\n")
            f.write(f"# Phases: {', '.join(phases)}\n")
            f.write(f"# Total conditions tested: {num_results}\n")
            f.write(f"# Passed: {passed_conditions}\n")
            f.write(f"# Failed: {num_results - passed_conditions}\n")
            f.write(f"# Pass rate: {passed_conditions/num_results*100:.1f}%\n")
            f.write(f"# Maximum GM difference: {max_gm_diff:.2f} J/mol\n")
            f.write(f"# Maximum MU difference: {max_mu_diff:.2f} J/mol\n")
            f.write(f"# CPU time: {cpu_time:.1f} seconds\n")
            f.write(f"# GPU time: {gpu_time:.1f} seconds\n")
            f.write(f"# Speedup: {cpu_time/gpu_time:.1f}x\n")
            f.write(f"# Total time: {elapsed_time:.1f} seconds\n")
            
            if failed_conditions:
                f.write(f"\n# FAILED CONDITIONS:\n")
                for x_bi, temp in failed_conditions:
                    f.write(f"# X(BI)={x_bi:.1f}, T={temp:.0f}K\n")
        
        print(f"\nTest completed in {elapsed_time:.1f} seconds")
        print(f"Results saved to {output_file}")
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total conditions: {num_results}")
        print(f"Passed: {passed_conditions}")
        print(f"Failed: {num_results - passed_conditions}")
        print(f"Pass rate: {passed_conditions/num_results*100:.1f}%")
        print(f"Maximum GM difference: {max_gm_diff:.2f} J/mol")
        print(f"Maximum MU difference: {max_mu_diff:.2f} J/mol")
        
        print(f"\nPerformance:")
        print(f"  CPU time: {cpu_time:.1f} seconds")
        print(f"  GPU time: {gpu_time:.1f} seconds")
        print(f"  Speedup: {cpu_time/gpu_time:.1f}x")
        
        if failed_conditions:
            print(f"\nFailed conditions:")
            for x_bi, temp in failed_conditions[:10]:  # Show first 10
                print(f"  X(BI)={x_bi:.1f}, T={temp:.0f}K")
            if len(failed_conditions) > 10:
                print(f"  ... and {len(failed_conditions)-10} more")
        
        # Report overall status
        if passed_conditions == num_results:
            print("\n✓ ALL CONDITIONS PASSED!")
        elif passed_conditions / num_results >= 0.95:
            print(f"\n✓ {passed_conditions/num_results*100:.1f}% pass rate - EXCELLENT!")
        elif passed_conditions / num_results >= 0.80:
            print(f"\n⚠ {passed_conditions/num_results*100:.1f}% pass rate - Good but needs improvement")
        else:
            print(f"\n✗ {passed_conditions/num_results*100:.1f}% pass rate - Significant issues")
                
    except Exception as e:
        print(f"\nError during test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run comprehensive GPU vs CPU comparison for Au-Bi with ALL phases')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    
    run_comprehensive_test(verbose=args.verbose)