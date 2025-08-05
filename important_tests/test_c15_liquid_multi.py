#!/usr/bin/env python
"""Comprehensive GPU vs CPU comparison for C15 and LIQUID phases using multiple conditions per call."""

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
    """Run comprehensive GPU vs CPU comparison test with multiple conditions per call."""
    
    # Load database and set up calculation
    dbf = Database('AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    phases = ['LIQUID', 'AU2BI_C15']  # Only AU2BI_C15 and LIQUID phases
    
    # Define test conditions using the range syntax
    # This will create a grid of all combinations automatically
    conditions = {
        v.X('BI'): (0.1, 0.9, 0.1),  # 0.1 to 0.9 in 0.1 increments
        v.T: (400, 800, 100),        # 400 to 800 in 100 increments  
        v.P: 101325
    }
    
    # Calculate expected number of conditions
    # Note: pycalphad's range syntax (start, stop, step) is inclusive of stop
    x_bi_values = np.arange(0.1, 0.9, 0.1)  # This gives [0.1, 0.2, ..., 0.8]
    temperatures = np.arange(400, 801, 100)  # This gives [400, 500, ..., 800]
    expected_conditions = 8 * 5  # 8 compositions * 5 temperatures = 40
    
    print(f"Testing AU2BI_C15 + LIQUID phases")
    print(f"Testing {len(x_bi_values)} compositions x {len(temperatures)} temperatures = {expected_conditions} total conditions")
    print("Running calculations with multiple conditions per call...")
    
    start_time = time.time()
    
    try:
        # CPU calculation - all conditions at once
        print("Running CPU calculation...")
        cpu_start = time.time()
        result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=verbose)
        cpu_time = time.time() - cpu_start
        print(f"CPU calculation completed in {cpu_time:.1f} seconds")
        
        # GPU calculation - all conditions at once
        print("Running GPU calculation...")
        gpu_start = time.time()
        result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=verbose)
        gpu_time = time.time() - gpu_start
        print(f"GPU calculation completed in {gpu_time:.1f} seconds")
        
        # Extract results
        cpu_gm = result_cpu.GM.values
        gpu_gm = result_gpu.GM.values
        cpu_mu_au = result_cpu.MU.sel(component='AU').values
        cpu_mu_bi = result_cpu.MU.sel(component='BI').values
        gpu_mu_au = result_gpu.MU.sel(component='AU').values
        gpu_mu_bi = result_gpu.MU.sel(component='BI').values
        
        # Debug shapes first
        print(f"\nDebug - Result shapes:")
        print(f"  cpu_gm.shape: {cpu_gm.shape}")
        print(f"  gpu_gm.shape: {gpu_gm.shape}")
        print(f"  result_cpu.X shape: {result_cpu.X.shape}")
        print(f"  result_cpu.T shape: {result_cpu.T.shape}")
        print(f"  result_cpu dimensions: {result_cpu.dims}")
        
        # Get the dimensions properly
        # Results are organized with dims like ('N', 'P', 'T', 'X_BI', 'vertex')
        # GM has shape (N, P, T, X_BI) - one value per condition
        # X has shape (N, P, T, X_BI, vertex, component) - composition at each vertex
        
        print(f"\nExtracting condition values from dataset...")
        print(f"  Dimensions in order: {list(result_cpu.dims.keys())}")
        
        # Get the actual T and X_BI coordinate values from the dataset
        # These are the grid points, not the equilibrium compositions
        if 'T' in result_cpu.coords:
            temp_coords = result_cpu.coords['T'].values
        else:
            temp_coords = temperatures
            
        if 'X_BI' in result_cpu.coords:
            x_bi_coords = result_cpu.coords['X_BI'].values  
        else:
            x_bi_coords = x_bi_values
            
        print(f"  Temperature coordinates: {temp_coords}")
        print(f"  X_BI coordinates: {x_bi_coords}")
        
        # Create meshgrid of all condition combinations
        # The order depends on how pycalphad organizes the dimensions
        num_results = len(temp_coords) * len(x_bi_coords)
        
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
        
        print(f"  Constructed {len(x_bi_flat)} condition pairs")
        print(f"  cpu_gm flattened shape: {cpu_gm.flatten().shape}")
        
        # Flatten all result arrays
        cpu_gm_flat = cpu_gm.flatten()
        gpu_gm_flat = gpu_gm.flatten()
        cpu_mu_au_flat = cpu_mu_au.flatten()
        cpu_mu_bi_flat = cpu_mu_bi.flatten()
        gpu_mu_au_flat = gpu_mu_au.flatten()
        gpu_mu_bi_flat = gpu_mu_bi.flatten()
        
        # Verify all arrays have the same length
        if len(cpu_gm_flat) != len(x_bi_flat):
            print(f"\nWARNING: Length mismatch - adjusting to GM array length")
            print(f"  cpu_gm_flat length: {len(cpu_gm_flat)}")
            print(f"  x_bi_flat length: {len(x_bi_flat)}")
            # Truncate to match GM length
            x_bi_flat = x_bi_flat[:len(cpu_gm_flat)]
            temp_flat = temp_flat[:len(cpu_gm_flat)]
        
        print(f"\nFlattened array lengths:")
        print(f"  cpu_gm_flat: {len(cpu_gm_flat)}")
        print(f"  x_bi_flat: {len(x_bi_flat)}")
        print(f"  temp_flat: {len(temp_flat)}")
        
        # Check that we have the expected number of results
        print(f"\nReceived {num_results} results (expected {expected_conditions})")
        
        # Get phase information
        cpu_phases = result_cpu.Phase
        gpu_phases = result_gpu.Phase
        cpu_np = result_cpu.NP
        gpu_np = result_gpu.NP
        
        # Open output file
        with open('gpu_cpu_c15_results.txt', 'w') as f:
            # Write header
            f.write("X(BI)\tT(K)\tCPU_GM\tGPU_GM\tGM_DIFF\tCPU_Phases\tGPU_Phases\tPhases_Match\tSTATUS\n")
            
            passed_conditions = 0
            failed_conditions = []
            
            # Compare results
            for i in range(num_results):
                x_bi = x_bi_flat[i]
                temp = temp_flat[i]
                
                # Skip NaN values
                if np.isnan(cpu_gm_flat[i]) or np.isnan(gpu_gm_flat[i]):
                    continue
                
                # Calculate differences
                gm_diff = abs(gpu_gm_flat[i] - cpu_gm_flat[i])
                
                # Get phase information for this condition
                # Need to find the correct indices in the multi-dimensional arrays
                t_idx = np.where(temp_coords == temp)[0][0]
                x_idx = np.where(x_bi_coords == x_bi)[0][0]
                
                # Extract phase names and amounts
                cpu_phase_names = []
                cpu_phase_amounts = []
                gpu_phase_names = []
                gpu_phase_amounts = []
                
                # Assuming the last dimension is vertex (phase index)
                for vertex in range(cpu_phases.shape[-1]):
                    cpu_phase = str(cpu_phases.isel(T=t_idx, X_BI=x_idx, vertex=vertex).values)
                    gpu_phase = str(gpu_phases.isel(T=t_idx, X_BI=x_idx, vertex=vertex).values)
                    
                    cpu_amt = float(cpu_np.isel(T=t_idx, X_BI=x_idx, vertex=vertex).values)
                    gpu_amt = float(gpu_np.isel(T=t_idx, X_BI=x_idx, vertex=vertex).values)
                    
                    if cpu_phase != '' and not np.isnan(cpu_amt) and cpu_amt > 1e-6:
                        cpu_phase_names.append(f"{cpu_phase}({cpu_amt:.3f})")
                    if gpu_phase != '' and not np.isnan(gpu_amt) and gpu_amt > 1e-6:
                        gpu_phase_names.append(f"{gpu_phase}({gpu_amt:.3f})")
                
                cpu_phases_str = ','.join(cpu_phase_names) if cpu_phase_names else 'None'
                gpu_phases_str = ','.join(gpu_phase_names) if gpu_phase_names else 'None'
                phases_match = cpu_phases_str == gpu_phases_str
                
                # Check tolerance (1 J/mol for energy)
                tolerance = 1.0
                status = "PASS" if (gm_diff < tolerance and phases_match) else "FAIL"
                
                if status == "PASS":
                    passed_conditions += 1
                else:
                    failed_conditions.append((x_bi, temp))
                
                # Write results
                f.write(f"{x_bi:.1f}\t{temp:.0f}\t{cpu_gm_flat[i]:.12f}\t{gpu_gm_flat[i]:.12f}\t{gm_diff:.12f}\t")
                f.write(f"{cpu_phases_str}\t{gpu_phases_str}\t{phases_match}\t{status}\n")
            
            # Write summary
            elapsed_time = time.time() - start_time
            f.write(f"\n# SUMMARY\n")
            f.write(f"# Total conditions tested: {num_results}\n")
            f.write(f"# Passed: {passed_conditions}\n")
            f.write(f"# Failed: {num_results - passed_conditions}\n")
            f.write(f"# Pass rate: {passed_conditions/num_results*100:.1f}%\n")
            f.write(f"# CPU time: {cpu_time:.1f} seconds\n")
            f.write(f"# GPU time: {gpu_time:.1f} seconds\n")
            f.write(f"# Speedup: {cpu_time/gpu_time:.1f}x\n")
            f.write(f"# Total time: {elapsed_time:.1f} seconds\n")
            
            if failed_conditions:
                f.write(f"\n# FAILED CONDITIONS:\n")
                for x_bi, temp in failed_conditions:
                    f.write(f"# X(BI)={x_bi:.1f}, T={temp:.0f}K\n")
        
        print(f"\nTest completed in {elapsed_time:.1f} seconds")
        print(f"Results saved to gpu_cpu_c15_results.txt")
        print(f"\nSummary:")
        print(f"  Total conditions: {num_results}")
        print(f"  Passed: {passed_conditions}")
        print(f"  Failed: {num_results - passed_conditions}")
        print(f"  Pass rate: {passed_conditions/num_results*100:.1f}%")
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
    parser = argparse.ArgumentParser(description='Run comprehensive GPU vs CPU comparison test for C15 and LIQUID phases')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    
    run_comprehensive_test(verbose=args.verbose)