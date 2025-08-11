#!/usr/bin/env python
"""Comprehensive GPU vs CPU comparison for Al-Cu-Fe ternary system with all phases enabled."""

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
    
    print(f"Available phases ({len(phases)}): {phases}")
    
    # Define test conditions
    # For ternary system, we need two composition conditions
    # Create arrays of conditions
    n_conditions = 20
    
    # Create a variety of conditions
    x_cu_values = np.linspace(0.05, 0.45, n_conditions)
    x_fe_values = np.linspace(0.05, 0.45, n_conditions)
    temperatures = np.linspace(600, 1400, n_conditions)
    
    # Ensure Al fraction stays positive
    # Adjust Fe values to ensure Cu + Fe < 0.95
    for i in range(n_conditions):
        if x_cu_values[i] + x_fe_values[i] > 0.95:
            x_fe_values[i] = 0.95 - x_cu_values[i]
    
    conditions = {
        v.T: temperatures,
        v.P: 101325 * np.ones(n_conditions),
        v.X('CU'): x_cu_values,
        v.X('FE'): x_fe_values
    }
    
    print(f"\nTesting {n_conditions} conditions:")
    print(f"  X(CU) range: {x_cu_values.min():.3f} to {x_cu_values.max():.3f}")
    print(f"  X(FE) range: {x_fe_values.min():.3f} to {x_fe_values.max():.3f}")
    print(f"  X(AL) range: {(1-x_cu_values-x_fe_values).min():.3f} to {(1-x_cu_values-x_fe_values).max():.3f}")
    print(f"  T range: {temperatures.min():.0f}K to {temperatures.max():.0f}K")
    print("\nRunning calculations...")
    
    start_time = time.time()
    
    try:
        # CPU calculation
        print("Running CPU calculation...")
        cpu_start = time.time()
        result_cpu = equilibrium(dbf, comps, phases, conditions, 
                                gpu=False, verbose=verbose, 
                                calc_opts={'pdens': 60})
        cpu_time = time.time() - cpu_start
        print(f"CPU calculation completed in {cpu_time:.1f} seconds")
        
        # GPU calculation
        print("Running GPU calculation...")
        gpu_start = time.time()
        result_gpu = equilibrium(dbf, comps, phases, conditions, 
                                gpu=True, verbose=verbose,
                                calc_opts={'pdens': 60})
        gpu_time = time.time() - gpu_start
        print(f"GPU calculation completed in {gpu_time:.1f} seconds")
        
        # Extract results
        cpu_gm = result_cpu.GM.values.flatten()
        gpu_gm = result_gpu.GM.values.flatten()
        
        # Extract chemical potentials
        cpu_mu_al = result_cpu.MU.sel(component='AL').values.flatten()
        cpu_mu_cu = result_cpu.MU.sel(component='CU').values.flatten()
        cpu_mu_fe = result_cpu.MU.sel(component='FE').values.flatten()
        
        gpu_mu_al = result_gpu.MU.sel(component='AL').values.flatten()
        gpu_mu_cu = result_gpu.MU.sel(component='CU').values.flatten()
        gpu_mu_fe = result_gpu.MU.sel(component='FE').values.flatten()
        
        print(f"\nComparing results...")
        
        # Open output file
        with open('gpu_cpu_alcufe_all_phases_results.txt', 'w') as f:
            # Write header
            f.write("Condition\tX(CU)\tX(FE)\tX(AL)\tT(K)\tCPU_GM\tGPU_GM\tGM_DIFF\t")
            f.write("CPU_MU_AL\tGPU_MU_AL\tMU_AL_DIFF\t")
            f.write("CPU_MU_CU\tGPU_MU_CU\tMU_CU_DIFF\t")
            f.write("CPU_MU_FE\tGPU_MU_FE\tMU_FE_DIFF\tSTATUS\n")
            
            passed_conditions = 0
            failed_conditions = []
            
            # Compare results
            for i in range(n_conditions):
                x_cu = x_cu_values[i]
                x_fe = x_fe_values[i]
                x_al = 1.0 - x_cu - x_fe
                temp = temperatures[i]
                
                # Skip NaN values
                if np.isnan(cpu_gm[i]) or np.isnan(gpu_gm[i]):
                    continue
                
                # Calculate differences
                gm_diff = abs(gpu_gm[i] - cpu_gm[i])
                mu_al_diff = abs(gpu_mu_al[i] - cpu_mu_al[i])
                mu_cu_diff = abs(gpu_mu_cu[i] - cpu_mu_cu[i])
                mu_fe_diff = abs(gpu_mu_fe[i] - cpu_mu_fe[i])
                
                # Check tolerance (1 J/mol for energy)
                tolerance = 1.0
                status = "PASS" if (gm_diff < tolerance and 
                                  mu_al_diff < tolerance and 
                                  mu_cu_diff < tolerance and 
                                  mu_fe_diff < tolerance) else "FAIL"
                
                if status == "PASS":
                    passed_conditions += 1
                else:
                    failed_conditions.append(i)
                
                # Write results
                f.write(f"{i+1}\t{x_cu:.3f}\t{x_fe:.3f}\t{x_al:.3f}\t{temp:.0f}\t")
                f.write(f"{cpu_gm[i]:.6f}\t{gpu_gm[i]:.6f}\t{gm_diff:.6f}\t")
                f.write(f"{cpu_mu_al[i]:.6f}\t{gpu_mu_al[i]:.6f}\t{mu_al_diff:.6f}\t")
                f.write(f"{cpu_mu_cu[i]:.6f}\t{gpu_mu_cu[i]:.6f}\t{mu_cu_diff:.6f}\t")
                f.write(f"{cpu_mu_fe[i]:.6f}\t{gpu_mu_fe[i]:.6f}\t{mu_fe_diff:.6f}\t")
                f.write(f"{status}\n")
                
                # Also print to console for immediate feedback
                if verbose or status == "FAIL":
                    print(f"Condition {i+1}: X(CU)={x_cu:.3f}, X(FE)={x_fe:.3f}, T={temp:.0f}K")
                    print(f"  GM: CPU={cpu_gm[i]:.2f}, GPU={gpu_gm[i]:.2f}, diff={gm_diff:.6f}")
                    print(f"  MU_AL: CPU={cpu_mu_al[i]:.2f}, GPU={gpu_mu_al[i]:.2f}, diff={mu_al_diff:.6f}")
                    print(f"  MU_CU: CPU={cpu_mu_cu[i]:.2f}, GPU={gpu_mu_cu[i]:.2f}, diff={mu_cu_diff:.6f}")
                    print(f"  MU_FE: CPU={cpu_mu_fe[i]:.2f}, GPU={gpu_mu_fe[i]:.2f}, diff={mu_fe_diff:.6f}")
                    print(f"  Status: {status}")
            
            # Write summary
            elapsed_time = time.time() - start_time
            f.write(f"\n# SUMMARY\n")
            f.write(f"# Total conditions tested: {n_conditions}\n")
            f.write(f"# Passed: {passed_conditions}\n")
            f.write(f"# Failed: {n_conditions - passed_conditions}\n")
            f.write(f"# Pass rate: {passed_conditions/n_conditions*100:.1f}%\n")
            f.write(f"# CPU time: {cpu_time:.1f} seconds\n")
            f.write(f"# GPU time: {gpu_time:.1f} seconds\n")
            f.write(f"# Speedup: {cpu_time/gpu_time:.1f}x\n")
            f.write(f"# Total time: {elapsed_time:.1f} seconds\n")
            
            if failed_conditions:
                f.write(f"\n# FAILED CONDITIONS:\n")
                for idx in failed_conditions:
                    f.write(f"# Condition {idx+1}: X(CU)={x_cu_values[idx]:.3f}, X(FE)={x_fe_values[idx]:.3f}, T={temperatures[idx]:.0f}K\n")
                    f.write(f"#   GM diff: {abs(gpu_gm[idx] - cpu_gm[idx]):.6f}\n")
                    f.write(f"#   MU_AL diff: {abs(gpu_mu_al[idx] - cpu_mu_al[idx]):.6f}\n")
                    f.write(f"#   MU_CU diff: {abs(gpu_mu_cu[idx] - cpu_mu_cu[idx]):.6f}\n")
                    f.write(f"#   MU_FE diff: {abs(gpu_mu_fe[idx] - cpu_mu_fe[idx]):.6f}\n")
        
        print(f"\n" + "="*60)
        print(f"TEST SUMMARY")
        print(f"="*60)
        print(f"Total conditions tested: {n_conditions}")
        print(f"Passed: {passed_conditions} ({passed_conditions/n_conditions*100:.1f}%)")
        print(f"Failed: {n_conditions - passed_conditions} ({(n_conditions-passed_conditions)/n_conditions*100:.1f}%)")
        print(f"CPU time: {cpu_time:.1f} seconds")
        print(f"GPU time: {gpu_time:.1f} seconds")
        print(f"Speedup: {cpu_time/gpu_time:.1f}x")
        print(f"Total test time: {elapsed_time:.1f} seconds")
        print(f"\nResults saved to: gpu_cpu_alcufe_all_phases_results.txt")
        
        # Also print first few results for quick check
        print(f"\nFirst 5 conditions:")
        print(f"{'Cond':<5} {'X(CU)':<6} {'X(FE)':<6} {'T(K)':<6} {'CPU GM':<12} {'GPU GM':<12} {'Diff':<10} {'Status':<8}")
        print("-" * 80)
        for i in range(min(5, n_conditions)):
            print(f"{i+1:<5} {x_cu_values[i]:<6.3f} {x_fe_values[i]:<6.3f} {temperatures[i]:<6.0f} "
                  f"{cpu_gm[i]:<12.2f} {gpu_gm[i]:<12.2f} {abs(gpu_gm[i]-cpu_gm[i]):<10.6f} "
                  f"{'PASS' if abs(gpu_gm[i]-cpu_gm[i]) < 1.0 else 'FAIL':<8}")
        
        return passed_conditions == n_conditions
        
    except Exception as e:
        print(f"\nERROR during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test GPU vs CPU for Al-Cu-Fe system')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    
    success = run_comprehensive_test(verbose=args.verbose)
    exit(0 if success else 1)