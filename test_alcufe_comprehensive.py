#!/usr/bin/env python
"""
Comprehensive test of Al-Cu-Fe system comparing CPU vs GPU calculations
Tests at 1000°C with compositions varying by 0.1 increments
"""

import pycalphad as cp
import numpy as np
import warnings
import time
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def test_alcufe_comprehensive():
    """Test CPU vs GPU for Al-Cu-Fe system at 1000°C"""
    
    # Load database
    db = cp.Database('Al-Cu-Fe.tdb')
    components = ['AL', 'CU', 'FE', 'VA']
    
    # Get all phases except VA
    all_phases = [phase for phase in db.phases.keys() if phase != 'VA']
    
    # Temperature fixed at 1000°C
    temperature = 1273.15  # K
    
    # Results storage
    results = []
    
    # Header
    print("Al-Cu-Fe System Comprehensive Test at 1000°C")
    print("=" * 100)
    print(f"Temperature: {temperature} K")
    print(f"Phases tested: {len(all_phases)} phases")
    print("=" * 100)
    
    # Table header
    print(f"{'X(AL)':<6} {'X(CU)':<6} {'X(FE)':<6} {'CPU_GM':<12} {'GPU_GM':<12} {'GM_DIFF':<10} {'CPU_MU_AL':<12} {'CPU_MU_CU':<12} {'CPU_MU_FE':<12} {'GPU_MU_AL':<12} {'GPU_MU_CU':<12} {'GPU_MU_FE':<12} {'STATUS':<8}")
    print("-" * 150)
    
    # Test grid - 0.1 increments
    al_fractions = np.arange(0, 1.1, 0.1)
    cu_fractions = np.arange(0, 1.1, 0.1)
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    skipped_tests = 0
    
    start_time = time.time()
    
    for x_al in al_fractions:
        for x_cu in cu_fractions:
            # Skip if sum > 1 (invalid composition)
            if x_al + x_cu > 1.0 + 1e-10:
                continue
                
            x_fe = 1.0 - x_al - x_cu
            
            # Skip if any component is negative (shouldn't happen with our grid)
            if x_fe < -1e-10:
                continue
            
            total_tests += 1
            
            # Set up conditions - only specify 2 mole fractions
            conditions = {
                cp.v.T: temperature,
                cp.v.P: 101325
            }
            
            # Only add mole fractions if not at endpoints
            if x_al > 1e-10 and x_al < 1-1e-10:
                conditions[cp.v.X('AL')] = x_al
            if x_cu > 1e-10 and x_cu < 1-1e-10:
                conditions[cp.v.X('CU')] = x_cu
                
            # Skip pure components (need at least 2 components)
            if len(conditions) < 3:  # T, P + at least one X
                skipped_tests += 1
                continue
            
            # CPU calculation
            cpu_gm = None
            cpu_mu = [None, None, None]
            cpu_success = False
            
            try:
                eq_cpu = cp.equilibrium(db, components, all_phases, conditions, 
                                       verbose=False, debug=False)
                cpu_gm = float(eq_cpu.GM.values.flat[0])
                
                # Extract chemical potentials in order AL, CU, FE
                mu_values = eq_cpu.MU.values.flat
                if len(mu_values) >= 3:
                    cpu_mu[0] = float(mu_values[0])  # AL
                    cpu_mu[1] = float(mu_values[1])  # CU
                    cpu_mu[2] = float(mu_values[2])  # FE
                    
                cpu_success = True
                
            except Exception as e:
                # CPU failed
                pass
            
            # GPU calculation
            gpu_gm = None
            gpu_mu = [None, None, None]
            gpu_success = False
            
            try:
                eq_gpu = cp.equilibrium(db, components, all_phases, conditions, 
                                       gpu=True, verbose=False, debug=False)
                gpu_gm = float(eq_gpu.GM.values.flat[0])
                
                # Extract chemical potentials
                mu_values = eq_gpu.MU.values.flat
                if len(mu_values) >= 3:
                    gpu_mu[0] = float(mu_values[0])  # AL
                    gpu_mu[1] = float(mu_values[1])  # CU
                    gpu_mu[2] = float(mu_values[2])  # FE
                    
                gpu_success = True
                
            except Exception as e:
                # GPU failed
                pass
            
            # Compare results
            if cpu_success and gpu_success:
                gm_diff = abs(cpu_gm - gpu_gm)
                
                if gm_diff < 1.0:
                    status = "PASS"
                    passed_tests += 1
                else:
                    status = "FAIL"
                    failed_tests += 1
                    
                # Format output
                print(f"{x_al:<6.1f} {x_cu:<6.1f} {x_fe:<6.1f} "
                      f"{cpu_gm:<12.3f} {gpu_gm:<12.3f} {gm_diff:<10.6f} "
                      f"{cpu_mu[0]:<12.3f} {cpu_mu[1]:<12.3f} {cpu_mu[2]:<12.3f} "
                      f"{gpu_mu[0]:<12.3f} {gpu_mu[1]:<12.3f} {gpu_mu[2]:<12.3f} "
                      f"{status:<8}")
                      
                results.append({
                    'x_al': x_al,
                    'x_cu': x_cu,
                    'x_fe': x_fe,
                    'cpu_gm': cpu_gm,
                    'gpu_gm': gpu_gm,
                    'gm_diff': gm_diff,
                    'cpu_mu': cpu_mu,
                    'gpu_mu': gpu_mu,
                    'status': status
                })
                
            elif cpu_success and not gpu_success:
                print(f"{x_al:<6.1f} {x_cu:<6.1f} {x_fe:<6.1f} "
                      f"{cpu_gm:<12.3f} {'GPU_FAIL':<12} {'N/A':<10} "
                      f"{cpu_mu[0]:<12.3f} {cpu_mu[1]:<12.3f} {cpu_mu[2]:<12.3f} "
                      f"{'N/A':<12} {'N/A':<12} {'N/A':<12} "
                      f"{'GPU_ERR':<8}")
                failed_tests += 1
                
            elif not cpu_success and gpu_success:
                print(f"{x_al:<6.1f} {x_cu:<6.1f} {x_fe:<6.1f} "
                      f"{'CPU_FAIL':<12} {gpu_gm:<12.3f} {'N/A':<10} "
                      f"{'N/A':<12} {'N/A':<12} {'N/A':<12} "
                      f"{gpu_mu[0]:<12.3f} {gpu_mu[1]:<12.3f} {gpu_mu[2]:<12.3f} "
                      f"{'CPU_ERR':<8}")
                failed_tests += 1
                
            else:
                print(f"{x_al:<6.1f} {x_cu:<6.1f} {x_fe:<6.1f} "
                      f"{'BOTH_FAIL':<12} {'BOTH_FAIL':<12} {'N/A':<10} "
                      f"{'N/A':<12} {'N/A':<12} {'N/A':<12} "
                      f"{'N/A':<12} {'N/A':<12} {'N/A':<12} "
                      f"{'BOTH_ERR':<8}")
                failed_tests += 1
            
            sys.stdout.flush()
    
    end_time = time.time()
    
    # Summary statistics
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Total conditions tested: {total_tests}")
    print(f"Skipped (pure components): {skipped_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    
    if passed_tests + failed_tests > 0:
        print(f"Pass rate: {100.0 * passed_tests / (passed_tests + failed_tests):.1f}%")
    
    print(f"Total time: {end_time - start_time:.1f} seconds")
    
    # Error analysis for successful comparisons
    if results:
        gm_errors = [r['gm_diff'] for r in results if r['status'] in ['PASS', 'FAIL']]
        if gm_errors:
            print(f"\nError Statistics (GM):")
            print(f"  Mean error: {np.mean(gm_errors):.6f} J/mol")
            print(f"  Max error: {np.max(gm_errors):.6f} J/mol")
            print(f"  Min error: {np.min(gm_errors):.6f} J/mol")
            print(f"  Std dev: {np.std(gm_errors):.6f} J/mol")
            
            # Check against 806 J/mol benchmark
            errors_above_806 = [e for e in gm_errors if e > 806]
            if errors_above_806:
                print(f"\nErrors above 806 J/mol benchmark: {len(errors_above_806)}")
                print(f"  Worst case: {max(errors_above_806):.1f} J/mol ({max(errors_above_806)/806:.1f}x worse)")
            else:
                print(f"\nAll errors below 806 J/mol benchmark!")
                if np.max(gm_errors) > 0:
                    print(f"  Improvement: {806/np.max(gm_errors):.1f}x better")
    
    # Write detailed results to file
    with open('alcufe_comprehensive_results.txt', 'w') as f:
        f.write("X(AL)\tX(CU)\tX(FE)\tT(K)\tCPU_GM\tGPU_GM\tGM_DIFF\tCPU_MU_AL\tCPU_MU_CU\tCPU_MU_FE\tGPU_MU_AL\tGPU_MU_CU\tGPU_MU_FE\tSTATUS\n")
        for r in results:
            f.write(f"{r['x_al']:.1f}\t{r['x_cu']:.1f}\t{r['x_fe']:.1f}\t{temperature}\t")
            f.write(f"{r['cpu_gm']:.6f}\t{r['gpu_gm']:.6f}\t{r['gm_diff']:.6f}\t")
            f.write(f"{r['cpu_mu'][0]:.6f}\t{r['cpu_mu'][1]:.6f}\t{r['cpu_mu'][2]:.6f}\t")
            f.write(f"{r['gpu_mu'][0]:.6f}\t{r['gpu_mu'][1]:.6f}\t{r['gpu_mu'][2]:.6f}\t")
            f.write(f"{r['status']}\n")
    
    print(f"\nDetailed results written to: alcufe_comprehensive_results.txt")

if __name__ == "__main__":
    test_alcufe_comprehensive()