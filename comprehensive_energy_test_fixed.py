#!/usr/bin/env python
"""Comprehensive CPU vs GPU comparison across composition and temperature ranges."""

import os
import sys
import numpy as np
import contextlib
import io
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test ranges
x_ti_values = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
temperatures = [500, 600, 700, 800, 900, 1000]
pressure = 101325  # Standard pressure

results = []

print(f"Testing {len(x_ti_values)} compositions × {len(temperatures)} temperatures = {len(x_ti_values) * len(temperatures)} conditions")
print("This may take several minutes...")

total_tests = len(x_ti_values) * len(temperatures)
completed = 0

for x_ti in x_ti_values:
    for temp in temperatures:
        completed += 1
        print(f"Progress: {completed}/{total_tests} - X(TI)={x_ti:.2f}, T={temp}K", end="", flush=True)
        
        conditions = {v.X('TI'): x_ti, v.T: temp, v.P: pressure}
        
        try:
            # CPU calculation (suppress output)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
            
            cpu_gm = result_cpu.GM.values.flatten()[0]
            cpu_mu = result_cpu.MU.values.flatten()  # Chemical potentials
            
            # GPU calculation (suppress output)  
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
            
            gpu_gm = result_gpu.GM.values.flatten()[0]
            gpu_mu = result_gpu.MU.values.flatten()  # Chemical potentials
            
            # Calculate errors
            gm_abs_error = abs(cpu_gm - gpu_gm)
            mu_abs_errors = [abs(c - g) for c, g in zip(cpu_mu, gpu_mu)]
            max_mu_error = max(mu_abs_errors) if mu_abs_errors else 0
            
            result = {
                'x_ti': x_ti,
                'temp': temp,
                'cpu_gm': cpu_gm,
                'gpu_gm': gpu_gm,
                'gm_abs_error': gm_abs_error,
                'cpu_mu': cpu_mu.tolist(),
                'gpu_mu': gpu_mu.tolist(),
                'mu_abs_errors': mu_abs_errors,
                'max_mu_error': max_mu_error,
                'success': True
            }
            
            print(f" - GM error: {gm_abs_error:.3f} J/mol")
            
        except Exception as e:
            result = {
                'x_ti': x_ti,
                'temp': temp,
                'success': False,
                'error': str(e)
            }
            print(f" - ERROR: {str(e)[:50]}")
        
        results.append(result)

# Save results to file
print(f"\nSaving results to /tmp/output.txt...")

with open('/tmp/output.txt', 'w') as f:
    f.write("CPU vs GPU Free Energy and Chemical Potential Comparison\n")
    f.write("=" * 80 + "\n\n")
    
    # Header
    f.write(f"{'X(TI)':<6} {'T(K)':<6} {'CPU_GM':<12} {'GPU_GM':<12} {'GM_Error':<10} {'MU_Error':<10} {'Status'}\n")
    f.write("-" * 80 + "\n")
    
    successful_results = []
    failed_results = []
    
    for r in results:
        if r['success']:
            f.write(f"{r['x_ti']:<6.2f} {r['temp']:<6} {r['cpu_gm']:<12.3f} {r['gpu_gm']:<12.3f} {r['gm_abs_error']:<10.3f} {r['max_mu_error']:<10.3f} SUCCESS\n")
            successful_results.append(r)
        else:
            f.write(f"{r['x_ti']:<6.2f} {r['temp']:<6} {'ERROR':<12} {'ERROR':<12} {'N/A':<10} {'N/A':<10} FAILED\n")
            failed_results.append(r)
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("DETAILED ANALYSIS\n")
    f.write("=" * 80 + "\n\n")
    
    if successful_results:
        gm_errors = [r['gm_abs_error'] for r in successful_results]
        mu_errors = [r['max_mu_error'] for r in successful_results]
        
        f.write(f"Total successful tests: {len(successful_results)}\n")
        f.write(f"Total failed tests: {len(failed_results)}\n\n")
        
        f.write("FREE ENERGY (GM) ERRORS:\n")
        f.write(f"  Minimum GM error: {min(gm_errors):.6f} J/mol\n")
        f.write(f"  Maximum GM error: {max(gm_errors):.6f} J/mol\n")
        f.write(f"  Average GM error: {np.mean(gm_errors):.6f} J/mol\n")
        f.write(f"  Std dev GM error: {np.std(gm_errors):.6f} J/mol\n\n")
        
        f.write("CHEMICAL POTENTIAL (MU) ERRORS:\n")
        f.write(f"  Minimum MU error: {min(mu_errors):.6f} J/mol\n")
        f.write(f"  Maximum MU error: {max(mu_errors):.6f} J/mol\n")
        f.write(f"  Average MU error: {np.mean(mu_errors):.6f} J/mol\n")
        f.write(f"  Std dev MU error: {np.std(mu_errors):.6f} J/mol\n\n")
        
        # Error distribution
        large_gm_errors = [r for r in successful_results if r['gm_abs_error'] > 10.0]
        large_mu_errors = [r for r in successful_results if r['max_mu_error'] > 100.0]
        
        f.write("ERROR DISTRIBUTION:\n")
        f.write(f"  GM errors > 10 J/mol: {len(large_gm_errors)} cases\n")
        f.write(f"  MU errors > 100 J/mol: {len(large_mu_errors)} cases\n\n")
        
        if large_gm_errors:
            f.write("LARGE GM ERRORS (> 10 J/mol):\n")
            for r in large_gm_errors:
                f.write(f"  X(TI)={r['x_ti']:.2f}, T={r['temp']}K: {r['gm_abs_error']:.3f} J/mol\n")
            f.write("\n")
        
        if large_mu_errors:
            f.write("LARGE MU ERRORS (> 100 J/mol):\n")
            for r in large_mu_errors:
                f.write(f"  X(TI)={r['x_ti']:.2f}, T={r['temp']}K: {r['max_mu_error']:.3f} J/mol\n")
            f.write("\n")
        
        # Representative values
        f.write("REPRESENTATIVE VALUES:\n")
        f.write(f"{'X(TI)':<6} {'T(K)':<6} {'CPU_GM':<15} {'GPU_GM':<15} {'CPU_MU_NB':<15} {'GPU_MU_NB':<15}\n")
        f.write("-" * 90 + "\n")
        
        # Show a few representative cases
        sample_indices = [0, len(successful_results)//4, len(successful_results)//2, 3*len(successful_results)//4, -1]
        for i in sample_indices:
            if i < len(successful_results):
                r = successful_results[i]
                cpu_mu_nb = r['cpu_mu'][0] if len(r['cpu_mu']) > 0 else 0
                gpu_mu_nb = r['gpu_mu'][0] if len(r['gpu_mu']) > 0 else 0
                f.write(f"{r['x_ti']:<6.2f} {r['temp']:<6} {r['cpu_gm']:<15.3f} {r['gpu_gm']:<15.3f} {cpu_mu_nb:<15.3f} {gpu_mu_nb:<15.3f}\n")
    
    if failed_results:
        f.write(f"\nFAILED TESTS ({len(failed_results)}):\n")
        for r in failed_results:
            f.write(f"  X(TI)={r['x_ti']:.2f}, T={r['temp']}K: {r['error']}\n")

print("Results saved!")
print("\n" + "="*80)