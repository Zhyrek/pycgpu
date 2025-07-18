#!/usr/bin/env python
"""Working CPU vs GPU energy comparison that saves results to /tmp/output.txt."""

import os
import sys
import numpy as np
import warnings
sys.path.insert(0, os.getcwd())

# Suppress all warnings
warnings.filterwarnings('ignore')

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test conditions
test_conditions = [
    {'x_ti': 0.05, 'temp': 800},
    {'x_ti': 0.1, 'temp': 1000},
    {'x_ti': 0.5, 'temp': 700},
    {'x_ti': 0.5, 'temp': 1000},
    {'x_ti': 0.9, 'temp': 800},
]

results = []

print("Running CPU vs GPU energy comparison...")

for i, test in enumerate(test_conditions):
    x_ti, temp = test['x_ti'], test['temp']
    print(f"Test {i+1}/{len(test_conditions)}: X(TI)={x_ti:.2f}, T={temp}K")
    
    conditions = {v.X('TI'): x_ti, v.T: temp, v.P: 101325}
    
    try:
        # CPU calculation - use os.devnull redirection
        import subprocess
        import sys
        from io import StringIO
        
        # Capture stdout/stderr but don't suppress completely since we need to see errors
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        try:
            # CPU calculation
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            cpu_gm = result_cpu.GM.values.flatten()[0]
            cpu_mu = result_cpu.MU.values.flatten()
            
            # GPU calculation  
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            gpu_gm = result_gpu.GM.values.flatten()[0]
            gpu_mu = result_gpu.MU.values.flatten()
            
            # Calculate errors
            gm_error = abs(cpu_gm - gpu_gm)
            mu_errors = [abs(c - g) for c, g in zip(cpu_mu, gpu_mu)]
            max_mu_error = max(mu_errors) if mu_errors else 0
            
            results.append({
                'x_ti': x_ti,
                'temp': temp,
                'cpu_gm': cpu_gm,
                'gpu_gm': gpu_gm,
                'gm_error': gm_error,
                'cpu_mu': cpu_mu.tolist(),
                'gpu_mu': gpu_mu.tolist(),
                'max_mu_error': max_mu_error,
                'success': True
            })
            
            print(f"  CPU_GM: {cpu_gm:.3f}, GPU_GM: {gpu_gm:.3f}, Error: {gm_error:.3f} J/mol")
            
        except Exception as e:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            raise e
            
    except Exception as e:
        results.append({
            'x_ti': x_ti,
            'temp': temp,
            'success': False,
            'error': str(e)
        })
        print(f"  ERROR: {str(e)}")

# Save results to /tmp/output.txt
print(f"\nSaving results to /tmp/output.txt...")

with open('/tmp/output.txt', 'w') as f:
    f.write("CPU vs GPU Free Energy and Chemical Potential Comparison\n")
    f.write("=" * 80 + "\n\n")
    
    # Summary table
    f.write(f"{'X(TI)':<6} {'T(K)':<6} {'CPU_GM':<12} {'GPU_GM':<12} {'GM_Error':<10} {'MU_Error':<10} {'Status'}\n")
    f.write("-" * 80 + "\n")
    
    successful_results = []
    failed_results = []
    
    for r in results:
        if r['success']:
            f.write(f"{r['x_ti']:<6.2f} {r['temp']:<6} {r['cpu_gm']:<12.3f} {r['gpu_gm']:<12.3f} {r['gm_error']:<10.3f} {r['max_mu_error']:<10.3f} SUCCESS\n")
            successful_results.append(r)
        else:
            f.write(f"{r['x_ti']:<6.2f} {r['temp']:<6} {'ERROR':<12} {'ERROR':<12} {'N/A':<10} {'N/A':<10} FAILED\n")
            failed_results.append(r)
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("DETAILED ANALYSIS\n")
    f.write("=" * 80 + "\n\n")
    
    if successful_results:
        gm_errors = [r['gm_error'] for r in successful_results]
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
        
        # Show actual values
        f.write("REPRESENTATIVE VALUES:\n")
        f.write(f"{'X(TI)':<6} {'T(K)':<6} {'CPU_GM':<15} {'GPU_GM':<15} {'Abs_Error':<15}\n")
        f.write("-" * 75 + "\n")
        
        for r in successful_results:
            f.write(f"{r['x_ti']:<6.2f} {r['temp']:<6} {r['cpu_gm']:<15.3f} {r['gpu_gm']:<15.3f} {r['gm_error']:<15.3f}\n")
    
    if failed_results:
        f.write(f"\nFAILED TESTS ({len(failed_results)}):\n")
        for r in failed_results:
            f.write(f"  X(TI)={r['x_ti']:.2f}, T={r['temp']}K: {r['error']}\n")

print("Results saved to /tmp/output.txt")
print("Test completed!")