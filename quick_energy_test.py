#!/usr/bin/env python
"""Quick CPU vs GPU energy comparison with minimal output."""

import os
import sys
import numpy as np
import subprocess
import io
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Redirect ALL output to devnull during equilibrium calculations
def run_equilibrium_silent(dbf, comps, phases, conditions, gpu=False):
    """Run equilibrium with all output suppressed."""
    import contextlib
    from io import StringIO
    import warnings
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Redirect stdout, stderr, and warnings
        with contextlib.redirect_stdout(StringIO()), \
             contextlib.redirect_stderr(StringIO()):
            # Also set CUDA_VISIBLE_DEVICES to suppress CUDA init messages
            old_cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            if gpu:
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            try:
                result = equilibrium(dbf, comps, phases, conditions, gpu=gpu, verbose=False)
                return result
            finally:
                os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda_devices

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test just a few representative conditions
test_conditions = [
    {'x_ti': 0.05, 'temp': 800},
    {'x_ti': 0.1, 'temp': 1000},
    {'x_ti': 0.5, 'temp': 700},
    {'x_ti': 0.9, 'temp': 900},
]

print("Quick CPU vs GPU Free Energy Comparison")
print("=" * 60)
print(f"{'X(TI)':<6} {'T(K)':<6} {'CPU_GM':<12} {'GPU_GM':<12} {'Error':<10}")
print("-" * 60)

results = []

for test in test_conditions:
    x_ti, temp = test['x_ti'], test['temp']
    conditions = {v.X('TI'): x_ti, v.T: temp, v.P: 101325}
    
    try:
        print(f"{x_ti:<6.2f} {temp:<6}", end=" ", flush=True)
        
        # CPU calculation
        result_cpu = run_equilibrium_silent(dbf, comps, phases, conditions, gpu=False)
        cpu_gm = result_cpu.GM.values.flatten()[0]
        print(f"{cpu_gm:<12.3f}", end=" ", flush=True)
        
        # GPU calculation  
        result_gpu = run_equilibrium_silent(dbf, comps, phases, conditions, gpu=True)
        gpu_gm = result_gpu.GM.values.flatten()[0]
        print(f"{gpu_gm:<12.3f}", end=" ", flush=True)
        
        # Error
        error = abs(cpu_gm - gpu_gm)
        print(f"{error:<10.3f}")
        
        results.append({
            'x_ti': x_ti, 'temp': temp,
            'cpu_gm': cpu_gm, 'gpu_gm': gpu_gm, 'error': error,
            'success': True
        })
        
    except Exception as e:
        print(f"ERROR: {str(e)[:30]}")
        results.append({'x_ti': x_ti, 'temp': temp, 'success': False, 'error': str(e)})

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

successful = [r for r in results if r['success']]
if successful:
    errors = [r['error'] for r in successful]
    print(f"Successful tests: {len(successful)}/{len(results)}")
    print(f"Max absolute error: {max(errors):.6f} J/mol")
    print(f"Avg absolute error: {np.mean(errors):.6f} J/mol")
    
    print(f"\nRepresentative GM values:")
    for r in successful[:2]:  # Show first 2 successful results
        print(f"  X(TI)={r['x_ti']:.2f}, T={r['temp']}K: CPU={r['cpu_gm']:.3f}, GPU={r['gpu_gm']:.3f} J/mol")
else:
    print("All tests failed!")
    for r in results:
        if not r['success']:
            print(f"  X(TI)={r['x_ti']:.2f}, T={r['temp']}K: {r['error']}")

print("=" * 60)