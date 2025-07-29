#!/usr/bin/env python
"""Quick test of GPU vs CPU for Al-Cu-Fe system with fewer conditions."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import warnings
import time

# Suppress warnings
warnings.filterwarnings("ignore")

# Load database
dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']
phases = filter_phases(dbf, comps)

print(f"Testing Al-Cu-Fe system with {len(phases)} available phases")
print()

# Test just a few key conditions with lower pdens for speed
test_conditions = [
    # Binary edges
    {'desc': 'Al-Cu (X_CU=0.3)', 'X_CU': 0.3, 'X_FE': 0.0, 'T': 800},
    {'desc': 'Al-Fe (X_FE=0.3)', 'X_CU': 0.0, 'X_FE': 0.3, 'T': 800},
    # Ternary
    {'desc': 'Ternary center', 'X_CU': 0.33, 'X_FE': 0.33, 'T': 900},
]

pressure = 101325
results = []

print(f"Testing {len(test_conditions)} conditions with pdens=10 for quick results")
print("="*80)

for i, cond in enumerate(test_conditions):
    desc = cond['desc']
    x_cu = cond['X_CU']
    x_fe = cond['X_FE']
    x_al = 1.0 - x_cu - x_fe
    temp = cond['T']
    
    print(f"\nTest {i+1}: {desc}")
    print(f"  X(AL)={x_al:.2f}, X(CU)={x_cu:.2f}, X(FE)={x_fe:.2f}, T={temp}K")
    
    try:
        # Set up conditions
        conditions = {v.T: temp, v.P: pressure, v.N: 1}
        
        # Add composition constraints
        if x_cu < 0.999:
            conditions[v.X('CU')] = x_cu
        if x_fe < 0.999 and x_cu + x_fe < 0.999:
            conditions[v.X('FE')] = x_fe
        
        # CPU calculation with lower pdens for speed
        print("  Running CPU...")
        cpu_start = time.time()
        cpu_result = equilibrium(dbf, comps, phases, conditions, 
                               calc_opts={'pdens': 10}, 
                               verbose=False)
        cpu_time = time.time() - cpu_start
        cpu_gm = float(cpu_result.GM.values)
        
        # Extract CPU phases
        cpu_phases = []
        for idx in range(cpu_result.dims['vertex']):
            np_val = float(cpu_result.NP.values[0,0,0,0,idx])
            if np_val > 1e-6:
                phase_name = str(cpu_result.Phase.values[0,0,0,0,idx])
                if phase_name and phase_name != '':
                    cpu_phases.append(f"{phase_name}({np_val:.3f})")
        
        # GPU calculation
        print("  Running GPU...")
        gpu_start = time.time()
        gpu_result = equilibrium(dbf, comps, phases, conditions, 
                               calc_opts={'pdens': 10}, 
                               verbose=False, 
                               gpu=True)
        gpu_time = time.time() - gpu_start
        gpu_gm = float(gpu_result.GM.values)
        
        # Extract GPU phases  
        gpu_phases = []
        for idx in range(gpu_result.dims['vertex']):
            np_val = float(gpu_result.NP.values[0,0,0,0,idx])
            if np_val > 1e-6:
                phase_name = str(gpu_result.Phase.values[0,0,0,0,idx])
                if phase_name and phase_name != '':
                    gpu_phases.append(f"{phase_name}({np_val:.3f})")
        
        # Compare
        gm_diff = abs(gpu_gm - cpu_gm)
        status = "PASS" if gm_diff < 1.0 else "FAIL"
        
        print(f"  CPU: GM={cpu_gm:.2f} J/mol, phases={','.join(cpu_phases) if cpu_phases else 'NONE'} (time={cpu_time:.2f}s)")
        print(f"  GPU: GM={gpu_gm:.2f} J/mol, phases={','.join(gpu_phases) if gpu_phases else 'NONE'} (time={gpu_time:.2f}s)")
        print(f"  Difference: {gm_diff:.6f} J/mol - {status}")
        
        results.append({
            'desc': desc,
            'cpu_gm': cpu_gm,
            'gpu_gm': gpu_gm,
            'diff': gm_diff,
            'status': status,
            'cpu_phases': cpu_phases,
            'gpu_phases': gpu_phases,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time
        })
        
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        results.append({
            'desc': desc,
            'status': 'ERROR',
            'error': str(e)
        })

# Summary
print("\n" + "="*80)
print("SUMMARY:")
passed = sum(1 for r in results if r.get('status') == 'PASS')
failed = sum(1 for r in results if r.get('status') == 'FAIL')
errors = sum(1 for r in results if r.get('status') == 'ERROR')

print(f"  Total: {len(results)}")
print(f"  Passed: {passed}")
print(f"  Failed: {failed}")
print(f"  Errors: {errors}")
print(f"  Pass rate: {100*passed/len(results):.1f}%" if len(results) > 0 else "  No tests completed")

# Show timing comparison
if any(r.get('cpu_time') for r in results):
    avg_cpu_time = np.mean([r['cpu_time'] for r in results if r.get('cpu_time')])
    avg_gpu_time = np.mean([r['gpu_time'] for r in results if r.get('gpu_time')])
    print(f"\n  Average CPU time: {avg_cpu_time:.2f}s")
    print(f"  Average GPU time: {avg_gpu_time:.2f}s")
    print(f"  GPU speedup: {avg_cpu_time/avg_gpu_time:.1f}x" if avg_gpu_time > 0 else "  GPU speedup: N/A")