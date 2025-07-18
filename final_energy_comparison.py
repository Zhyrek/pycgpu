#!/usr/bin/env python
"""Final comprehensive CPU vs GPU energy comparison."""

import os
import sys
import subprocess
import tempfile
import re
sys.path.insert(0, os.getcwd())

test_conditions = [
    {'x_ti': 0.05, 'temp': 800, 'name': 'Low Ti, Medium T'},
    {'x_ti': 0.1, 'temp': 1000, 'name': 'Low Ti, High T'},
    {'x_ti': 0.5, 'temp': 700, 'name': 'Medium Ti, Low T'},
    {'x_ti': 0.5, 'temp': 1000, 'name': 'Medium Ti, High T'},
    {'x_ti': 0.9, 'temp': 800, 'name': 'High Ti, Medium T'},
]

print("CPU vs GPU Free Energy Comparison - Final Results")
print("=" * 80)
print(f"{'Condition':<25} {'X(TI)':<6} {'T(K)':<6} {'CPU (J/mol)':<12} {'GPU (J/mol)':<12} {'Error':<8}")
print("-" * 80)

results = []

for test in test_conditions:
    x_ti, temp = test['x_ti'], test['temp']
    
    # Create temporary script for isolated execution
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(f"""
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '{os.getcwd()}')

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {{v.X('TI'): {x_ti}, v.T: {temp}, v.P: 101325}}

try:
    result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
    cpu_gm = result_cpu.GM.values.flatten()[0]
    print(f"CPU_GM: {{cpu_gm:.6f}}")
    
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    gpu_gm = result_gpu.GM.values.flatten()[0]  
    print(f"GPU_GM: {{gpu_gm:.6f}}")
    
    error = abs(cpu_gm - gpu_gm)
    print(f"ERROR: {{error:.6f}}")
    
except Exception as e:
    print(f"FAILED: {{str(e)}}")
""")
        temp_file = f.name
    
    try:
        # Run with timeout and capture all output
        result = subprocess.run(['python', temp_file], 
                              capture_output=True, text=True, timeout=180,
                              env={**os.environ, 'PYTHONUNBUFFERED': '1'})
        
        # Extract values from output
        cpu_match = re.search(r'CPU_GM: ([-\d.]+)', result.stdout)
        gpu_match = re.search(r'GPU_GM: ([-\d.]+)', result.stdout)
        error_match = re.search(r'ERROR: ([\d.]+)', result.stdout)
        
        if cpu_match and gpu_match and error_match:
            cpu_gm = float(cpu_match.group(1))
            gpu_gm = float(gpu_match.group(1))
            error = float(error_match.group(1))
            
            print(f"{test['name']:<25} {x_ti:<6.2f} {temp:<6} {cpu_gm:<12.3f} {gpu_gm:<12.3f} {error:<8.3f}")
            
            results.append({
                'name': test['name'],
                'x_ti': x_ti,
                'temp': temp,
                'cpu_gm': cpu_gm,
                'gpu_gm': gpu_gm,
                'error': error,
                'success': True
            })
        else:
            print(f"{test['name']:<25} {x_ti:<6.2f} {temp:<6} {'ERROR':<12} {'ERROR':<12} {'N/A':<8}")
            results.append({
                'name': test['name'],
                'x_ti': x_ti,
                'temp': temp,
                'success': False,
                'stdout': result.stdout,
                'stderr': result.stderr
            })
            
    except subprocess.TimeoutExpired:
        print(f"{test['name']:<25} {x_ti:<6.2f} {temp:<6} {'TIMEOUT':<12} {'TIMEOUT':<12} {'N/A':<8}")
        results.append({
            'name': test['name'],
            'x_ti': x_ti,
            'temp': temp,
            'success': False,
            'error_type': 'timeout'
        })
    except Exception as e:
        print(f"{test['name']:<25} {x_ti:<6.2f} {temp:<6} {'EXCEPT':<12} {'EXCEPT':<12} {'N/A':<8}")
        results.append({
            'name': test['name'],
            'x_ti': x_ti,
            'temp': temp,
            'success': False,
            'error_type': str(e)
        })
    finally:
        os.unlink(temp_file)

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

successful = [r for r in results if r['success']]
failed = [r for r in results if not r['success']]

if successful:
    errors = [r['error'] for r in successful]
    cpu_values = [r['cpu_gm'] for r in successful]
    
    print(f"Successful tests: {len(successful)}/{len(results)}")
    print(f"Failed tests: {len(failed)}")
    print(f"\nFree Energy Error Analysis:")
    print(f"  Minimum absolute error: {min(errors):.6f} J/mol")
    print(f"  Maximum absolute error: {max(errors):.6f} J/mol")
    print(f"  Average absolute error: {sum(errors)/len(errors):.6f} J/mol")
    
    print(f"\nFree Energy Magnitude Context:")
    avg_magnitude = sum(abs(gm) for gm in cpu_values) / len(cpu_values)
    max_error = max(errors)
    avg_error = sum(errors) / len(errors)
    print(f"  Typical |GM| magnitude: {avg_magnitude:.0f} J/mol")
    print(f"  Max error as % of |GM|: {(max_error/avg_magnitude)*100:.4f}%")
    print(f"  Avg error as % of |GM|: {(avg_error/avg_magnitude)*100:.4f}%")
    
    if max_error < 1.0:
        status = "EXCELLENT (< 1 J/mol)"
    elif max_error < 10.0:
        status = "VERY GOOD (< 10 J/mol)"
    elif max_error < 100.0:
        status = "GOOD (< 100 J/mol)"
    else:
        status = "NEEDS INVESTIGATION (≥ 100 J/mol)"
    
    print(f"\nOverall Assessment: {status}")
    
    # Show actual representative values
    print(f"\nRepresentative Free Energy Values:")
    for r in successful[:3]:  # Show first 3
        print(f"  {r['name']}: CPU={r['cpu_gm']:.3f}, GPU={r['gpu_gm']:.3f} J/mol (error={r['error']:.3f})")

else:
    print("All tests failed!")

if failed:
    print(f"\nFailed Tests Details:")
    for r in failed:
        if 'error_type' in r:
            print(f"  {r['name']}: {r['error_type']}")
        else:
            print(f"  {r['name']}: Check output manually")

print("=" * 80)