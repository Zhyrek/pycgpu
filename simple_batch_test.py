#!/usr/bin/env python
"""Simple batch test with just final results."""

import os
import sys
sys.path.insert(0, os.getcwd())
import subprocess

# Test different conditions
conditions = [
    "T=1000, X_TI=0.005",
    "T=1000, X_TI=0.01", 
    "T=1000, X_TI=0.02",
]

def run_single_test(t, x_ti, use_gpu=False):
    """Run a single test and return final X(TI) value."""
    gpu_flag = "gpu=True" if use_gpu else "gpu=False"
    code = f'''
import sys, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database("NbTi.tdb")
comps = ["NB", "TI", "VA"]
phases = filter_phases(dbf, comps)
conditions = {{v.X("TI"): {x_ti}, v.T: {t}, v.P: 101325}}

result = equilibrium(dbf, comps, phases, conditions, {gpu_flag}, verbose=False)
x_ti_final = result.X.sel(component="TI").values.flatten()[0]
print(f"{{x_ti_final:.12f}}")
'''
    
    try:
        proc = subprocess.run([sys.executable, '-c', code], 
                             capture_output=True, text=True, timeout=60)
        if proc.returncode == 0:
            # Find the last line with a number
            lines = [line.strip() for line in proc.stdout.split('\n') if line.strip()]
            for line in reversed(lines):
                try:
                    return float(line)
                except ValueError:
                    continue
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

print("Testing GPU vs CPU for multiple conditions...")
print("=" * 50)

for cond in conditions:
    t_val = int(cond.split('T=')[1].split(',')[0])
    x_val = float(cond.split('X_TI=')[1])
    
    print(f"Testing {cond}...")
    
    # Run CPU test
    cpu_result = run_single_test(t_val, x_val, use_gpu=False)
    
    # Run GPU test  
    gpu_result = run_single_test(t_val, x_val, use_gpu=True)
    
    if cpu_result is not None and gpu_result is not None:
        diff = abs(cpu_result - gpu_result)
        status = "PASS" if diff < 1e-6 else "FAIL"
        print(f"  CPU: {cpu_result:.8f}")
        print(f"  GPU: {gpu_result:.8f}")
        print(f"  Diff: {diff:.2e} [{status}]")
    else:
        print(f"  FAILED - CPU: {cpu_result}, GPU: {gpu_result}")
    
    print()

print("Test completed.")