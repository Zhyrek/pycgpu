#!/usr/bin/env python
"""Test individual vs batch condition processing on GPU."""

import os
import sys
sys.path.insert(0, os.getcwd())
import subprocess
import numpy as np

def run_individual_gpu_test(x_ti, t=1000):
    """Run GPU test on a single condition set."""
    code = f'''
import sys, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database("NbTi.tdb")
comps = ["NB", "TI", "VA"]
phases = filter_phases(dbf, comps)

# Single condition
conditions = {{v.X("TI"): {x_ti}, v.T: {t}, v.P: 101325}}

result = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
x_ti_final = result.X.sel(component="TI").values.flatten()[0]
print(f"RESULT:{{x_ti_final:.12f}}")
'''
    
    try:
        proc = subprocess.run([sys.executable, '-c', code], 
                             capture_output=True, text=True, timeout=60)
        if proc.returncode == 0:
            # Extract result
            for line in proc.stdout.split('\n'):
                if line.startswith('RESULT:'):
                    return float(line.split(':')[1])
        return None
    except Exception as e:
        print(f"Error in individual test: {e}")
        return None

def run_batch_gpu_test(x_ti_values, t=1000):
    """Run GPU test on multiple condition sets in one call."""
    x_ti_array = f"[{', '.join(map(str, x_ti_values))}]"
    
    code = f'''
import sys, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database("NbTi.tdb")
comps = ["NB", "TI", "VA"]
phases = filter_phases(dbf, comps)

# Multiple conditions in one call
x_ti_values = np.array({x_ti_array})
conditions = {{v.X("TI"): x_ti_values, v.T: {t}, v.P: 101325}}

result = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
x_ti_results = result.X.sel(component="TI").values.flatten()

for i, x_result in enumerate(x_ti_results):
    print(f"RESULT{{i}}:{{x_result:.12f}}")
'''
    
    try:
        proc = subprocess.run([sys.executable, '-c', code], 
                             capture_output=True, text=True, timeout=120)
        if proc.returncode == 0:
            # Extract results
            results = []
            for line in proc.stdout.split('\n'):
                if line.startswith('RESULT') and ':' in line:
                    idx = int(line.split('RESULT')[1].split(':')[0])
                    value = float(line.split(':')[1])
                    results.append((idx, value))
            
            # Sort by index and return values
            results.sort(key=lambda x: x[0])
            return [value for idx, value in results]
        return None
    except Exception as e:
        print(f"Error in batch test: {e}")
        return None

# Test different X(TI) values
test_x_ti_values = [0.005, 0.01, 0.02]

print("Testing GPU data isolation: Individual vs Batch processing")
print("=" * 60)

print("\n1. INDIVIDUAL GPU TESTS (separate equilibrium calls)")
print("-" * 40)
individual_results = []
for x_ti in test_x_ti_values:
    result = run_individual_gpu_test(x_ti)
    individual_results.append(result)
    print(f"X(TI) = {x_ti:.3f} → GPU result = {result:.8f}" if result else f"X(TI) = {x_ti:.3f} → FAILED")

print("\n2. BATCH GPU TEST (single equilibrium call with array)")
print("-" * 40)
batch_results = run_batch_gpu_test(test_x_ti_values)
if batch_results:
    for i, (x_ti, result) in enumerate(zip(test_x_ti_values, batch_results)):
        print(f"X(TI) = {x_ti:.3f} → GPU result = {result:.8f}")
else:
    print("BATCH TEST FAILED")

print("\n3. COMPARISON")
print("-" * 40)
if individual_results and batch_results and len(individual_results) == len(batch_results):
    all_match = True
    for i, (x_ti, individual, batch) in enumerate(zip(test_x_ti_values, individual_results, batch_results)):
        if individual is not None and batch is not None:
            diff = abs(individual - batch)
            match = diff < 1e-10
            all_match = all_match and match
            status = "MATCH" if match else "DIFFER"
            print(f"X(TI) = {x_ti:.3f}: Individual = {individual:.8f}, Batch = {batch:.8f}, Diff = {diff:.2e} [{status}]")
        else:
            print(f"X(TI) = {x_ti:.3f}: FAILED")
            all_match = False
    
    print(f"\nOVERALL: {'ALL MATCH' if all_match else 'DATA ISOLATION ISSUE DETECTED'}")
else:
    print("COMPARISON FAILED - incomplete results")

print("\n4. CPU REFERENCE (for comparison)")
print("-" * 40)
for x_ti in test_x_ti_values:
    # CPU reference
    code = f'''
import sys, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database("NbTi.tdb")
comps = ["NB", "TI", "VA"]
phases = filter_phases(dbf, comps)
conditions = {{v.X("TI"): {x_ti}, v.T: 1000, v.P: 101325}}

result = equilibrium(dbf, comps, phases, conditions, verbose=False)
x_ti_final = result.X.sel(component="TI").values.flatten()[0]
print(f"RESULT:{{x_ti_final:.12f}}")
'''
    
    try:
        proc = subprocess.run([sys.executable, '-c', code], 
                             capture_output=True, text=True, timeout=60)
        if proc.returncode == 0:
            for line in proc.stdout.split('\n'):
                if line.startswith('RESULT:'):
                    cpu_result = float(line.split(':')[1])
                    print(f"X(TI) = {x_ti:.3f} → CPU result = {cpu_result:.8f}")
                    break
        else:
            print(f"X(TI) = {x_ti:.3f} → CPU FAILED")
    except Exception as e:
        print(f"X(TI) = {x_ti:.3f} → CPU ERROR: {e}")