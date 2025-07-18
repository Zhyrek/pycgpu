#!/usr/bin/env python
"""Compare CPU and GPU Hessian values across iterations."""

import os
import sys
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import numpy as np
import subprocess

# Create test case
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

print("=== Comparing CPU and GPU Hessian Values ===")
print("Testing with T=1000K, X(TI)=0.01")

# Test conditions
conditions = {v.X('TI'): 0.01, v.T: 1000, v.P: 101325}

# Run with Hessian debug output and capture output
env = os.environ.copy()
env['PYCALPHAD_DEBUG_CATEGORIES'] = 'HESSIAN'

# Run test
proc = subprocess.run([sys.executable, 'test_cpu_gpu_comparison.py'], 
                     env=env, capture_output=True, text=True)

output = proc.stdout

# Parse CPU Hessian values
cpu_hessians = []
gpu_hessians = []

for line in output.split('\n'):
    if '[CPU HESSIAN]' in line and 'after formulahess:' in line:
        # Extract phase number
        phase = int(line.split('Phase ')[1].split(' ')[0])
        cpu_hessians.append({'phase': phase, 'values': []})
    elif '  Row ' in line and cpu_hessians:
        # Extract row values
        parts = line.split(':')[1].strip().split()
        values = []
        for v in parts:
            try:
                values.append(float(v))
            except ValueError:
                pass  # Skip non-numeric values like '|'
        cpu_hessians[-1]['values'].extend(values)
    elif 'GPU DEBUG: Hessian calculated for phase record' in line:
        # Extract GPU phase
        phase = int(line.split('phase record ')[1])
        gpu_hessians.append({'phase': phase, 'values': []})
    elif '    [' in line and 'idx=' in line and gpu_hessians:
        # Extract GPU Hessian values
        # Parse values like: [0] 8.355014e+03 (idx=18) 1.304530e+04 (idx=19)
        import re
        # Find all scientific notation numbers
        values = re.findall(r'(\d+\.\d+e[+-]\d+)', line)
        for v in values:
            gpu_hessians[-1]['values'].append(float(v))

# Compare iterations
print(f"\nFound {len(cpu_hessians)} CPU Hessian calculations")
print(f"Found {len(gpu_hessians)} GPU Hessian calculations")

# Group by iteration (every 2 phases is one iteration)
cpu_iterations = []
gpu_iterations = []

for i in range(0, min(len(cpu_hessians), 20), 2):  # First 10 iterations
    if i+1 < len(cpu_hessians):
        cpu_iterations.append({
            'iter': i//2,
            'phase0': cpu_hessians[i]['values'],
            'phase1': cpu_hessians[i+1]['values']
        })

for i in range(0, min(len(gpu_hessians), 20), 2):
    if i+1 < len(gpu_hessians):
        gpu_iterations.append({
            'iter': i//2,
            'phase0': gpu_hessians[i]['values'],
            'phase1': gpu_hessians[i+1]['values']
        })

# Compare
print("\n=== Hessian Comparison by Iteration ===")
for i in range(min(len(cpu_iterations), len(gpu_iterations))):
    cpu_iter = cpu_iterations[i]
    gpu_iter = gpu_iterations[i]
    
    print(f"\nIteration {i}:")
    
    # Phase 0
    if cpu_iter['phase0'] and gpu_iter['phase0']:
        cpu_vals = cpu_iter['phase0']
        gpu_vals = gpu_iter['phase0']
        print(f"  Phase 0:")
        print(f"    CPU: H[3,3]={cpu_vals[0]:.3e}, H[3,4]={cpu_vals[1]:.3e}, H[4,4]={cpu_vals[3]:.3e}")
        print(f"    GPU: H[0,0]={gpu_vals[0]:.3e}, H[0,1]={gpu_vals[1]:.3e}, H[1,1]={gpu_vals[3]:.3e}")
        
        # Check differences
        diff_00 = abs(cpu_vals[0] - gpu_vals[0])
        diff_01 = abs(cpu_vals[1] - gpu_vals[1])
        diff_11 = abs(cpu_vals[3] - gpu_vals[3])
        
        if diff_00 > 1e-6 or diff_01 > 1e-6 or diff_11 > 1e-6:
            print(f"    ⚠️  DIFFERENCE: H[0,0] diff={diff_00:.3e}, H[0,1] diff={diff_01:.3e}, H[1,1] diff={diff_11:.3e}")
        else:
            print(f"    ✓ Values match")
    
    # Phase 1  
    if cpu_iter['phase1'] and gpu_iter['phase1']:
        cpu_vals = cpu_iter['phase1']
        gpu_vals = gpu_iter['phase1']
        print(f"  Phase 1:")
        print(f"    CPU: H[3,3]={cpu_vals[0]:.3e}, H[3,4]={cpu_vals[1]:.3e}, H[4,4]={cpu_vals[3]:.3e}")
        print(f"    GPU: H[0,0]={gpu_vals[0]:.3e}, H[0,1]={gpu_vals[1]:.3e}, H[1,1]={gpu_vals[3]:.3e}")
        
        # Check differences
        diff_00 = abs(cpu_vals[0] - gpu_vals[0])
        diff_01 = abs(cpu_vals[1] - gpu_vals[1])
        diff_11 = abs(cpu_vals[3] - gpu_vals[3])
        
        if diff_00 > 1e-6 or diff_01 > 1e-6 or diff_11 > 1e-6:
            print(f"    ⚠️  DIFFERENCE: H[0,0] diff={diff_00:.3e}, H[0,1] diff={diff_01:.3e}, H[1,1] diff={diff_11:.3e}")
        else:
            print(f"    ✓ Values match")