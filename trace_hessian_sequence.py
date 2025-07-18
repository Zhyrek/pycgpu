#!/usr/bin/env python
"""Trace the full sequence of Hessian calculations."""

import os
import sys
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import numpy as np
import subprocess
import re

# Create test case
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test conditions
conditions = {v.X('TI'): 0.01, v.T: 1000, v.P: 101325}

# Run with Hessian debug output and capture output
env = os.environ.copy()
env['PYCALPHAD_DEBUG_CATEGORIES'] = 'HESSIAN'

# Run test
proc = subprocess.run([sys.executable, 'test_cpu_gpu_comparison.py'], 
                     env=env, capture_output=True, text=True)

output = proc.stdout

print("=== Full Hessian Calculation Sequence ===\n")

# Track CPU and GPU separately
cpu_sequence = []
gpu_sequence = []
current_cpu_iter = -1
current_gpu_iter = -1
gpu_pre_iteration = True

for line in output.split('\n'):
    # CPU tracking
    if '[CPU MASS BALANCE] recompute() - iteration' in line:
        current_cpu_iter = int(line.split('iteration ')[1].split(':')[0])
    elif '[CPU HESSIAN]' in line and 'after formulahess:' in line:
        phase = int(line.split('Phase ')[1].split(' ')[0])
        cpu_sequence.append({'iteration': current_cpu_iter, 'phase': phase, 'line': line})
    
    # GPU tracking
    if '[GPU] SEGMENT 27: Construct equilibrium system' in line and 'iteration' in line:
        current_gpu_iter = int(line.split('(condition 0)')[0].split('iteration ')[-1])
        gpu_pre_iteration = False
    elif 'GPU DEBUG: Hessian calculated for phase record' in line:
        phase = int(line.split('phase record ')[1])
        context = 'pre-iteration' if gpu_pre_iteration else f'iteration {current_gpu_iter}'
        gpu_sequence.append({'context': context, 'phase': phase, 'line': line})

# Print sequences
print("CPU Hessian Sequence:")
for i, item in enumerate(cpu_sequence[:20]):
    print(f"{i:3d}: Iteration {item['iteration']}, Phase {item['phase']}")

print("\n\nGPU Hessian Sequence:")
for i, item in enumerate(gpu_sequence[:20]):
    print(f"{i:3d}: {item['context']}, Phase {item['phase']}")

# Now extract values for first few
print("\n\n=== Comparing First Hessian Values ===")

# Re-parse to get values
hessian_values = {'cpu': [], 'gpu': []}

for line in output.split('\n'):
    if '[CPU HESSIAN]' in line and 'after formulahess:' in line:
        phase = int(line.split('Phase ')[1].split(' ')[0])
        hessian_values['cpu'].append({'phase': phase, 'values': []})
    elif '  Row ' in line and hessian_values['cpu'] and not hessian_values['cpu'][-1]['values']:
        # Get next 2 rows for 2x2 matrix
        idx = output.split('\n').index(line)
        for j in range(2):
            row_line = output.split('\n')[idx + j]
            if '  Row ' in row_line:
                parts = row_line.split(':')[1].strip().split()
                vals = []
                for v in parts:
                    try:
                        vals.append(float(v))
                    except:
                        pass
                hessian_values['cpu'][-1]['values'].extend(vals)
    
    elif 'GPU DEBUG: Hessian calculated for phase record' in line:
        phase = int(line.split('phase record ')[1])
        hessian_values['gpu'].append({'phase': phase, 'values': []})
    elif '    [' in line and 'idx=' in line and hessian_values['gpu'] and len(hessian_values['gpu'][-1]['values']) < 4:
        values = re.findall(r'(\d+\.\d+e[+-]\d+)', line)
        for v in values:
            hessian_values['gpu'][-1]['values'].append(float(v))

# Compare first few
print("\nFirst CPU Hessian values:")
for i in range(min(4, len(hessian_values['cpu']))):
    h = hessian_values['cpu'][i]
    if len(h['values']) >= 4:
        print(f"  CPU #{i} Phase {h['phase']}: [{h['values'][0]:.3e}, {h['values'][1]:.3e}, {h['values'][3]:.3e}]")

print("\nFirst GPU Hessian values:")
for i in range(min(4, len(hessian_values['gpu']))):
    h = hessian_values['gpu'][i]
    if len(h['values']) >= 4:
        print(f"  GPU #{i} Phase {h['phase']}: [{h['values'][0]:.3e}, {h['values'][1]:.3e}, {h['values'][3]:.3e}]")