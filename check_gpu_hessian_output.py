#!/usr/bin/env python
"""Check GPU Hessian output parsing."""

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

# Test conditions
conditions = {v.X('TI'): 0.01, v.T: 1000, v.P: 101325}

# Run with Hessian debug output and capture output
env = os.environ.copy()
env['PYCALPHAD_DEBUG_CATEGORIES'] = 'HESSIAN'

# Run test
proc = subprocess.run([sys.executable, 'test_cpu_gpu_comparison.py'], 
                     env=env, capture_output=True, text=True)

output = proc.stdout

# Look for GPU Hessian outputs more carefully
print("=== GPU Hessian Output Pattern Analysis ===")

gpu_sections = []
current_section = None

for line in output.split('\n'):
    # Check for GPU iteration markers
    if '[GPU EQUILIBRIUM MATRIX] Filling equilibrium system at iteration' in line:
        iter_num = int(line.split('iteration ')[1])
        if current_section:
            gpu_sections.append(current_section)
        current_section = {'iteration': iter_num, 'hessians': []}
    
    # Capture GPU Hessian output
    elif 'GPU DEBUG: Hessian calculated for phase record' in line and current_section is not None:
        phase = int(line.split('phase record ')[1])
        current_section['hessians'].append({'phase': phase, 'values': []})
    
    elif '    [' in line and 'idx=' in line and current_section and current_section['hessians']:
        import re
        values = re.findall(r'(\d+\.\d+e[+-]\d+)', line)
        for v in values:
            current_section['hessians'][-1]['values'].append(float(v))

if current_section:
    gpu_sections.append(current_section)

# Print what we found
print(f"\nFound {len(gpu_sections)} GPU iteration sections")

for i, section in enumerate(gpu_sections[:10]):  # First 10 iterations
    print(f"\nGPU Iteration {section['iteration']}:")
    for hess in section['hessians']:
        if hess['values']:
            print(f"  Phase {hess['phase']}: H[0,0]={hess['values'][0]:.3e}, H[0,1]={hess['values'][1]:.3e}, H[1,1]={hess['values'][3]:.3e}")

# Also check CPU pattern
print("\n\n=== CPU Hessian Pattern ===")
cpu_iter = -1
for line in output.split('\n'):
    if '[CPU MASS BALANCE] recompute() - iteration' in line:
        cpu_iter = int(line.split('iteration ')[1].split(':')[0])
    elif '[CPU HESSIAN]' in line and 'after formulahess:' in line and cpu_iter >= 0:
        phase = int(line.split('Phase ')[1].split(' ')[0])
        print(f"CPU Iteration {cpu_iter}, Phase {phase}")