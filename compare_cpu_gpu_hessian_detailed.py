#!/usr/bin/env python
"""Compare CPU and GPU Hessian calculations in detail."""

import os
import sys
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import numpy as np
import subprocess
import re

# Run test and capture output
env = os.environ.copy()
env['PYCALPHAD_DEBUG_CATEGORIES'] = 'HESSIAN'

proc = subprocess.run([sys.executable, 'trace_hessian_inputs_test.py'], 
                     env=env, capture_output=True, text=True)

output = proc.stdout

# Parse CPU values
print("=== CPU vs GPU Hessian Comparison ===\n")

# Extract CPU DOF inputs and Hessian outputs
cpu_data = []
gpu_data = []

lines = output.split('\n')
i = 0
while i < len(lines):
    line = lines[i]
    
    # Look for CPU formulahess (we don't have the input trace for CPU yet)
    if '[CPU HESSIAN]' in line and 'after formulahess:' in line:
        phase = int(line.split('Phase ')[1].split(' ')[0])
        # Get iteration from previous lines
        iter_num = -1
        for j in range(max(0, i-10), i):
            if 'iteration' in lines[j] and 'recompute' in lines[j]:
                iter_num = int(lines[j].split('iteration ')[1].split(':')[0])
                break
        
        # Get Hessian values from next lines
        hess_vals = []
        if i+1 < len(lines) and 'Row 3:' in lines[i+1]:
            vals = lines[i+1].split(':')[1].strip().split()
            hess_vals.extend([float(v) for v in vals if v != '|'])
        if i+2 < len(lines) and 'Row 4:' in lines[i+2]:
            vals = lines[i+2].split(':')[1].strip().split()
            hess_vals.extend([float(v) for v in vals if v != '|'])
            
        if iter_num >= 0 and len(hess_vals) >= 3:
            cpu_data.append({
                'iteration': iter_num,
                'phase': phase,
                'hessian': [hess_vals[0], hess_vals[1], hess_vals[3]]  # [0,0], [0,1], [1,1]
            })
    
    # Look for GPU formulahess input and output
    elif '[GPU FORMULAHESS INPUT]' in line:
        match = re.search(r'Phase (\d+) iteration (\d+), DOF: (.+)', line)
        if match:
            phase = int(match.group(1))
            iter_num = int(match.group(2))
            dof_str = match.group(3)
            dof_vals = [float(x) for x in dof_str.split()]
            
            # Get Hessian values from next lines
            hess_vals = []
            for j in range(i+1, min(i+5, len(lines))):
                if '[0]' in lines[j] or '[1]' in lines[j]:
                    vals = re.findall(r'(\d+\.\d+e[+-]\d+)', lines[j])
                    hess_vals.extend([float(v) for v in vals])
            
            if len(hess_vals) >= 4:
                gpu_data.append({
                    'iteration': iter_num,
                    'phase': phase,
                    'dof': dof_vals,
                    'hessian': [hess_vals[0], hess_vals[1], hess_vals[3]]  # [0,0], [0,1], [1,1]
                })
    
    i += 1

# Compare by iteration
print("Note: CPU uses indices [3,3], [3,4], [4,4] for site fraction Hessian")
print("      GPU uses indices [0,0], [0,1], [1,1] for site fraction Hessian\n")

for iter_num in range(3):
    print(f"\n=== Iteration {iter_num} ===")
    
    # Get CPU values for this iteration
    cpu_iter = [d for d in cpu_data if d['iteration'] == iter_num]
    gpu_iter = [d for d in gpu_data if d['iteration'] == iter_num]
    
    for phase in [0, 1]:
        cpu_phase = [d for d in cpu_iter if d['phase'] == phase]
        gpu_phase = [d for d in gpu_iter if d['phase'] == phase]
        
        if cpu_phase and gpu_phase:
            cpu = cpu_phase[0]
            gpu = gpu_phase[0]
            
            print(f"\nPhase {phase}:")
            if 'dof' in gpu:
                print(f"  GPU DOF[3,4]: {gpu['dof'][3]:.6f}, {gpu['dof'][4]:.6f}")
            
            print(f"  CPU Hessian: [{cpu['hessian'][0]:.3e}, {cpu['hessian'][1]:.3e}, {cpu['hessian'][2]:.3e}]")
            print(f"  GPU Hessian: [{gpu['hessian'][0]:.3e}, {gpu['hessian'][1]:.3e}, {gpu['hessian'][2]:.3e}]")
            
            # Check differences
            diffs = [abs(cpu['hessian'][i] - gpu['hessian'][i]) for i in range(3)]
            max_diff = max(diffs)
            
            if max_diff > 1e-6:
                print(f"  ⚠️  DIFFERENCE: max={max_diff:.3e}")
            else:
                print(f"  ✓ Values match")

# Also look for multiple GPU calculations with same inputs
print("\n\n=== GPU Repeated Calculations ===")
seen_inputs = {}
for item in gpu_data[:20]:  # First 20
    if 'dof' in item:
        key = f"Phase{item['phase']}_Y{item['dof'][3]:.6f}_{item['dof'][4]:.6f}"
        if key in seen_inputs:
            prev = seen_inputs[key]
            print(f"\nRepeated calculation for {key}:")
            print(f"  First:  iteration {prev['iteration']}, Hessian={prev['hessian']}")
            print(f"  Repeat: iteration {item['iteration']}, Hessian={item['hessian']}")
        else:
            seen_inputs[key] = item