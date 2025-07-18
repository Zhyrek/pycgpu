#!/usr/bin/env python
"""Trace phase amounts at each iteration for CPU vs GPU."""

import os
import sys
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import numpy as np
import subprocess
import re

# Run test
proc = subprocess.run([sys.executable, 'test_cpu_gpu_comparison.py'], 
                     capture_output=True, text=True)

output = proc.stdout

print("=== Tracking Phase Amounts by Iteration ===\n")

# Extract phase amount updates
cpu_phases = {}
gpu_phases = {}
current_mode = None

for line in output.split('\n'):
    if '--- CPU Calculation ---' in line:
        current_mode = 'CPU'
    elif '--- GPU Calculation ---' in line:
        current_mode = 'GPU'
    
    # Look for phase advance messages
    if 'ADVANCE] Phase' in line and current_mode:
        # Parse: [CPU ADVANCE] Phase 0: old=5.743033e-01, delta=-1.282766e-01, step_size=1.000000e+00, actual_change=-1.282766e-01
        match = re.search(r'Phase (\d+): old=([\d.e+-]+), delta=([\d.e+-]+), step_size=([\d.e+-]+), actual_change=([\d.e+-]+)', line)
        if match:
            phase = int(match.group(1))
            old_val = float(match.group(2))
            delta = float(match.group(3))
            new_val = old_val + float(match.group(5))
            
            # Find iteration
            iter_match = re.search(r'iteration (\d+)', output[:output.find(line)][::-1][:200][::-1])
            if iter_match:
                iter_num = int(iter_match.group(1))
                
                if current_mode == 'CPU':
                    if iter_num not in cpu_phases:
                        cpu_phases[iter_num] = {}
                    cpu_phases[iter_num][phase] = {'old': old_val, 'delta': delta, 'new': new_val}
                else:
                    if iter_num not in gpu_phases:
                        gpu_phases[iter_num] = {}
                    gpu_phases[iter_num][phase] = {'old': old_val, 'delta': delta, 'new': new_val}

# Also extract initial phase amounts
for line in output.split('\n'):
    if 'InitialPhaseData' in line and 'amounts=' in line:
        match = re.search(r'amounts=\[([\d.]+) ([\d.]+)\]', line)
        if match:
            if 'CPU' in output[:output.find(line)][-100:]:
                cpu_phases[-1] = {0: {'new': float(match.group(1))}, 
                                  1: {'new': float(match.group(2))}}
            else:
                gpu_phases[-1] = {0: {'new': float(match.group(1))}, 
                                  1: {'new': float(match.group(2))}}

# Compare iterations
print("Initial phase amounts:")
if -1 in cpu_phases and -1 in gpu_phases:
    print(f"  CPU: Phase 0={cpu_phases[-1][0]['new']:.6f}, Phase 1={cpu_phases[-1][1]['new']:.6f}")
    print(f"  GPU: Phase 0={gpu_phases[-1][0]['new']:.6f}, Phase 1={gpu_phases[-1][1]['new']:.6f}")

for iter_num in range(5):
    if iter_num in cpu_phases and iter_num in gpu_phases:
        print(f"\n=== Iteration {iter_num} ===")
        
        for phase in [0, 1]:
            if phase in cpu_phases[iter_num] and phase in gpu_phases[iter_num]:
                cpu = cpu_phases[iter_num][phase]
                gpu = gpu_phases[iter_num][phase]
                
                delta_diff = abs(cpu['delta'] - gpu['delta'])
                new_diff = abs(cpu['new'] - gpu['new'])
                
                print(f"\nPhase {phase}:")
                print(f"  CPU: old={cpu['old']:.6f}, delta={cpu['delta']:+.6e}, new={cpu['new']:.6f}")
                print(f"  GPU: old={gpu['old']:.6f}, delta={gpu['delta']:+.6e}, new={gpu['new']:.6f}")
                print(f"  Difference in delta: {delta_diff:.2e}")
                print(f"  Difference in new: {new_diff:.2e}")
                
                if delta_diff > 1e-10:
                    print(f"  ⚠️  PHASE UPDATES DIVERGED!")

# Also check final result
print("\n\n=== Final Results ===")
final_cpu = None
final_gpu = None

for line in output.split('\n'):
    if 'X(TI) values:' in line:
        # Next line has the values
        idx = output.split('\n').index(line)
        if idx + 1 < len(output.split('\n')):
            next_line = output.split('\n')[idx + 1]
            if 'CPU:' in next_line:
                match = re.search(r'\[([\d.]+)', next_line)
                if match:
                    final_cpu = float(match.group(1))
            elif 'GPU:' in next_line:
                match = re.search(r'\[([\d.]+)', next_line)
                if match:
                    final_gpu = float(match.group(1))

if final_cpu and final_gpu:
    print(f"Final X(TI): CPU={final_cpu:.6f}, GPU={final_gpu:.6f}, diff={abs(final_cpu-final_gpu):.2e}")