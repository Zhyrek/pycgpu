#!/usr/bin/env python
"""Analyze the GPU Hessian calculation pattern."""

import os
import sys
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')
import subprocess
import re

# Run test and capture output
env = os.environ.copy()
env['PYCALPHAD_DEBUG_CATEGORIES'] = 'HESSIAN'

proc = subprocess.run([sys.executable, 'trace_hessian_inputs_test.py'], 
                     env=env, capture_output=True, text=True)

output = proc.stdout

print("=== GPU Hessian Calculation Analysis ===\n")

# Extract all GPU Hessian calculations with inputs and outputs
gpu_calcs = []
lines = output.split('\n')

for i, line in enumerate(lines):
    if '[GPU FORMULAHESS INPUT]' in line:
        match = re.search(r'Phase (\d+) iteration (\d+), DOF: (.+)', line)
        if match:
            phase = int(match.group(1))
            iter_num = int(match.group(2))
            dof_str = match.group(3)
            dof_vals = [float(x) for x in dof_str.split()]
            
            # Get Hessian values from next lines
            hess_vals = []
            for j in range(i+1, min(i+10, len(lines))):
                if '[0]' in lines[j] and 'idx=' in lines[j]:
                    vals = re.findall(r'(\d+\.\d+e[+-]\d+)', lines[j])
                    hess_vals.extend([float(v) for v in vals])
                elif '[1]' in lines[j] and 'idx=' in lines[j]:
                    vals = re.findall(r'(\d+\.\d+e[+-]\d+)', lines[j])
                    hess_vals.extend([float(v) for v in vals])
            
            gpu_calcs.append({
                'phase': phase,
                'iteration': iter_num,
                'dof': dof_vals,
                'hessian': hess_vals
            })

# Group by unique inputs
unique_inputs = {}
for calc in gpu_calcs[:30]:  # First 30 calculations
    if len(calc['dof']) >= 5 and len(calc['hessian']) >= 4:
        # Key based on site fractions only (DOF[3] and DOF[4])
        key = f"Phase{calc['phase']}_Y({calc['dof'][3]:.6f},{calc['dof'][4]:.6f})"
        
        if key not in unique_inputs:
            unique_inputs[key] = []
        unique_inputs[key].append(calc)

# Show repeated calculations
print("GPU Hessian calculations grouped by input site fractions:\n")

for key, calcs in unique_inputs.items():
    if len(calcs) > 1:
        print(f"\n{key}: {len(calcs)} calculations")
        for i, calc in enumerate(calcs):
            print(f"  [{i}] iter={calc['iteration']}, H[0,0]={calc['hessian'][0]:.3e}, H[1,1]={calc['hessian'][3]:.3e}")
        
        # Check if outputs are the same
        first_hess = calcs[0]['hessian']
        all_same = all(abs(c['hessian'][0] - first_hess[0]) < 1e-10 and 
                      abs(c['hessian'][3] - first_hess[3]) < 1e-10 
                      for c in calcs[1:])
        
        if all_same:
            print(f"  ⚠️  All {len(calcs)} calculations returned IDENTICAL Hessian values!")
        else:
            print(f"  ✓ Hessian values differ between calculations")

# Now check the actual Hessian values for early iterations
print("\n\n=== First Few Iterations ===")
for iter_num in range(3):
    iter_calcs = [c for c in gpu_calcs if c['iteration'] == iter_num and len(c['hessian']) >= 4]
    if iter_calcs:
        print(f"\nIteration {iter_num}:")
        for calc in iter_calcs[:2]:  # First 2 phases
            print(f"  Phase {calc['phase']}: Y=({calc['dof'][3]:.6f}, {calc['dof'][4]:.6f}) -> H[0,0]={calc['hessian'][0]:.3e}, H[1,1]={calc['hessian'][3]:.3e}")