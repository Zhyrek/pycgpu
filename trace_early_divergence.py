#!/usr/bin/env python
"""Trace CPU vs GPU values in early iterations to find where divergence starts."""

import os
import sys
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import numpy as np
import subprocess
import re

# Run both CPU and GPU with debug output
env = os.environ.copy()
env['PYCALPHAD_DEBUG_CATEGORIES'] = 'HESSIAN'

# Create test case
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']  
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.01, v.T: 1000, v.P: 101325}

print("=== Comparing CPU vs GPU Early Iterations ===\n")

# Run test and capture output
proc = subprocess.run([sys.executable, 'test_cpu_gpu_comparison.py'], 
                     env=env, capture_output=True, text=True)

output = proc.stdout

# Extract key values for first 3 iterations
cpu_data = {0: {}, 1: {}, 2: {}}
gpu_data = {0: {}, 1: {}, 2: {}}

lines = output.split('\n')

# Track current mode and iteration
current_mode = None
current_iter = -1

for i, line in enumerate(lines):
    # Detect mode
    if '--- CPU Calculation ---' in line:
        current_mode = 'CPU'
    elif '--- GPU Calculation ---' in line:
        current_mode = 'GPU'
    
    # Track iteration
    if 'recompute() - iteration' in line:
        match = re.search(r'iteration (\d+):', line)
        if match:
            current_iter = int(match.group(1))
    
    # Extract key values
    if current_mode and 0 <= current_iter <= 2:
        # Phase amounts
        if 'sum(phase_amt) =' in line:
            match = re.search(r'sum\(phase_amt\) = ([\d.e+-]+)', line)
            if match:
                if current_mode == 'CPU':
                    cpu_data[current_iter]['sum_phase_amt'] = float(match.group(1))
                else:
                    gpu_data[current_iter]['sum_phase_amt'] = float(match.group(1))
        
        # Site fractions from formulahess input
        if 'FORMULAHESS INPUT' in line and 'Phase 0' in line:
            match = re.search(r'DOF: ([\d.e+-]+) ([\d.e+-]+) ([\d.e+-]+) ([\d.e+-]+) ([\d.e+-]+)', line)
            if match:
                y_nb = float(match.group(4))
                y_ti = float(match.group(5))
                if current_mode == 'CPU':
                    cpu_data[current_iter]['Y_NB'] = y_nb
                    cpu_data[current_iter]['Y_TI'] = y_ti
                else:
                    gpu_data[current_iter]['Y_NB'] = y_nb
                    gpu_data[current_iter]['Y_TI'] = y_ti
        
        # Hessian values
        if '[0] ' in line and 'idx=' in line:
            hess_vals = re.findall(r'([\d.]+e[+-]\d+)', line)
            if len(hess_vals) >= 2 and current_iter in (0, 1, 2):
                key = 'H_00' if '[0]' in line else 'H_11'
                if current_mode == 'CPU':
                    if 'cpu_hess' not in cpu_data[current_iter]:
                        cpu_data[current_iter]['cpu_hess'] = {}
                    cpu_data[current_iter]['cpu_hess'][key] = float(hess_vals[0])
                else:
                    if 'gpu_hess' not in gpu_data[current_iter]:
                        gpu_data[current_iter]['gpu_hess'] = {}
                    gpu_data[current_iter]['gpu_hess'][key] = float(hess_vals[0])
        
        # Phase energies
        if 'phase_0_energy:' in line:
            match = re.search(r'phase_0_energy: ([-\d.e+]+)', line)
            if match:
                if current_mode == 'CPU':
                    cpu_data[current_iter]['energy'] = float(match.group(1))
                else:
                    gpu_data[current_iter]['energy'] = float(match.group(1))
        
        # Equilibrium solution (phase amounts)
        if 'Equilibrium solution at iteration' in line and str(current_iter) in line:
            # Look for the solution values in next lines
            for j in range(i+1, min(i+5, len(lines))):
                if 'delta_np[' in lines[j] or 'GPU DEBUG: Equilibrium solution' in lines[j]:
                    # Extract the phase amount changes
                    vals = re.findall(r'([-\d.e+]+)', lines[j])
                    if len(vals) >= 4:
                        if current_mode == 'CPU':
                            cpu_data[current_iter]['delta_phase0'] = float(vals[-2])
                            cpu_data[current_iter]['delta_phase1'] = float(vals[-1])
                        else:
                            gpu_data[current_iter]['delta_phase0'] = float(vals[-2])
                            gpu_data[current_iter]['delta_phase1'] = float(vals[-1])
                    break

# Compare iterations
print("Note: Looking for first divergence between CPU and GPU\n")

for iter_num in range(3):
    print(f"\n=== Iteration {iter_num} ===")
    
    cpu = cpu_data[iter_num]
    gpu = gpu_data[iter_num]
    
    # Compare site fractions
    if 'Y_NB' in cpu and 'Y_NB' in gpu:
        y_nb_diff = abs(cpu['Y_NB'] - gpu['Y_NB'])
        y_ti_diff = abs(cpu['Y_TI'] - gpu['Y_TI'])
        print(f"Site fractions:")
        print(f"  CPU: Y(NB)={cpu['Y_NB']:.6f}, Y(TI)={cpu['Y_TI']:.6f}")
        print(f"  GPU: Y(NB)={gpu['Y_NB']:.6f}, Y(TI)={gpu['Y_TI']:.6f}")
        print(f"  Diff: Y(NB)={y_nb_diff:.2e}, Y(TI)={y_ti_diff:.2e}")
        
        if y_nb_diff > 1e-10 or y_ti_diff > 1e-10:
            print("  ⚠️  SITE FRACTIONS DIVERGED!")
    
    # Compare energies
    if 'energy' in cpu and 'energy' in gpu:
        energy_diff = abs(cpu['energy'] - gpu['energy'])
        print(f"\nPhase energy:")
        print(f"  CPU: {cpu['energy']:.6e}")
        print(f"  GPU: {gpu['energy']:.6e}")
        print(f"  Diff: {energy_diff:.2e}")
        
        if energy_diff > 1e-6:
            print("  ⚠️  ENERGIES DIVERGED!")
    
    # Compare phase amount changes
    if 'delta_phase0' in cpu and 'delta_phase0' in gpu:
        delta0_diff = abs(cpu['delta_phase0'] - gpu['delta_phase0'])
        delta1_diff = abs(cpu['delta_phase1'] - gpu['delta_phase1'])
        print(f"\nPhase amount changes:")
        print(f"  CPU: Δphase0={cpu['delta_phase0']:.6e}, Δphase1={cpu['delta_phase1']:.6e}")
        print(f"  GPU: Δphase0={gpu['delta_phase0']:.6e}, Δphase1={gpu['delta_phase1']:.6e}")
        print(f"  Diff: {delta0_diff:.2e}, {delta1_diff:.2e}")
        
        if delta0_diff > 1e-10 or delta1_diff > 1e-10:
            print("  ⚠️  PHASE UPDATES DIVERGED!")
    
    # Summary for this iteration
    diverged = False
    if 'Y_NB' in cpu and 'Y_NB' in gpu:
        if abs(cpu['Y_NB'] - gpu['Y_NB']) > 1e-10:
            diverged = True
    
    if diverged:
        print(f"\n>>> DIVERGENCE DETECTED AT ITERATION {iter_num} <<<")
        break
    else:
        print(f"\n✓ CPU and GPU still match at iteration {iter_num}")

print("\n\n=== Summary ===")
print("The divergence analysis shows where CPU and GPU first differ.")