#!/usr/bin/env python
"""Debug test to find where CPU and GPU diverge."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import os

# Clear GPU cache
os.environ['CUDA_CACHE_DISABLE'] = '1'
import cupy as cp
cp.clear_memo()

# Load database and set up conditions
db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']
phases = list(db.phases.keys())

# Test a single condition where divergence occurs
conditions = {
    v.T: 800,  # Lower temperature where divergence was noted
    v.P: 101325,
    v.X('AL'): 0.1,  # Composition where divergence occurs
    v.X('CU'): 0.1,
    v.N: 1
}

print('Testing divergence at specific condition...')
print(f'T={conditions[v.T]}, X(AL)={conditions[v.X("AL")]}, X(CU)={conditions[v.X("CU")]}')
print('='*80)

# Run CPU with verbose output
print('\nCPU Calculation:')
print('-'*40)
cpu_result = equilibrium(db, components, phases, conditions, verbose=True, calc_opts={'pdens': 50})

# Extract CPU results
cpu_phases = []
cpu_np = []
cpu_x = {}
for i in range(len(cpu_result.Phase.values[0])):
    phase = cpu_result.Phase.values[0][i]
    if isinstance(phase, str) and phase != '':
        cpu_phases.append(phase)
        cpu_np.append(cpu_result.NP.values[0][i])
        for comp in ['AL', 'CU', 'FE']:
            x_comp = cpu_result['X(' + phase + ',' + comp + ')'].values[0][i]
            cpu_x[f'{phase},{comp}'] = x_comp

print(f'\nCPU Final Result:')
print(f'  Phases: {cpu_phases}')
print(f'  NP: {cpu_np}')
print(f'  Compositions:')
for key, val in cpu_x.items():
    if any(p in key for p in cpu_phases):
        print(f'    X({key}): {val:.6f}')

print('\n' + '='*80)
print('\nGPU Calculation:')
print('-'*40)

# Run GPU with verbose output
gpu_result = equilibrium(db, components, phases, conditions, verbose=True, gpu=True, calc_opts={'pdens': 50})

# Extract GPU results
gpu_phases = []
gpu_np = []
gpu_x = {}
for i in range(len(gpu_result.Phase.values[0])):
    phase = gpu_result.Phase.values[0][i]
    if isinstance(phase, str) and phase != '':
        gpu_phases.append(phase)
        gpu_np.append(gpu_result.NP.values[0][i])
        for comp in ['AL', 'CU', 'FE']:
            x_comp = gpu_result['X(' + phase + ',' + comp + ')'].values[0][i]
            gpu_x[f'{phase},{comp}'] = x_comp

print(f'\nGPU Final Result:')
print(f'  Phases: {gpu_phases}')
print(f'  NP: {gpu_np}')
print(f'  Compositions:')
for key, val in gpu_x.items():
    if any(p in key for p in gpu_phases):
        print(f'    X({key}): {val:.6f}')

# Compare results
print('\n' + '='*80)
print('COMPARISON:')
print('-'*40)

# Check phase selection
phases_match = set(cpu_phases) == set(gpu_phases)
print(f'Phase selection matches: {phases_match}')
if not phases_match:
    print(f'  CPU phases: {cpu_phases}')
    print(f'  GPU phases: {gpu_phases}')

# Check phase amounts
np_match = True
if len(cpu_np) == len(gpu_np):
    for i, (cpu_val, gpu_val) in enumerate(zip(cpu_np, gpu_np)):
        if not np.isclose(cpu_val, gpu_val, rtol=1e-3, atol=1e-6):
            np_match = False
            print(f'  NP[{i}] differs: CPU={cpu_val:.6f}, GPU={gpu_val:.6f}, diff={abs(cpu_val-gpu_val):.6e}')
else:
    np_match = False
    print(f'  Different number of phases: CPU={len(cpu_np)}, GPU={len(gpu_np)}')

if np_match:
    print('Phase amounts match!')

# Check compositions
comp_match = True
for phase in set(cpu_phases) & set(gpu_phases):
    for comp in ['AL', 'CU', 'FE']:
        key = f'{phase},{comp}'
        if key in cpu_x and key in gpu_x:
            if not np.isclose(cpu_x[key], gpu_x[key], rtol=1e-3, atol=1e-6):
                comp_match = False
                print(f'  X({key}) differs: CPU={cpu_x[key]:.6f}, GPU={gpu_x[key]:.6f}, diff={abs(cpu_x[key]-gpu_x[key]):.6e}')

if comp_match and phases_match:
    print('Compositions match!')

print('\n' + '='*80)
if phases_match and np_match and comp_match:
    print('SUCCESS: CPU and GPU results match!')
else:
    print('ERROR: CPU and GPU results differ!')
    print('\nPossible causes:')
    print('1. Hessian indexing issue (already fixed)')
    print('2. Gradient calculation differences')
    print('3. Different convergence paths')
    print('4. Numerical precision differences in matrix operations')
print('='*80)