#!/usr/bin/env python
"""Trace the divergence for X(TI)=0.1, T=500K case."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import os
import cupy as cp

# Clear GPU cache
os.environ['CUDA_CACHE_DISABLE'] = '1'
cp.clear_memo()

# Load database
db = Database('NbTi.tdb')
components = ['NB', 'TI', 'VA']
phases = list(db.phases.keys())

# Test specific condition where divergence occurs
conditions = {
    v.T: 500,
    v.P: 101325,
    v.X('TI'): 0.1,
    v.N: 1
}

print('='*80)
print(f'Tracing divergence for X(TI)={conditions[v.X("TI")]}, T={conditions[v.T]}K')
print('='*80)

# First, run both to get final results
print('\n1. Running CPU calculation...')
cpu_result = equilibrium(db, components, phases, conditions, verbose=False)
cpu_gm = float(cpu_result.GM.values[0])
cpu_mu = cpu_result.MU.values[0]
cpu_phases = []
cpu_np = []
for i in range(len(cpu_result.Phase.values[0])):
    if cpu_result.NP.values[0][i] > 1e-10:
        cpu_phases.append(cpu_result.Phase.values[0][i])
        cpu_np.append(cpu_result.NP.values[0][i])

print(f'CPU GM: {cpu_gm:.6f}')
print(f'CPU MU: {cpu_mu}')
print(f'CPU phases: {cpu_phases}')
print(f'CPU NP: {cpu_np}')

print('\n2. Running GPU calculation...')
gpu_result = equilibrium(db, components, phases, conditions, verbose=False, gpu=True)
gpu_gm = float(gpu_result.GM.values[0])
gpu_mu = gpu_result.MU.values[0]
gpu_phases = []
gpu_np = []
for i in range(len(gpu_result.Phase.values[0])):
    if gpu_result.NP.values[0][i] > 1e-10:
        gpu_phases.append(gpu_result.Phase.values[0][i])
        gpu_np.append(gpu_result.NP.values[0][i])

print(f'GPU GM: {gpu_gm:.6f}')
print(f'GPU MU: {gpu_mu}')
print(f'GPU phases: {gpu_phases}')
print(f'GPU NP: {gpu_np}')

print(f'\nGM difference: {abs(cpu_gm - gpu_gm):.6f}')
print(f'MU[NB] difference: {abs(cpu_mu[0] - gpu_mu[0]):.6f}')
print(f'MU[TI] difference: {abs(cpu_mu[1] - gpu_mu[1]):.6f}')

# Now let's trace the calculation with verbose output
print('\n' + '='*80)
print('3. Detailed CPU trace:')
print('='*80)

# Monkey patch to add more debug output
import pycalphad.core.eqsolver
original_solve = pycalphad.core.eqsolver._solve_eq_at_conditions

def debug_solve(spec, state, verbose=False):
    if verbose:
        print(f'\n[CPU DEBUG] Iteration {state.iteration}:')
        print(f'  Number of phases: {state.num_free_stable_compsets}')
        for i in range(state.num_free_stable_compsets):
            idx = state.free_stable_compset_indices[i]
            cs = state.compsets[idx]
            print(f'  Phase {i}: {cs.phase_record.phase_name}')
            print(f'    Energy: {cs.energy:.6f}')
            print(f'    NP: {cs.NP:.6f}')
            print(f'    DOF: {cs.dof[:cs.phase_record.phase_dof]}')
    
    return original_solve(spec, state, verbose)

pycalphad.core.eqsolver._solve_eq_at_conditions = debug_solve

cpu_result_verbose = equilibrium(db, components, phases, conditions, verbose=True)

# Restore original
pycalphad.core.eqsolver._solve_eq_at_conditions = original_solve

print('\n' + '='*80)
print('4. Key observations:')
print('='*80)

# Check if it's really in miscibility gap
if len(cpu_phases) == 2 and cpu_phases[0] == cpu_phases[1]:
    print(f'✓ System is in miscibility gap with two {cpu_phases[0]} phases')
else:
    print(f'✗ System phase configuration: {cpu_phases}')

# Check site fractions
print('\nCPU site fractions:')
for i, phase in enumerate(cpu_phases):
    y_vals = []
    for j in range(len(components)-1):  # Exclude VA
        key = f'Y({phase},{j},{components[j]})'
        if key in cpu_result:
            val = cpu_result[key].values[0][i]
            y_vals.append(f'{components[j]}={val:.6f}')
    print(f'  {phase}: {", ".join(y_vals)}')

print('\nGPU site fractions:')
for i, phase in enumerate(gpu_phases):
    y_vals = []
    for j in range(len(components)-1):  # Exclude VA
        key = f'Y({phase},{j},{components[j]})'
        if key in gpu_result:
            val = gpu_result[key].values[0][i]
            y_vals.append(f'{components[j]}={val:.6f}')
    print(f'  {phase}: {", ".join(y_vals)}')

# The main difference could be in:
# 1. Initial guess
# 2. Hessian calculation
# 3. Gradient calculation
# 4. Convergence criteria

print('\n5. Hypothesis for divergence:')
print('- Small differences in Hessian mapping accumulate over iterations')
print('- The miscibility gap makes the system sensitive to small changes')
print('- Check if both converge to local minima with similar but not identical energies')