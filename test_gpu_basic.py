#!/usr/bin/env python
"""Basic test to check if GPU kernel runs at all."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import os
import cupy as cp

# Clear cache
os.environ['CUDA_CACHE_DISABLE'] = '1'
cp.clear_memo()

# Load database
db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']
phases = list(db.phases.keys())

# Very simple test condition
conditions = {
    v.T: 1000,
    v.P: 101325,
    v.X('AL'): 0.5,
    v.X('CU'): 0.2,
    v.N: 1
}

print('Testing basic GPU functionality...')
print(f'Conditions: T={conditions[v.T]}, X(AL)={conditions[v.X("AL")]}, X(CU)={conditions[v.X("CU")]}')

try:
    # Run CPU
    print('\nRunning CPU calculation...')
    cpu_result = equilibrium(db, components, phases, conditions, verbose=False)
    cpu_phases = [p for p in cpu_result.Phase.values[0] if isinstance(p, str) and p != '']
    print(f'CPU phases: {cpu_phases}')
    
    # Run GPU
    print('\nRunning GPU calculation...')
    gpu_result = equilibrium(db, components, phases, conditions, verbose=False, gpu=True)
    gpu_phases = [p for p in gpu_result.Phase.values[0] if isinstance(p, str) and p != '']
    print(f'GPU phases: {gpu_phases}')
    
    # Compare
    if set(cpu_phases) == set(gpu_phases):
        print('\nSUCCESS: Phase selection matches!')
    else:
        print('\nERROR: Phase selection differs!')
        
except Exception as e:
    print(f'\nERROR: {type(e).__name__}: {str(e)}')
    import traceback
    traceback.print_exc()