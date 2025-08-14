#!/usr/bin/env python
"""Test 21 phases with working compilation."""

from pycalphad import Database, equilibrium
import pycalphad.variables as v
import time

dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']
phases = list(dbf.phases.keys())  # ALL 21 phases -> 19 models

print(f'Testing with ALL {len(phases)} phases')
print('=' * 60)

conditions = {
    v.T: 800,
    v.P: 101325,
    v.X('AL'): 0.3,
    v.X('CU'): 0.3
}

# Test GPU 
print('\nGPU calculation:')
start = time.time()
try:
    result = equilibrium(dbf, comps, phases, conditions, verbose=False, gpu=True)
    gpu_time = time.time() - start
    gm = result.GM.values.item()
    phases_present = [p for p in result.Phase.values[0,0,0,:] if p != '']
    print(f'  SUCCESS! GM = {gm:.2f} J/mol')
    print(f'  Time: {gpu_time:.2f} seconds')
    print(f'  Phases in equilibrium: {phases_present}')
    gpu_gm = gm
    gpu_phases = phases_present
except Exception as e:
    print(f'  Failed: {e}')
    gpu_gm = None
    gpu_phases = None

# Test CPU
print('\nCPU calculation:')
start = time.time()
try:
    result = equilibrium(dbf, comps, phases, conditions, verbose=False, gpu=False)
    cpu_time = time.time() - start
    gm = result.GM.values.item()
    phases_present = [p for p in result.Phase.values[0,0,0,:] if p != '']
    print(f'  SUCCESS! GM = {gm:.2f} J/mol')
    print(f'  Time: {cpu_time:.2f} seconds')
    print(f'  Phases in equilibrium: {phases_present}')
    cpu_gm = gm
    cpu_phases = phases_present
except Exception as e:
    print(f'  Failed: {e}')
    cpu_gm = None
    cpu_phases = None

# Compare
if gpu_gm is not None and cpu_gm is not None:
    print('\n' + '=' * 60)
    print('COMPARISON:')
    diff = abs(gpu_gm - cpu_gm)
    print(f'  GM difference: {diff:.6f} J/mol')
    if diff < 1.0:
        print('  ✓ GM values match within tolerance!')
    else:
        print('  ✗ GM values differ')
    
    if set(gpu_phases) == set(cpu_phases):
        print('  ✓ Same phases in equilibrium')
    else:
        print('  ✗ Different phases')
        
    print(f'  GPU time: {gpu_time:.2f}s')
    print(f'  CPU time: {cpu_time:.2f}s')
    if gpu_time > 0:
        print(f'  Speedup: {cpu_time/gpu_time:.2f}x')
    print('=' * 60)