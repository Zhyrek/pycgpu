#!/usr/bin/env python
"""Final test of 21 phases CPU vs GPU comparison."""

from pycalphad import Database, equilibrium
import pycalphad.variables as v
import time

# Test with ALL 21 phases
dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']
phases = list(dbf.phases.keys())  # ALL phases

print('=' * 60)
print(f'Testing with ALL {len(phases)} phases')
print('=' * 60)

conditions = {
    v.T: 800,
    v.P: 101325,
    v.X('AL'): 0.3,
    v.X('CU'): 0.3
}

# Test GPU
print('\nRunning GPU calculation with 21 phases...')
start = time.time()
try:
    result = equilibrium(dbf, comps, phases, conditions, verbose=False, gpu=True)
    gpu_time = time.time() - start
    gm = float(result.GM.values[0,0,0])
    phases_present = [p for p in result.Phase.values[0,0,0,:] if p != '']
    print(f'GPU SUCCESS!')
    print(f'  GM = {gm:.2f} J/mol')
    print(f'  Time = {gpu_time:.2f} seconds')
    print(f'  Phases in equilibrium: {phases_present}')
    gpu_gm = gm
    gpu_phases = phases_present
except Exception as e:
    print(f'GPU Failed: {e}')
    gpu_gm = None
    gpu_phases = None

# Test CPU
print('\nRunning CPU calculation with 21 phases...')
start = time.time()
try:
    result = equilibrium(dbf, comps, phases, conditions, verbose=False, gpu=False)
    cpu_time = time.time() - start
    gm = float(result.GM.values[0,0,0])
    phases_present = [p for p in result.Phase.values[0,0,0,:] if p != '']
    print(f'CPU SUCCESS!')
    print(f'  GM = {gm:.2f} J/mol')
    print(f'  Time = {cpu_time:.2f} seconds')
    print(f'  Phases in equilibrium: {phases_present}')
    cpu_gm = gm
    cpu_phases = phases_present
except Exception as e:
    print(f'CPU Failed: {e}')
    cpu_gm = None
    cpu_phases = None

# Compare results
if gpu_gm is not None and cpu_gm is not None:
    diff = abs(gpu_gm - cpu_gm)
    print(f'\n' + '=' * 60)
    print('COMPARISON:')
    print(f'  GM difference: {diff:.6f} J/mol')
    if diff < 1.0:
        print('  ✓ Results match within tolerance!')
    else:
        print('  ✗ Results differ significantly')
    
    # Compare phases
    if set(gpu_phases) == set(cpu_phases):
        print('  ✓ Same phases in equilibrium')
    else:
        print('  ✗ Different phases in equilibrium')
        print(f'    GPU phases: {gpu_phases}')
        print(f'    CPU phases: {cpu_phases}')
    
    if gpu_time > 0 and cpu_time > 0:
        speedup = cpu_time / gpu_time
        print(f'  Speedup: {speedup:.2f}x')
    print('=' * 60)