#!/usr/bin/env python
"""Simple Al-Cu-Fe comparison using multi-condition range format."""

import numpy as np
from pycalphad import Database, equilibrium
import pycalphad.variables as v
import warnings
import os
import time

# Suppress all warnings and debug output
warnings.filterwarnings('ignore')
os.environ['PYCALPHAD_CPU_DEBUG'] = '0'

# Load database
db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']
phases = list(db.phases.keys())

print("="*60)
print("Al-Cu-Fe System Comparison (Fixed GPU Code)")
print("="*60)
print(f"Temperature: 1000°C")
print(f"Phases: {len(phases)}")

# Simple multi-condition specification using ranges
# Pycalphad will automatically filter invalid compositions
conditions = {
    v.T: 1273.15,
    v.P: 101325,
    v.X('AL'): (0, 1, 0.1),  # 11 points
    v.X('CU'): (0, 1, 0.1),  # 11 points  
    v.N: 1
}

# CPU calculation
print("\nCPU calculation...", end='', flush=True)
start = time.time()
try:
    cpu_result = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 50})
    cpu_time = time.time() - start
    print(f" SUCCESS ({cpu_time:.1f}s)")
    print(f"  Shape: {cpu_result.GM.shape}")
    cpu_gm = cpu_result.GM.values.flatten()
    cpu_success = True
except Exception as e:
    print(f" FAILED: {str(e)[:50]}...")
    cpu_success = False

# GPU calculation
print("\nGPU calculation...", end='', flush=True)
start = time.time()
try:
    gpu_result = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 50}, gpu=True, verbose=False)
    gpu_time = time.time() - start
    print(f" SUCCESS ({gpu_time:.1f}s)")
    print(f"  Shape: {gpu_result.GM.shape}")
    gpu_gm = gpu_result.GM.values.flatten()
    gpu_success = True
except Exception as e:
    print(f" FAILED: {str(e)[:50]}...")
    gpu_success = False

# Compare
if cpu_success and gpu_success:
    print("\n" + "="*60)
    print("RESULTS:")
    
    # Basic comparison
    diff = np.abs(cpu_gm - gpu_gm)
    matching = np.sum(diff < 1.0)
    total = len(diff)
    
    print(f"Points within 1 J/mol: {matching}/{total} ({100*matching/total:.1f}%)")
    print(f"Max difference: {np.max(diff):.3f} J/mol")
    print(f"Mean difference: {np.mean(diff):.3f} J/mol")
    
    if gpu_time > 0:
        print(f"\nSpeedup: {cpu_time/gpu_time:.1f}x")
    
    # Show worst points if any
    if np.max(diff) > 1.0:
        print("\nLargest differences:")
        worst = np.argsort(diff)[-5:][::-1]
        for i in worst[:3]:
            if diff[i] > 1.0:
                print(f"  Point {i}: CPU={cpu_gm[i]:.1f}, GPU={gpu_gm[i]:.1f}, Δ={diff[i]:.1f}")

print("\n" + "="*60)
if cpu_success and gpu_success:
    if matching == total:
        print("✓ GPU matches CPU perfectly!")
    else:
        print(f"✓ GPU code generation fix is working ({100*matching/total:.1f}% match)")
else:
    print("✗ Test failed")