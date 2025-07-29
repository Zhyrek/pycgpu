#!/usr/bin/env python
"""Test GPU with LIQUID and ALCU_PRIME phases."""

from pycalphad import Database, equilibrium
import pycalphad.variables as v
import numpy as np
import time

# Load database
db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']

# Test conditions
conditions = {
    v.T: 1273.15,  # 1000°C
    v.P: 101325,
    v.X('AL'): 0.3,
    v.X('CU'): 0.3,
    v.N: 1
}

print("Testing GPU with LIQUID and ALCU_PRIME phases")
print("="*60)

# Test 1: LIQUID only
print("\n1. Testing LIQUID phase only...")
print("   (Max Hessian line: 16,419 chars)")
start = time.time()
try:
    result_liquid = equilibrium(db, components, ['LIQUID'], conditions, 
                               calc_opts={'pdens': 10}, 
                               gpu=True, verbose=False)
    gpu_time = time.time() - start
    print(f"   ✓ SUCCESS in {gpu_time:.1f}s")
    print(f"   GM = {result_liquid.GM.values[0]:.1f} J/mol")
except Exception as e:
    print(f"   ✗ FAILED: {type(e).__name__}")
    if 'nvcc' in str(e).lower():
        print("   nvcc compilation error")

# Test 2: ALCU_PRIME only  
print("\n2. Testing ALCU_PRIME phase only...")
print("   (Max Hessian line: 8,446 chars)")
start = time.time()
try:
    result_prime = equilibrium(db, components, ['ALCU_PRIME'], conditions, 
                              calc_opts={'pdens': 10}, 
                              gpu=True, verbose=False)
    gpu_time = time.time() - start
    print(f"   ✓ SUCCESS in {gpu_time:.1f}s")
    print(f"   GM = {result_prime.GM.values[0]:.1f} J/mol")
except Exception as e:
    print(f"   ✗ FAILED: {type(e).__name__}")
    if 'nvcc' in str(e).lower():
        print("   nvcc compilation error")

# Test 3: Both phases together
print("\n3. Testing LIQUID + ALCU_PRIME together...")
start = time.time()
try:
    result_both = equilibrium(db, components, ['LIQUID', 'ALCU_PRIME'], conditions, 
                             calc_opts={'pdens': 10}, 
                             gpu=True, verbose=False)
    gpu_time = time.time() - start
    print(f"   ✓ SUCCESS in {gpu_time:.1f}s")
    print(f"   GM = {result_both.GM.values[0]:.1f} J/mol")
    
    # Show phase amounts
    for phase in ['LIQUID', 'ALCU_PRIME']:
        np_val = result_both.NP.sel(phase=phase).values[0]
        if np_val > 1e-10:
            print(f"   {phase}: {np_val:.4f}")
            
except Exception as e:
    print(f"   ✗ FAILED: {type(e).__name__}")
    if 'nvcc' in str(e).lower():
        print("   nvcc compilation error")

# Also test CPU for comparison
print("\n4. CPU comparison (same phases)...")
start = time.time()
try:
    result_cpu = equilibrium(db, components, ['LIQUID', 'ALCU_PRIME'], conditions, 
                            calc_opts={'pdens': 10}, 
                            gpu=False, verbose=False)
    cpu_time = time.time() - start
    print(f"   ✓ CPU SUCCESS in {cpu_time:.3f}s")
    print(f"   GM = {result_cpu.GM.values[0]:.1f} J/mol")
    
    # Show phase amounts
    for phase in ['LIQUID', 'ALCU_PRIME']:
        np_val = result_cpu.NP.sel(phase=phase).values[0]
        if np_val > 1e-10:
            print(f"   {phase}: {np_val:.4f}")
            
except Exception as e:
    print(f"   ✗ CPU FAILED: {type(e).__name__}")

print("\nDone.")