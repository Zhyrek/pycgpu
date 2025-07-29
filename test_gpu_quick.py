#!/usr/bin/env python
"""Quick GPU test with timeout."""

from pycalphad import Database, equilibrium
import pycalphad.variables as v
import signal
import sys

def timeout_handler(signum, frame):
    print("\n✗ GPU calculation timed out after 30 seconds")
    print("  The GPU compilation is taking too long, likely due to complex expressions")
    sys.exit(1)

# Set timeout
signal.signal(signal.SIGALRM, timeout_handler)

# Load database
db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']

# Test conditions
conditions = {
    v.T: 1273.15,
    v.P: 101325,
    v.X('AL'): 0.3,
    v.X('CU'): 0.3,
    v.N: 1
}

print("Quick GPU test")
print("="*60)

# Test 1: LIQUID only
print("\n1. Testing LIQUID phase only...")
signal.alarm(30)  # 30 second timeout
try:
    result = equilibrium(db, components, ['LIQUID'], conditions, 
                        calc_opts={'pdens': 10}, 
                        gpu=True, verbose=False)
    signal.alarm(0)  # Cancel timeout
    print("✓ SUCCESS: LIQUID works on GPU")
except Exception as e:
    signal.alarm(0)
    print(f"✗ FAILED: {type(e).__name__}")

# Test 2: ALCU_ZETA only  
print("\n2. Testing ALCU_ZETA phase only...")
signal.alarm(30)
try:
    result = equilibrium(db, components, ['ALCU_ZETA'], conditions, 
                        calc_opts={'pdens': 10}, 
                        gpu=True, verbose=False)
    signal.alarm(0)
    print("✓ SUCCESS: ALCU_ZETA works on GPU")
except Exception as e:
    signal.alarm(0)
    print(f"✗ FAILED: {type(e).__name__}")

# Test 3: Both phases
print("\n3. Testing LIQUID + ALCU_ZETA...")
signal.alarm(30)
try:
    result = equilibrium(db, components, ['LIQUID', 'ALCU_ZETA'], conditions, 
                        calc_opts={'pdens': 10}, 
                        gpu=True, verbose=False)
    signal.alarm(0)
    print("✓ SUCCESS: Both phases work on GPU")
except Exception as e:
    signal.alarm(0)
    print(f"✗ FAILED: {type(e).__name__}")

print("\nDone.")