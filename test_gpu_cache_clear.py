#!/usr/bin/env python
"""Test if clearing GPU cache helps with constraint issues."""

import os
import sys
sys.path.insert(0, os.getcwd())

import subprocess

def run_gpu_test_with_cache_clear(x_ti):
    """Run GPU test with explicit cache clearing."""
    code = f'''
import sys, warnings, gc
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

try:
    import cupy as cp
    # Clear GPU memory
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    print(f"[DEBUG] Cleared GPU memory for X(TI)={x_ti}")
except:
    pass

# Clear any module caches
for module_name in list(sys.modules.keys()):
    if 'pycalphad' in module_name:
        if module_name in sys.modules:
            del sys.modules[module_name]

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database("NbTi.tdb")
comps = ["NB", "TI", "VA"]
phases = filter_phases(dbf, comps)

conditions = {{v.X("TI"): {x_ti}, v.T: 1000, v.P: 101325}}

result = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
x_ti_final = result.X.sel(component="TI").values.flatten()[0]
print(f"RESULT:{{x_ti_final:.12f}}")
'''
    
    try:
        proc = subprocess.run([sys.executable, '-c', code], 
                             capture_output=True, text=True, timeout=60)
        if proc.returncode == 0:
            for line in proc.stdout.split('\n'):
                if line.startswith('RESULT:'):
                    return float(line.split(':')[1])
                elif line.startswith('[DEBUG]'):
                    print(line)
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

# Test the same values that showed issues
test_values = [0.005, 0.01, 0.02]

print("Testing GPU with cache clearing between calls...")
print("=" * 50)

for x_ti in test_values:
    print(f"\nTesting X(TI) = {x_ti:.3f}")
    result = run_gpu_test_with_cache_clear(x_ti)
    if result is not None:
        print(f"  GPU result: {result:.8f}")
        
        # Show expected CPU result for comparison
        expected = x_ti  # CPU should return the constraint value exactly
        diff = abs(result - expected)
        status = "MATCH" if diff < 1e-6 else "DIFFER"
        print(f"  Expected:   {expected:.8f}")
        print(f"  Difference: {diff:.6f} [{status}]")
    else:
        print(f"  FAILED")