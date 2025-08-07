#\!/usr/bin/env python
"""Test to understand the thread pattern better."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Load database
dbf = Database('important_tests/AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
phases = filter_phases(dbf, comps)

# Run tests with different numbers of conditions to see the pattern
test_sizes = [8, 16, 24, 32, 40]

for n_cond in test_sizes:
    print(f"\n=== Testing with {n_cond} conditions ===")
    
    # Create conditions
    n_x = min(8, n_cond)
    n_t = (n_cond + n_x - 1) // n_x  # Ceiling division
    
    x_values = np.linspace(0.1, 0.8, n_x)[:n_x]
    t_values = np.linspace(400, 700, n_t)[:n_t]
    
    cond = {
        v.X('BI'): x_values.tolist(),
        v.T: t_values.tolist(),
        v.P: 101325
    }
    
    # Run calculations
    try:
        result_gpu = equilibrium(dbf, comps, phases, cond, gpu=True, verbose=False)
        result_cpu = equilibrium(dbf, comps, phases, cond, gpu=False, verbose=False)
        
        gm_gpu = result_gpu.GM.values.flatten()
        gm_cpu = result_cpu.GM.values.flatten()
        
        # Check for failures
        failures = []
        for i in range(min(len(gm_gpu), len(gm_cpu))):
            diff = abs(gm_cpu[i] - gm_gpu[i])
            if diff > 1e-3:
                failures.append((i, diff))
        
        print(f"Failures: {len(failures)}")
        if failures:
            for idx, diff in failures[:5]:  # Show first 5
                print(f"  Thread {idx}: diff = {diff:.6f}, {idx} % 7 = {idx % 7}")
                
    except Exception as e:
        print(f"Error: {e}")
EOF < /dev/null
