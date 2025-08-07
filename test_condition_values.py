#!/usr/bin/env python
"""Check the exact condition values being passed to the GPU."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Load database
dbf = Database('important_tests/AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
phases = filter_phases(dbf, comps)

# Test conditions
x_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
t_values = [400, 500, 600, 700]

print("Condition mapping for 32-condition test:")
print("="*60)

# Generate all conditions
idx = 0
for t in t_values:
    for x in x_values:
        marker = " **FAILS**" if idx in [10, 17] else ""
        print(f"Condition {idx:2d}: X(BI)={x}, T={t}K{marker}")
        idx += 1

print(f"\nCondition 10: X(BI)={x_values[10 % 8]}, T={t_values[10 // 8]}")
print(f"Condition 17: X(BI)={x_values[17 % 8]}, T={t_values[17 // 8]}")

# Verify by running CPU calculation
cond = {
    v.X('BI'): x_values,
    v.T: t_values,
    v.P: 101325
}

result_cpu = equilibrium(dbf, comps, phases, cond, gpu=False, verbose=False)
print(f"\nCPU results shape: {result_cpu.GM.shape}")
print(f"CPU results flattened shape: {result_cpu.GM.values.flatten().shape}")

# Get GM values
gm_cpu = result_cpu.GM.values.flatten()
print(f"\nCondition 10 CPU: GM={gm_cpu[10]:.6f}")
print(f"Condition 17 CPU: GM={gm_cpu[17]:.6f}")

# Also check phase information
print("\nPhase information from CPU:")
for idx in [10, 17]:
    t_idx = idx // 8
    x_idx = idx % 8
    print(f"\nCondition {idx} (T={t_values[t_idx]}, X(BI)={x_values[x_idx]}):")
    # Get phase info for this condition
    phases_at_idx = result_cpu.Phase.isel(T=t_idx, X_BI=x_idx).values
    np_at_idx = result_cpu.NP.isel(T=t_idx, X_BI=x_idx).values
    for i, (phase, amount) in enumerate(zip(phases_at_idx, np_at_idx)):
        if not np.isnan(amount) and amount > 0:
            print(f"  {phase}: {amount:.6f}")