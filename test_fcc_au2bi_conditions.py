#!/usr/bin/env python
"""Test all conditions with FCC_A1 + AU2BI_C15 phase assemblage."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Load database
dbf = Database('important_tests/AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
phases = filter_phases(dbf, comps)

# Full batch conditions
cond_full = {
    v.X('BI'): [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    v.T: [400, 500, 600, 700],
    v.P: 101325
}

# FCC_A1 + AU2BI_C15 conditions: [0, 1, 2, 8, 9, 10, 16, 17, 18]
fcc_au2bi_conditions = [0, 1, 2, 8, 9, 10, 16, 17, 18]

print("Testing all FCC_A1 + AU2BI_C15 conditions:")
print("="*60)

# Run GPU calculation
result_gpu = equilibrium(dbf, comps, phases, cond_full, gpu=True, verbose=False)
result_cpu = equilibrium(dbf, comps, phases, cond_full, gpu=False, verbose=False)

gm_gpu = result_gpu.GM.values.flatten()
gm_cpu = result_cpu.GM.values.flatten()

# Check each FCC_A1 + AU2BI_C15 condition
failures = []
for idx in fcc_au2bi_conditions:
    t_idx = idx // 8
    x_idx = idx % 8
    t_val = cond_full[v.T][t_idx]
    x_val = cond_full[v.X('BI')][x_idx]
    
    diff = abs(gm_cpu[idx] - gm_gpu[idx])
    status = "PASS" if diff < 1e-3 else "FAIL"
    if status == "FAIL":
        failures.append(idx)
    
    print(f"Condition {idx:2d} (X={x_val}, T={t_val}K): CPU={gm_cpu[idx]:.2f}, GPU={gm_gpu[idx]:.2f}, Diff={diff:.6f} - {status}")

print(f"\nSummary: {len(failures)} out of {len(fcc_au2bi_conditions)} FCC_A1+AU2BI_C15 conditions fail")
print(f"Failing conditions: {failures}")

# Check modulo patterns
print("\nModulo 7 pattern for FCC_A1+AU2BI_C15 conditions:")
for idx in fcc_au2bi_conditions:
    mod7 = idx % 7
    status = "FAIL" if idx in failures else "PASS"
    print(f"Condition {idx}: {idx} % 7 = {mod7} - {status}")

# Check if there's a pattern in the phase amounts
print("\nPhase amounts for failing conditions:")
for idx in failures:
    t_idx = idx // 8
    x_idx = idx % 8
    
    phases_cpu = result_cpu.Phase.isel(T=t_idx, X_BI=x_idx).values.flatten()
    np_cpu = result_cpu.NP.isel(T=t_idx, X_BI=x_idx).values.flatten()
    
    print(f"\nCondition {idx}:")
    for phase, amount in zip(phases_cpu, np_cpu):
        if not np.isnan(amount) and amount > 0:
            print(f"  {phase}: {amount:.6f}")