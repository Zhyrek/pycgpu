#!/usr/bin/env python
"""Test if the phase assemblage affects solver convergence."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Load database
dbf = Database('important_tests/AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
phases = filter_phases(dbf, comps)

print(f"Available phases: {phases}")

# Test the full batch to see phase assemblages
cond_full = {
    v.X('BI'): [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    v.T: [400, 500, 600, 700],
    v.P: 101325
}

# Run CPU calculation to get correct phase assemblages
result_cpu = equilibrium(dbf, comps, phases, cond_full, gpu=False, verbose=False)

# Map condition index to grid coordinates
def idx_to_grid(idx):
    t_idx = idx // 8
    x_idx = idx % 8
    return t_idx, x_idx

# Check phase assemblages for all conditions
print("\nPhase assemblages for all conditions:")
print("="*60)

phase_assemblages = {}
for idx in range(32):
    t_idx, x_idx = idx_to_grid(idx)
    phases_at_idx = result_cpu.Phase.isel(T=t_idx, X_BI=x_idx).values
    np_at_idx = result_cpu.NP.isel(T=t_idx, X_BI=x_idx).values
    
    active_phases = []
    for phase, amount in zip(phases_at_idx.flatten(), np_at_idx.flatten()):
        if not np.isnan(amount) and amount > 0:
            active_phases.append(phase)
    
    phase_key = tuple(sorted(active_phases))
    if phase_key not in phase_assemblages:
        phase_assemblages[phase_key] = []
    phase_assemblages[phase_key].append(idx)
    
    marker = " **FAILS**" if idx in [10, 17] else ""
    print(f"Condition {idx:2d}: {' + '.join(active_phases)}{marker}")

# Group by phase assemblage
print("\nConditions grouped by phase assemblage:")
print("="*60)
for phases, indices in sorted(phase_assemblages.items()):
    print(f"{' + '.join(phases)}: conditions {indices}")
    
# Check if failing conditions share a phase assemblage
print("\nAnalyzing failing conditions 10 and 17:")
for idx in [10, 17]:
    t_idx, x_idx = idx_to_grid(idx)
    t_val = cond_full[v.T][t_idx]
    x_val = cond_full[v.X('BI')][x_idx]
    
    phases_at_idx = result_cpu.Phase.isel(T=t_idx, X_BI=x_idx).values
    np_at_idx = result_cpu.NP.isel(T=t_idx, X_BI=x_idx).values
    
    print(f"\nCondition {idx} (X(BI)={x_val}, T={t_val}K):")
    for phase, amount in zip(phases_at_idx.flatten(), np_at_idx.flatten()):
        if not np.isnan(amount) and amount > 0:
            print(f"  {phase}: {amount:.6f}")