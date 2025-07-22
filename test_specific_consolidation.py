#!/usr/bin/env python
"""Debug GPU consolidation issue for X(TI)=0.1, T=600K."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Load database and set up calculation
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test condition: X(TI)=0.1, T=600K
conditions = {v.X('TI'): 0.1, v.T: 600, v.P: 101325}

print("Testing GPU consolidation for X(TI)=0.1, T=600K")
print("="*80)

# Run GPU calculation with verbose output
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)

# Show final results
print("\n\nFinal GPU Results:")
print(f"GM: {float(result_gpu.GM.values):.6f}")

# Count phases
phase_counts = {}
for phase in phases:
    mask = result_gpu.Phase.values.flatten() == phase
    if np.any(mask):
        phase_amt = result_gpu.NP.values.flatten()[mask]
        phase_counts[phase] = phase_amt[phase_amt > 1e-6]

print(f"Active phases: {len([p for p,v in phase_counts.items() if len(v) > 0])}")
for phase, amounts in phase_counts.items():
    if len(amounts) > 0:
        print(f"  {phase}: {amounts}")

# Compare with CPU
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
gm_cpu = float(result_cpu.GM.values)
gm_gpu = float(result_gpu.GM.values)

print(f"\nCPU GM: {gm_cpu:.6f}")
print(f"GPU GM: {gm_gpu:.6f}")
print(f"Difference: {abs(gm_gpu - gm_cpu):.6f}")