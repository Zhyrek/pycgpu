#!/usr/bin/env python
"""Test if removing the extra consolidation fixed the single-phase region issue."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import warnings
warnings.filterwarnings("ignore")

# Load database and set up calculation
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

print("TESTING CONSOLIDATION FIX")
print("=" * 60)

# Test a previously failing single-phase condition
conditions = {v.X('TI'): 0.1, v.T: 600, v.P: 101325}

# Run CPU calculation
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
cpu_gm = result_cpu.GM.values.flatten()[0]
cpu_phases = [(p, np) for p, np in zip(result_cpu.Phase.values.flatten(), 
                                       result_cpu.NP.values.flatten()) if np > 1e-6]

print(f"\nCPU Result for X(TI)=0.1, T=600K:")
print(f"  GM = {cpu_gm:.2f} J/mol")
print(f"  Phases: {len(cpu_phases)} - {cpu_phases}")

# Run GPU calculation
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
gpu_gm = result_gpu.GM.values.flatten()[0]
gpu_phases = [(p, np) for p, np in zip(result_gpu.Phase.values.flatten(), 
                                       result_gpu.NP.values.flatten()) if np > 1e-6]

print(f"\nGPU Result for X(TI)=0.1, T=600K:")
print(f"  GM = {gpu_gm:.2f} J/mol")
print(f"  Phases: {len(gpu_phases)} - {gpu_phases}")

# Compare
gm_diff = gpu_gm - cpu_gm
print(f"\nDifference: {gm_diff:.2f} J/mol")

if abs(gm_diff) < 1.0:
    print("✅ SUCCESS: GPU GM matches CPU within 1 J/mol tolerance!")
else:
    print(f"❌ FAILED: GM differs by {abs(gm_diff):.2f} J/mol")

# Test more conditions
print("\n" + "=" * 60)
print("Testing additional conditions:")

test_conditions = [
    {v.X('TI'): 0.5, v.T: 600, v.P: 101325},  # Two-phase (should still pass)
    {v.X('TI'): 0.9, v.T: 600, v.P: 101325},  # Single-phase (was passing)
    {v.X('TI'): 0.1, v.T: 700, v.P: 101325},  # Single-phase (was failing)
]

for cond in test_conditions:
    x_ti = cond[v.X('TI')]
    temp = cond[v.T]
    
    result_cpu = equilibrium(dbf, comps, phases, cond, gpu=False, verbose=False)
    result_gpu = equilibrium(dbf, comps, phases, cond, gpu=True, verbose=False)
    
    cpu_gm = result_cpu.GM.values.flatten()[0]
    gpu_gm = result_gpu.GM.values.flatten()[0]
    gm_diff = gpu_gm - cpu_gm
    
    status = "✅ PASS" if abs(gm_diff) < 1.0 else "❌ FAIL"
    print(f"\nX(TI)={x_ti}, T={temp}K: {status}")
    print(f"  CPU GM: {cpu_gm:.2f}, GPU GM: {gpu_gm:.2f}")
    print(f"  Difference: {gm_diff:.2f} J/mol")