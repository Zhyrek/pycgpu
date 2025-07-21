#!/usr/bin/env python
"""Test phase consolidation behavior in single-phase vs two-phase regions."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import warnings
warnings.filterwarnings("ignore")

# Load database and set up calculation
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

print("TESTING PHASE CONSOLIDATION: Single-phase vs Two-phase regions")
print("=" * 80)

# Test conditions that PASS (two-phase region)
print("\nTWO-PHASE REGION (PASSES):")
print("-" * 40)
conditions_pass = {v.X('TI'): 0.5, v.T: 600, v.P: 101325}

result_cpu = equilibrium(dbf, comps, phases, conditions_pass, gpu=False, verbose=False)
result_gpu = equilibrium(dbf, comps, phases, conditions_pass, gpu=True, verbose=False)

cpu_phases = [(p, np) for p, np in zip(result_cpu.Phase.values.flatten(), 
                                       result_cpu.NP.values.flatten()) if np > 1e-6]
gpu_phases = [(p, np) for p, np in zip(result_gpu.Phase.values.flatten(), 
                                       result_gpu.NP.values.flatten()) if np > 1e-6]

print(f"X(TI)=0.5, T=600K:")
print(f"  CPU: {len(cpu_phases)} phases - {cpu_phases}")
print(f"  GPU: {len(gpu_phases)} phases - {gpu_phases}")
print(f"  GM difference: {float(result_gpu.GM.values[0] - result_cpu.GM.values[0]):.6f} J/mol")

# Test conditions that FAIL (likely single-phase region)
print("\n\nSINGLE-PHASE REGION (FAILS):")
print("-" * 40)
conditions_fail = {v.X('TI'): 0.1, v.T: 600, v.P: 101325}

result_cpu = equilibrium(dbf, comps, phases, conditions_fail, gpu=False, verbose=False)
result_gpu = equilibrium(dbf, comps, phases, conditions_fail, gpu=True, verbose=False)

cpu_phases = [(p, np) for p, np in zip(result_cpu.Phase.values.flatten(), 
                                       result_cpu.NP.values.flatten()) if np > 1e-6]
gpu_phases = [(p, np) for p, np in zip(result_gpu.Phase.values.flatten(), 
                                       result_gpu.NP.values.flatten()) if np > 1e-6]

print(f"X(TI)=0.1, T=600K:")
print(f"  CPU: {len(cpu_phases)} phases - {cpu_phases}")
print(f"  GPU: {len(gpu_phases)} phases - {gpu_phases}")
print(f"  GM difference: {float(result_gpu.GM.values[0] - result_cpu.GM.values[0]):.6f} J/mol")

# Test more conditions to see the pattern
print("\n\nADDITIONAL TESTS:")
print("-" * 40)

test_conditions = [
    # Likely single-phase (Ti-rich)
    {v.X('TI'): 0.9, v.T: 500, v.P: 101325},  # PASSES
    {v.X('TI'): 0.9, v.T: 600, v.P: 101325},  # PASSES
    {v.X('TI'): 0.1, v.T: 700, v.P: 101325},  # FAILS
    # Two-phase region
    {v.X('TI'): 0.5, v.T: 500, v.P: 101325},  # PASSES
    {v.X('TI'): 0.5, v.T: 700, v.P: 101325},  # PASSES
]

for cond in test_conditions:
    x_ti = cond[v.X('TI')]
    temp = cond[v.T]
    
    result_cpu = equilibrium(dbf, comps, phases, cond, gpu=False, verbose=False)
    result_gpu = equilibrium(dbf, comps, phases, cond, gpu=True, verbose=False)
    
    cpu_phases = [(p, np) for p, np in zip(result_cpu.Phase.values.flatten(), 
                                           result_cpu.NP.values.flatten()) if np > 1e-6]
    gpu_phases = [(p, np) for p, np in zip(result_gpu.Phase.values.flatten(), 
                                           result_gpu.NP.values.flatten()) if np > 1e-6]
    
    gm_diff = result_gpu.GM.values[0] - result_cpu.GM.values[0]
    status = "PASS" if abs(gm_diff) < 1.0 else "FAIL"
    
    print(f"\nX(TI)={x_ti}, T={temp}K: {status}")
    print(f"  CPU phases: {len(cpu_phases)}, GPU phases: {len(gpu_phases)}")
    print(f"  GM diff: {gm_diff:.3f} J/mol")
    
    # Show phase details if different number of phases
    if len(cpu_phases) != len(gpu_phases):
        print(f"  WARNING: Different number of phases!")
        print(f"  CPU: {cpu_phases}")
        print(f"  GPU: {gpu_phases}")

print("\n" + "=" * 80)
print("KEY OBSERVATION:")
print("Failures seem to correlate with single-phase regions where consolidation")
print("should reduce multiple BCC_A2 phases to a single phase.")
print("=" * 80)