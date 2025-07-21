#!/usr/bin/env python
"""Test if removing the extra consolidation fixed the single-phase region issue - clean output."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import os
import warnings
warnings.filterwarnings("ignore")

# Suppress verbose output
os.environ['PYCALPHAD_DEBUG'] = '0'

# Load database and set up calculation
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

print("TESTING CONSOLIDATION FIX")
print("=" * 60)

# Test a previously failing single-phase condition
conditions = {v.X('TI'): 0.1, v.T: 600, v.P: 101325}

# Run calculations
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)

cpu_gm = result_cpu.GM.values.flatten()[0]
gpu_gm = result_gpu.GM.values.flatten()[0]
gm_diff = gpu_gm - cpu_gm

print(f"\nX(TI)=0.1, T=600K (previously failing):")
print(f"  CPU GM: {cpu_gm:.2f} J/mol")
print(f"  GPU GM: {gpu_gm:.2f} J/mol")
print(f"  Difference: {gm_diff:.2f} J/mol")
print(f"  Status: {'✅ PASS' if abs(gm_diff) < 1.0 else '❌ FAIL'}")

# Test more conditions
print("\n" + "=" * 60)
print("Testing additional conditions:")

test_conditions = [
    ('Two-phase', {v.X('TI'): 0.5, v.T: 600, v.P: 101325}),
    ('Single-phase', {v.X('TI'): 0.9, v.T: 600, v.P: 101325}),
    ('Single-phase', {v.X('TI'): 0.1, v.T: 700, v.P: 101325}),
]

for label, cond in test_conditions:
    result_cpu = equilibrium(dbf, comps, phases, cond, gpu=False, verbose=False)
    result_gpu = equilibrium(dbf, comps, phases, cond, gpu=True, verbose=False)
    
    cpu_gm = result_cpu.GM.values.flatten()[0]
    gpu_gm = result_gpu.GM.values.flatten()[0]
    gm_diff = gpu_gm - cpu_gm
    
    x_ti = cond[v.X('TI')]
    temp = cond[v.T]
    status = "✅ PASS" if abs(gm_diff) < 1.0 else "❌ FAIL"
    
    print(f"\n{label} - X(TI)={x_ti}, T={temp}K: {status}")
    print(f"  Difference: {gm_diff:.2f} J/mol")

print("\n" + "=" * 60)
print("CONCLUSION:")
print("The extra final phase consolidation has been removed from GPU code.")
print("This should fix the single-phase region errors.")