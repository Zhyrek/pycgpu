#!/usr/bin/env python
"""Trace the exact point where CPU and GPU diverge after consolidation."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("TRACING EXACT CPU/GPU DIVERGENCE POINT")
print("=" * 60)

print("\nFrom previous analysis:")
print("- Both CPU and GPU consolidate to single phase at iteration 1")
print("- Both have X(TI) = 0.903147 after consolidation")
print("- CPU then converges to X(TI) = 0.900000 (exact)")
print("- GPU converges to X(TI) = 0.902960 (wrong)")

print("\n" + "="*60)
print("KEY QUESTION: What happens in the NEXT iteration after consolidation?")

print("\nCPU behavior after consolidation:")
print("- Has single phase with X(TI) = 0.903147")
print("- Constructs 3x3 equilibrium matrix")
print("- Solves system to get updates")
print("- Updates site fractions to achieve X(TI) = 0.900000")

print("\nGPU behavior after consolidation:")
print("- Has single phase with X(TI) = 0.903147")
print("- Constructs 3x3 equilibrium matrix")
print("- Solves system to get updates")
print("- Updates site fractions but gets X(TI) = 0.902960")

print("\n" + "="*60)
print("POSSIBLE DIVERGENCE POINTS:")

print("\n1. Matrix construction differences:")
print("   - Different gradient calculations?")
print("   - Different c_G values?")
print("   - Different constraint row coefficients?")

print("\n2. RHS construction differences:")
print("   - Different residual calculations?")
print("   - Different target value handling?")

print("\n3. Linear solver differences:")
print("   - SVD vs LAPACK numerical differences?")
print("   - Different solution vectors?")

print("\n4. Update application differences:")
print("   - Different step size calculations?")
print("   - Different site fraction update formulas?")

print("\n" + "="*60)
print("NEXT STEPS:")
print("1. Compare equilibrium matrices at iteration 2 (after consolidation)")
print("2. Compare RHS vectors at iteration 2")
print("3. Compare solution vectors from linear solvers")
print("4. Compare how updates are applied to site fractions")

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("\nRunning calculations to capture iteration 2 details...")

# Run both to capture debug output
try:
    print("\nCPU calculation:")
    result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
    cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()[0]
    print(f"CPU final: X(TI) = {cpu_x_ti:.10f}")
except Exception as e:
    print(f"CPU error: {e}")

try:
    print("\nGPU calculation:")
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()[0]
    print(f"GPU final: X(TI) = {gpu_x_ti:.10f}")
except Exception as e:
    print(f"GPU error: {e}")