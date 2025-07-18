#!/usr/bin/env python
"""Test to examine gradient values for single phase after consolidation."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("=== Gradient Analysis for Single Phase ===")
print("\nAfter consolidation to single BCC_A2 phase:")
print("Current: X(TI) = 0.9029604")
print("Target: X(TI) = 0.9000000")
print("Error: 0.0029604")

print("\nFor BCC_A2 phase, X(TI) = Y(TI) (direct mapping)")
print("So we need Y(TI) to change from 0.9029604 to 0.9000000")
print("Required delta_Y(TI) = -0.0029604")

print("\nThe equilibrium system should produce:")
print("1. Delta chemical potentials that drive this change")
print("2. Delta phase amount (likely small)")
print("3. Delta site fractions including delta_Y(TI) ≈ -0.003")

print("\nIf GPU produces delta_Y(TI) ≈ -1e-7 instead of -0.003,")
print("that's a 4 order of magnitude error!")

print("\nPossible root causes:")
print("1. Gradient values are incorrect")
print("2. c_G calculation is wrong")
print("3. c_component matrix is incorrect")
print("4. Equilibrium matrix becomes singular/ill-conditioned")
print("5. Solution vector is incorrectly scaled")

# Run calculations to see what happens
print("\n--- Running GPU test ---")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()[0]
print(f"GPU X(TI) = {gpu_x_ti:.10f}")

print("\n--- Running CPU test ---")
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()[0]
print(f"CPU X(TI) = {cpu_x_ti:.10f}")

if abs(gpu_x_ti - 0.9) > 1e-6:
    print(f"\n❌ GPU fails with error: {gpu_x_ti - 0.9:.10f}")
else:
    print("\n✓ GPU converges correctly")