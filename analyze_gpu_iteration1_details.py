#!/usr/bin/env python
"""Analyze GPU iteration 1 details to understand the site fraction update."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("ANALYZING GPU ITERATION 1 SITE FRACTION UPDATE")
print("=" * 60)

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("\nRunning GPU calculation to capture iteration 1 details...")

try:
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    print("GPU completed")
except Exception as e:
    print(f"GPU error: {e}")

print("\n" + "="*60)
print("FROM GPU DEBUG OUTPUT:")

print("\nIteration 1 state:")
print("- Start: X(TI) = 0.903147 (after consolidation)")
print("- End: X(TI) = 0.902960 (shown at iteration 2)")
print("- Change: -0.000187")

print("\nThis corresponds to site fraction change:")
print("- Y(TI) changed from 0.903147 to 0.902960")
print("- delta_y[TI] ≈ -0.000187")

print("\n" + "="*60)
print("HYPOTHESIS ABOUT CPU vs GPU DIFFERENCE:")

print("\n1. Different equilibrium solution at iteration 1:")
print("   - Both solve 3x3 system")
print("   - GPU gets non-zero solution → applies site fraction update")
print("   - CPU gets zero solution → no site fraction update")

print("\n2. Different c_G values:")
print("   - GPU: c_G leads to non-zero delta_y")
print("   - CPU: c_G = 0 or very small → delta_y ≈ 0")

print("\n3. Different step size calculation:")
print("   - GPU: step_size = 1.0")
print("   - CPU: step_size = 0 or very small?")

print("\nThe key is that CPU keeps X(TI) = 0.903147 unchanged at iteration 1")
print("while GPU updates it to 0.902960.")