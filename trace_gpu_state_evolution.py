#!/usr/bin/env python
"""Trace GPU state evolution to find where X(TI) changes from 0.903147 to 0.902960."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("TRACING GPU STATE EVOLUTION")
print("=" * 60)

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("\nRunning GPU with detailed output to trace state changes...")
print("\nLooking for:")
print("1. State after consolidation: X(TI) = 0.903147")
print("2. When/how it changes to: X(TI) = 0.902960")
print("3. What operation causes this change")

try:
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    print("\nGPU completed")
except Exception as e:
    print(f"GPU error: {e}")

print("\n" + "="*60)
print("KEY EVENTS TO TRACE:")
print("\n1. End of iteration 1:")
print("   - Consolidation happens")
print("   - Single phase with X(TI) = 0.903147")

print("\n2. Between iterations 1 and 2:")
print("   - Some update must happen")
print("   - Changes X(TI) from 0.903147 to 0.902960")

print("\n3. Start of iteration 2:")
print("   - GPU shows X(TI) = 0.902960")
print("   - CPU still has X(TI) = 0.903147")

print("\n" + "="*60)
print("POSSIBLE CAUSES:")
print("1. GPU applies site fraction updates at end of iteration 1")
print("2. GPU has different iteration flow than CPU")
print("3. GPU updates state differently after consolidation")