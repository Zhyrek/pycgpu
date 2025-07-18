#!/usr/bin/env python
"""Compare exact solver deltas between CPU and GPU at the critical iteration."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("COMPARING SOLVER DELTAS AT CRITICAL ITERATION")
print("=" * 60)

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("\nKnown facts:")
print("- After consolidation, both have single phase with X(TI) = 0.903147")
print("- Target: X(TI) = 0.900000")
print("- Residual = 0.903147 - 0.900000 = 0.003147")
print("\n- CPU: Successfully corrects to X(TI) = 0.900000")
print("- GPU: Only achieves X(TI) = 0.902960 (wrong)")

print("\n" + "="*60)
print("KEY ANALYSIS NEEDED:")
print("\n1. What is the equilibrium matrix after consolidation?")
print("2. What is the RHS vector after consolidation?")
print("3. What solution vector does each solver produce?")
print("4. How is the solution applied differently?")

print("\nFrom debug outputs, we know:")
print("- GPU RHS at iteration 1: ~0.00314 (the residual)")
print("- GPU solution norm is very small")
print("- GPU phase amount delta is tiny")
print("- Site fraction updates are tiny")

print("\nThe problem appears to be:")
print("1. GPU solver produces a tiny solution despite non-zero RHS")
print("2. This suggests either:")
print("   - Matrix is poorly conditioned")
print("   - Solver tolerance is too strict")
print("   - Solution vector ordering is wrong")

print("\nBased on previous debug output:")
print("- GPU singular values show very poor conditioning")
print("- Condition number ~1e16 after consolidation")
print("- This causes SVD solver to produce near-zero solution")