#!/usr/bin/env python
"""Analyze what happens during the solve at iteration 1 that causes divergence."""

print("ANALYZING ITERATION 1 SOLVE")
print("=" * 60)

print("\nSETUP at start of iteration 1:")
print("- Both CPU and GPU have single phase after consolidation")
print("- Both have X(TI) = 0.903147")
print("- Both need to satisfy constraint X(TI) = 0.9")

print("\n" + "="*60)
print("WHAT SHOULD HAPPEN:")

print("\n1. Construct 3x3 equilibrium matrix:")
print("   Row 0: Phase gradient equation")
print("   Row 1: Mass balance constraint (X(TI) = 0.9)")
print("   Row 2: System amount constraint")

print("\n2. Calculate RHS:")
print("   Row 1 RHS = some function of (target - current)")
print("   Row 1 RHS = f(0.9 - 0.903147) = f(-0.003147)")

print("\n3. Solve system to get updates:")
print("   delta_mu (chemical potentials)")
print("   delta_NP (phase amount)")

print("\n4. Apply updates:")
print("   Update chemical potentials")
print("   Update phase amounts")
print("   Update site fractions based on new chemical potentials")

print("\n" + "="*60)
print("WHERE DIVERGENCE OCCURS:")

print("\nFrom debug output:")
print("- CPU at iteration 2: X(TI) = 0.903147 (unchanged)")
print("- GPU at iteration 2: X(TI) = 0.902960 (changed!)")

print("\nThis means during iteration 1 solve:")
print("- CPU: Does NOT update site fractions (or updates them to same value)")
print("- GPU: Updates site fractions from 0.903147 to 0.902960")

print("\n" + "="*60)
print("HYPOTHESIS:")

print("\nThe GPU is applying site fraction updates differently than CPU:")
print("1. GPU calculates delta_y and applies it")
print("2. CPU either doesn't calculate delta_y or applies it differently")
print("3. The ~0.0002 change (0.903147 → 0.902960) suggests a small update")

print("\nNeed to compare:")
print("1. How CPU and GPU calculate site fraction updates after consolidation")
print("2. Whether CPU even updates site fractions at iteration 1")
print("3. The c_G values and chemical potentials used for the update")