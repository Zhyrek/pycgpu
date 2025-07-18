#!/usr/bin/env python
"""Trace what CPU does at iteration 1 regarding site fraction updates."""

print("TRACING CPU ITERATION 1 BEHAVIOR")
print("=" * 60)

print("\nKEY FINDING:")
print("- At start of iteration 1: Both CPU and GPU have X(TI) = 0.903147")
print("- At start of iteration 2: CPU has X(TI) = 0.903147, GPU has X(TI) = 0.902960")

print("\nThis means during iteration 1:")
print("- GPU updates site fractions by ~0.0002")
print("- CPU does NOT update site fractions (or updates by 0)")

print("\n" + "="*60)
print("POSSIBLE REASONS CPU DOESN'T UPDATE:")

print("\n1. CPU's delta_y = 0 (no calculated change)")
print("   - Different c_G values?")
print("   - Different chemical potentials?")

print("\n2. CPU skips the update")
print("   - Step size = 0?")
print("   - Update logic differs?")

print("\n3. CPU updates but then reverts")
print("   - Bounds checking?")
print("   - Constraint violation?")

print("\n" + "="*60)
print("FROM GPU DEBUG OUTPUT:")
print("GPU delta_y at iteration 1 (implied from change):")
print("  delta_y[TI] = 0.902960 - 0.903147 = -0.000187")
print("  This is a small but non-zero update")

print("\n" + "="*60)
print("NEXT STEPS:")
print("1. Add debug output to CPU to show delta_y at iteration 1")
print("2. Check if CPU's c_G values are different at iteration 1")
print("3. Verify CPU's step size calculation at iteration 1")