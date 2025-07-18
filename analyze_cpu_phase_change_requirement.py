#!/usr/bin/env python
"""Analyze CPU's phase change iteration requirement."""

print("CPU PHASE CHANGE ITERATION REQUIREMENT")
print("=" * 60)

print("\nKEY FINDING:")
print("CPU requires: iterations_since_last_phase_change >= 5")
print("This means CPU must wait 5 iterations after any phase change before converging!")

print("\n" + "="*60)
print("ITERATION SEQUENCE:")

print("\nIteration 1:")
print("- Consolidation happens (phases_changed = True)")
print("- iterations_since_last_phase_change = 0 (reset)")
print("- Cannot converge yet")

print("\nIteration 2:")
print("- No phase changes")
print("- iterations_since_last_phase_change = 1")
print("- Cannot converge (1 < 5)")

print("\nIteration 3:")
print("- No phase changes")
print("- iterations_since_last_phase_change = 2")
print("- Cannot converge (2 < 5)")

print("\nIterations 4-6:")
print("- Would need to continue until iterations_since_last_phase_change >= 5")

print("\n" + "="*60)
print("BUT WAIT!")

print("\nThe CPU converges at iteration 3 with X(TI) = 0.900000")
print("This suggests the convergence happens differently...")

print("\nPossible explanation:")
print("- The CPU might not mark itself as 'converged' immediately")
print("- But it still updates site fractions to satisfy constraints")
print("- The '5 iteration' requirement might be for final convergence check")

print("\n" + "="*60)
print("GPU BEHAVIOR:")

print("\nThe GPU likely has a different requirement:")
print("- May allow updates immediately after consolidation")
print("- May have different iterations_since_last_phase_change threshold")
print("- This allows it to update at iteration 1, but in wrong direction")