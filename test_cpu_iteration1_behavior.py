#!/usr/bin/env python
"""Test CPU behavior at iteration 1 to understand the delay."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("TESTING CPU ITERATION 1 BEHAVIOR")
print("=" * 60)

# I'll examine the CPU output more carefully
print("\nBased on previous analysis:")
print("- CPU consolidates at iteration 1")
print("- CPU keeps X(TI) = 0.903147 at iterations 1 and 2")
print("- CPU achieves X(TI) = 0.900000 at iteration 3")

print("\n" + "="*60)
print("UNDERSTANDING THE DELAY:")

print("\n1. EQUILIBRIUM SOLUTION AT ITERATION 1:")
print("   CPU solves 3x3 system but gets small/zero updates")
print("   This keeps site fractions unchanged")

print("\n2. EQUILIBRIUM SOLUTION AT ITERATION 2:")
print("   CPU still has X(TI) = 0.903147")
print("   Matrix shows same site fractions as iteration 1")
print("   Still no significant update")

print("\n3. EQUILIBRIUM SOLUTION AT ITERATION 3:")
print("   CPU finally updates to X(TI) = 0.900000")
print("   Something changes to allow the update")

print("\n" + "="*60)
print("KEY INSIGHT:")

print("\nThe CPU might be using a different algorithm after consolidation:")
print("- Gradual approach to constraint satisfaction")
print("- Requires multiple iterations to stabilize")
print("- Avoids large jumps in site fractions")

print("\nThe GPU might be more aggressive:")
print("- Immediately tries to correct constraint violation")
print("- Makes larger update at iteration 1")
print("- But update is in slightly wrong direction")

print("\n" + "="*60)
print("THE FUNDAMENTAL DIFFERENCE:")

print("\nCPU: Conservative approach after consolidation")
print("      Waits for system to stabilize before large updates")
print("      Eventually reaches exact constraint value")

print("\nGPU: Aggressive approach after consolidation")
print("      Immediately updates site fractions")
print("      Gets stuck at wrong value due to premature update")