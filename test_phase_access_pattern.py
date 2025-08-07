#!/usr/bin/env python
"""Test if phase record access pattern causes the issue."""

import numpy as np

# Phase indices in failing conditions
# FCC_A1 = 2, AU2BI_C15 = 0

# The pattern seems to be:
# - Thread ID % 7 = 3
# - Phase combination: FCC_A1 (idx 2) + AU2BI_C15 (idx 0)

print("Analysis of the pattern:")
print("="*60)

# Failing threads
failing_threads = [10, 17]

for tid in failing_threads:
    print(f"\nThread {tid}:")
    print(f"  tid % 7 = {tid % 7}")
    print(f"  Memory offset = {tid * 520} bytes")
    print(f"  Memory offset % 7 = {(tid * 520) % 7}")
    
    # If we access phase records 0 and 2
    # And the access pattern involves tid somehow
    phase0_access = (tid + 0) % 7
    phase2_access = (tid + 2) % 7
    
    print(f"  (tid + phase_idx) patterns:")
    print(f"    tid + 0 (AU2BI_C15) = {tid + 0}, % 7 = {phase0_access}")
    print(f"    tid + 2 (FCC_A1) = {tid + 2}, % 7 = {phase2_access}")
    
    # Check if there's something special about these values
    if phase0_access == 3 or phase2_access == 5:
        print(f"  ** Special pattern detected!")

# Another theory: stride interaction
print("\n\nStride interaction theory:")
print("="*60)

# With 6 phases, stride is 65 doubles
# Thread access pattern with stride
for tid in range(32):
    memory_offset = tid * 65  # in doubles
    
    # When accessing phase records, maybe there's an issue
    # with how memory is accessed
    
    if tid % 7 == 3:
        print(f"Thread {tid}: offset={memory_offset} doubles, pattern match!")
        
        # Check if this creates bank conflicts or other issues
        # GPU memory is often organized in banks
        bank = memory_offset % 32  # Common bank size
        print(f"  Memory bank: {bank}")

# The magic number 7 might come from somewhere in the calculation
print("\n\nLooking for source of 7:")
print("="*60)
print("Number of phases: 6")
print("Number of components: 3 (AU, BI, VA)")
print("Statevars: P, T = 2")
print("6 + 1 = 7  <-- Could this be it?")

# Or it could be related to phase record structure
print("\nPhase record function pointers: ~9-10")
print("Phase record integers: ~6-7")
print("Total fields in PhaseRecord: could be around 15-16")