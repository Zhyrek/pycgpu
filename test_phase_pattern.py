#!/usr/bin/env python
"""Test the pattern with phase indices."""

import numpy as np
from pycalphad import Database
from pycalphad.core.utils import filter_phases

# Load database
dbf = Database('important_tests/AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
phases = filter_phases(dbf, comps)

print("Phase list:")
for i, phase in enumerate(phases):
    print(f"  {i}: {phase}")

# Check indices
fcc_idx = phases.index('FCC_A1')
au2bi_idx = phases.index('AU2BI_C15')

print(f"\nFCC_A1 index: {fcc_idx}")
print(f"AU2BI_C15 index: {au2bi_idx}")

# The pattern: threads fail when thread_id % 7 = 3
# Threads 10 and 17 both have this property
print(f"\nThread pattern:")
print(f"Thread 10 % 7 = {10 % 7}")
print(f"Thread 17 % 7 = {17 % 7}")

# Check if phase indices relate to the pattern
print(f"\nPhase index patterns:")
print(f"FCC_A1 ({fcc_idx}) % 7 = {fcc_idx % 7}")
print(f"AU2BI_C15 ({au2bi_idx}) % 7 = {au2bi_idx % 7}")

# The stride in memory is 65 doubles (520 bytes) when we have 6 phases
# 65 = 13 * 5
# 520 = 65 * 8
print(f"\nMemory stride analysis:")
print(f"Stride = 65 doubles = 520 bytes")
print(f"65 % 7 = {65 % 7}")  # This is 2
print(f"520 % 7 = {520 % 7}")  # This is 2

# Thread memory offsets
for tid in [9, 10, 11, 16, 17, 18]:
    offset = tid * 520
    print(f"Thread {tid}: offset={offset} bytes, offset%7={offset%7}, tid%7={tid%7}")