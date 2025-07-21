#!/usr/bin/env python3
"""
Fix for GPU system amount constraint handling after phase consolidation.

The issue: GPU writes system amount constraint inside phase loops, meaning it gets
written differently when phases are consolidated (2 phases -> 1 phase changes the
matrix structure).

The fix: Write system amount constraint ONCE, outside phase loops, with all phases
contributing to the same row (matching CPU behavior).
"""

# The fix needs to be applied to minimizer.h
# Here's what needs to change:

print("""
GPU System Amount Constraint Fix
================================

The problem:
- GPU writes system amount rows INSIDE phase loops (lines 1797-1813, 1855-1871)
- This means with 2 phases, it writes 2 system amount rows
- When consolidated to 1 phase, it writes 1 system amount row
- This changes the matrix structure and causes incorrect results

The solution:
- Move system amount constraint OUTSIDE phase loops
- Write it ONCE at the correct row index
- Have ALL phases contribute to this single row

Specific changes needed in minimizer.h:

1. Remove lines 1797-1813 (system amount inside free phase loop)
2. Remove lines 1855-1871 (system amount inside fixed phase loop)
3. Add a new section AFTER all phase loops that writes the system amount constraint once

The new code structure should be:
- Loop over free phases (write phase rows and mole fraction constraints)
- Loop over fixed phases (write phase rows and mole fraction constraints)
- THEN write system amount constraint ONCE with all phases contributing

This matches the CPU behavior where system_amount_index is calculated once
and all phases contribute to that single row.
""")