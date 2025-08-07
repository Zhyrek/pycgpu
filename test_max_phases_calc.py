#!/usr/bin/env python
"""Debug MAX_PHASES calculation."""

# Simulate the calculation
for actual_phases in range(1, 10):
    padding_factor = 1.2
    safety_minimum = 4
    
    # Current calculation
    calc1 = int(actual_phases * padding_factor)
    max_phases = max(safety_minimum, calc1)
    
    print(f"actual_phases={actual_phases}: int({actual_phases} * 1.2) = {calc1}, max(4, {calc1}) = {max_phases}")
    
    # Check if it's sufficient
    if max_phases < actual_phases:
        print(f"  ✗ PROBLEM: MAX_PHASES ({max_phases}) < actual_phases ({actual_phases})")
    else:
        padding = max_phases - actual_phases
        print(f"  ✓ OK: Padding = {padding} phases")