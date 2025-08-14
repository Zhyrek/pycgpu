#!/usr/bin/env python
"""Test memory offset calculations for multi-condition batches."""

import numpy as np

# Simulate the stride calculation for 3 components, 8 phases
MAX_COMPONENTS = 4  # Padded from 3
MAX_PHASES = 9      # Padded from 8
MAX_DOF_PER_PHASE = 6

# Calculate stride (doubles per condition)
doubles_per_struct = (MAX_PHASES + MAX_PHASES + 
                     (MAX_PHASES * MAX_DOF_PER_PHASE) + 
                     (MAX_PHASES * MAX_COMPONENTS) + 
                     MAX_COMPONENTS + 1)

print("Memory layout for initial phase data:")
print(f"  MAX_COMPONENTS: {MAX_COMPONENTS}")
print(f"  MAX_PHASES: {MAX_PHASES}")
print(f"  MAX_DOF_PER_PHASE: {MAX_DOF_PER_PHASE}")
print(f"  Stride: {doubles_per_struct} doubles = {doubles_per_struct * 8} bytes")

# Test accessing conditions at different thread indices
num_conditions = 67  # From the test that fails

print(f"\nMemory offsets for {num_conditions} conditions:")
print("Thread | Condition | Offset (doubles) | Offset (bytes) | Cache line")
print("-" * 70)

# Check first few and problematic threads
test_threads = [0, 1, 2, 10, 17, 24, 31, 38, 45, 52, 59, 66]

for tid in test_threads:
    if tid < num_conditions:
        condition_idx = tid  # Direct mapping
        struct_offset = condition_idx * doubles_per_struct
        byte_offset = struct_offset * 8
        cache_line = byte_offset // 64
        
        print(f"{tid:6} | {condition_idx:9} | {struct_offset:16} | {byte_offset:14} | {cache_line:10}")

# Check for potential memory overlap
print("\nChecking for potential memory overlaps:")
print("Distance between consecutive conditions: {} doubles = {} bytes".format(
    doubles_per_struct, doubles_per_struct * 8))

# Check if threads 10 and 17 have a pattern
print("\nPattern analysis for failing threads (10, 17, 24, 31, 38):")
for tid in [10, 17, 24, 31, 38]:
    print(f"  Thread {tid}: tid % 7 = {tid % 7}")

# Check memory alignment
print("\nMemory alignment analysis:")
cache_line_size = 64  # bytes
bytes_per_struct = doubles_per_struct * 8

if bytes_per_struct % cache_line_size == 0:
    print(f"✓ Stride is cache-line aligned ({bytes_per_struct} bytes = {bytes_per_struct // cache_line_size} cache lines)")
else:
    misalignment = bytes_per_struct % cache_line_size
    print(f"✗ Stride is NOT cache-line aligned (off by {misalignment} bytes)")
    print(f"  This can cause false sharing between threads!")
    
    # Calculate which conditions might overlap cache lines
    print("\n  Conditions that share cache lines:")
    for i in range(min(10, num_conditions - 1)):
        start_byte = i * bytes_per_struct
        end_byte = start_byte + bytes_per_struct - 1
        start_cache = start_byte // cache_line_size
        end_cache = end_byte // cache_line_size
        
        if start_cache != end_cache:
            next_start_byte = (i + 1) * bytes_per_struct
            next_cache = next_start_byte // cache_line_size
            if end_cache == next_cache:
                print(f"    Condition {i} ends in cache line {end_cache}, condition {i+1} starts there")