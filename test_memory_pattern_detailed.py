#!/usr/bin/env python
"""Detailed analysis of memory access patterns for failing threads."""

import numpy as np

# Constants
stride = 65  # doubles per condition
CACHE_LINE_SIZE = 128  # bytes
DOUBLE_SIZE = 8

print("Detailed memory pattern analysis:")
print("=" * 80)

# Analyze specific patterns
for tid in [9, 10, 11, 16, 17, 18]:
    base_doubles = tid * stride
    base_bytes = base_doubles * DOUBLE_SIZE
    
    print(f"\nThread {tid}:")
    print(f"  Base address: {base_bytes} bytes ({base_doubles} doubles)")
    
    # Check different fields
    fields = [
        ("phase_indices", 0, 6),
        ("phase_amounts", 6, 6),
        ("site_fractions", 12, 24),
        ("compositions", 36, 24),
        ("chemical_potentials", 60, 4),
        ("num_phases", 64, 1)
    ]
    
    for field_name, field_offset, field_size in fields:
        start_doubles = base_doubles + field_offset
        end_doubles = start_doubles + field_size - 1
        start_bytes = start_doubles * DOUBLE_SIZE
        end_bytes = (end_doubles * DOUBLE_SIZE) + 7
        
        start_cache_line = start_bytes // CACHE_LINE_SIZE
        end_cache_line = end_bytes // CACHE_LINE_SIZE
        
        crosses = "CROSSES!" if start_cache_line != end_cache_line else ""
        
        print(f"    {field_name:20s}: bytes {start_bytes:5d}-{end_bytes:5d}, cache lines {start_cache_line}-{end_cache_line} {crosses}")

# Check pattern with 7
print("\n\nPattern analysis with modulo 7:")
print("-" * 40)
for tid in range(32):
    mod7 = tid % 7
    fails = "**FAILS**" if tid in [10, 17] else ""
    print(f"Thread {tid:2d} % 7 = {mod7} {fails}")

# Both 10 and 17 have remainder 3 when divided by 7
print("\nThreads with remainder 3 when divided by 7:", [t for t in range(32) if t % 7 == 3])

# Check if there's a pattern with specific memory addresses
print("\n\nMemory address patterns:")
print("-" * 60)

# Check if certain bit patterns in addresses correlate with failures
for tid in [10, 17]:
    base_bytes = tid * stride * DOUBLE_SIZE
    chem_pot_bytes = base_bytes + 60 * DOUBLE_SIZE
    
    print(f"\nThread {tid} chemical potentials address: {chem_pot_bytes} (0x{chem_pot_bytes:x})")
    print(f"  Binary: {bin(chem_pot_bytes)}")
    
    # Check various bit masks
    for shift in [3, 4, 5, 6, 7, 8, 9, 10]:
        mask = (1 << shift) - 1
        masked = chem_pot_bytes & mask
        print(f"  Bits [0:{shift-1}]: {masked} (0x{masked:x})")

# Check warp-level memory coalescing
print("\n\nWarp memory coalescing analysis:")
print("-" * 60)
print("For optimal coalescing, consecutive threads should access consecutive memory")

# Check access pattern for chemical potentials
print("\nChemical potential access pattern (all threads in warp 0):")
for tid in range(16, 20):  # Around the failing threads
    chem_pot_addr = (tid * stride + 60) * DOUBLE_SIZE
    print(f"Thread {tid}: {chem_pot_addr} bytes", end="")
    if tid > 0:
        prev_addr = ((tid-1) * stride + 60) * DOUBLE_SIZE
        gap = chem_pot_addr - prev_addr
        print(f" (gap: {gap} bytes)", end="")
    if tid in [10, 17]:
        print(" **FAILS**", end="")
    print()