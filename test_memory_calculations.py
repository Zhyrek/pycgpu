#!/usr/bin/env python
"""Calculate memory offsets and check for potential conflicts."""

# Constants from the GPU code
MAX_PHASES = 6
MAX_COMPONENTS = 4  # Including VA
MAX_STATEVARS = 4
MAX_DOF_PER_PHASE = 4
MAX_FIXED_MOLE_FRACTION_CONDITIONS = 4

# Calculate offsets in InitialPhaseDataSingle struct
print("InitialPhaseDataSingle memory layout (in doubles):")
print("-" * 50)

offset = 0
print(f"phase_indices[{MAX_PHASES}]:        offset {offset}-{offset+MAX_PHASES-1}")
offset += MAX_PHASES

print(f"phase_amounts[{MAX_PHASES}]:        offset {offset}-{offset+MAX_PHASES-1}")
offset += MAX_PHASES

site_fractions_size = MAX_PHASES * MAX_DOF_PER_PHASE
print(f"site_fractions[{site_fractions_size}]:      offset {offset}-{offset+site_fractions_size-1}")
offset += site_fractions_size

compositions_size = MAX_PHASES * MAX_COMPONENTS
print(f"compositions[{compositions_size}]:        offset {offset}-{offset+compositions_size-1}")
offset += compositions_size

print(f"chemical_potentials[{MAX_COMPONENTS}]: offset {offset}-{offset+MAX_COMPONENTS-1}")
offset += MAX_COMPONENTS

print(f"num_phases:                  offset {offset}")
offset += 1

print(f"\nTotal size per condition: {offset} doubles ({offset*8} bytes)")

# Check alignment
print(f"\nAlignment check:")
print(f"  Size is {'aligned' if offset % 8 == 0 else 'NOT aligned'} to 64 bytes")
print(f"  Size is {'aligned' if offset % 16 == 0 else 'NOT aligned'} to 128 bytes")

# Check specific thread access patterns
print(f"\nThread access patterns for 32 conditions:")
print("-" * 50)

threads_per_block = 256
blocks = 1  # All 32 threads in one block

# Failing threads are 10 and 17
for tid in [9, 10, 11, 16, 17, 18]:
    base_offset = tid * offset
    chem_pot_offset = base_offset + (MAX_PHASES + MAX_PHASES + site_fractions_size + compositions_size)
    
    print(f"Thread {tid}:")
    print(f"  Base offset: {base_offset} doubles")
    print(f"  Chemical potentials: {chem_pot_offset}-{chem_pot_offset+MAX_COMPONENTS-1}")
    print(f"  Distance from Thread 0: {base_offset} doubles ({base_offset*8} bytes)")
    
    # Check if this maps to a cache line boundary
    cache_line = (base_offset * 8) // 128
    cache_offset = (base_offset * 8) % 128
    print(f"  Cache line: {cache_line}, offset within line: {cache_offset} bytes")

# Calculate stride between threads
print(f"\nStride between threads: {offset} doubles ({offset*8} bytes)")
print(f"Stride in cache lines: {(offset*8)/128:.2f}")

# Check for bank conflicts (32 banks, 4-byte words)
print(f"\nBank conflict analysis (32 banks, 4-byte words):")
for tid in [10, 17]:
    base_addr = tid * offset * 8  # in bytes
    bank = (base_addr // 4) % 32
    print(f"Thread {tid}: Bank {bank}")