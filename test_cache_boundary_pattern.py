#!/usr/bin/env python
"""Check if cache line boundary crossing correlates with failures."""

# Memory layout
MAX_PHASES = 6
MAX_COMPONENTS = 4
MAX_DOF_PER_PHASE = 4
stride_doubles = 65  # per condition

# Chemical potentials start at offset 60 within each condition's data
chem_pot_offset_within_condition = 60
chem_pot_size = MAX_COMPONENTS  # 4 doubles

print("Cache line boundary analysis for all 32 threads:")
print("-" * 60)
print("Thread | Cache Line | Byte Offset | Chem Pot Range | Crosses?")
print("-" * 60)

threads_that_cross = []

for tid in range(32):
    # Calculate memory addresses
    base_offset_doubles = tid * stride_doubles
    chem_pot_start_doubles = base_offset_doubles + chem_pot_offset_within_condition
    chem_pot_end_doubles = chem_pot_start_doubles + chem_pot_size - 1
    
    # Convert to bytes
    chem_pot_start_bytes = chem_pot_start_doubles * 8
    chem_pot_end_bytes = (chem_pot_end_doubles * 8) + 7  # Last byte of last double
    
    # Calculate cache lines (128 bytes each)
    start_cache_line = chem_pot_start_bytes // 128
    end_cache_line = chem_pot_end_bytes // 128
    byte_offset = chem_pot_start_bytes % 128
    
    crosses = start_cache_line != end_cache_line
    if crosses:
        threads_that_cross.append(tid)
    
    status = "**FAILS**" if tid in [10, 17] else ""
    cross_mark = "YES" if crosses else "NO"
    
    print(f"{tid:6d} | {start_cache_line:10d} | {byte_offset:11d} | {chem_pot_start_bytes}-{chem_pot_end_bytes} | {cross_mark:8s} {status}")

print(f"\nThreads where chemical potentials cross cache line boundary: {threads_that_cross}")
print(f"Known failing threads: [10, 17]")

# Check if crossing correlates with failures
if 10 in threads_that_cross or 17 in threads_that_cross:
    print("\n*** Pattern found: At least one failing thread has cache line crossing! ***")

# Also check which threads have certain offset patterns
print(f"\nThreads with specific byte offsets within cache line:")
for offset in [72, 80, 88, 96, 104, 112, 120, 0, 8, 16]:
    threads_with_offset = []
    for tid in range(32):
        base_offset_doubles = tid * stride_doubles
        chem_pot_start_doubles = base_offset_doubles + chem_pot_offset_within_condition
        chem_pot_start_bytes = chem_pot_start_doubles * 8
        byte_offset = chem_pot_start_bytes % 128
        if byte_offset == offset:
            threads_with_offset.append(tid)
    if threads_with_offset:
        print(f"  Offset {offset:3d}: threads {threads_with_offset}")