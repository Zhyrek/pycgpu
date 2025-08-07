#!/usr/bin/env python
"""Calculate optimal padding for InitialPhaseDataSingle struct."""

# Current size: 65 doubles (520 bytes) - problematic
current_size = 65

# Calculate padding options
print("Padding options for InitialPhaseDataSingle:")
print("-" * 50)

# Good alignment targets
targets = [
    (64, "512 bytes - power of 2, 4 cache lines"),
    (72, "576 bytes - multiple of 8 doubles"),
    (80, "640 bytes - 5 cache lines exactly"),
    (96, "768 bytes - 6 cache lines exactly"),
    (128, "1024 bytes - 8 cache lines, power of 2"),
]

for target, desc in targets:
    padding = target - current_size
    if padding >= 0:
        print(f"Pad to {target} doubles: add {padding} doubles padding")
        print(f"  -> {desc}")
        print(f"  -> Stride between threads: {target * 8} bytes")
        
        # Check specific threads
        for tid in [10, 17]:
            base_bytes = tid * target * 8
            chem_pot_offset = base_bytes + 60 * 8
            cache_line = chem_pot_offset // 128
            offset_in_line = chem_pot_offset % 128
            print(f"  -> Thread {tid} chem pots: cache line {cache_line}, offset {offset_in_line}")
        print()

print("\nRecommended: Pad to 80 doubles (640 bytes, 5 cache lines)")
print("This ensures clean cache line alignment for all threads.")