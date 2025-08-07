#\!/usr/bin/env python
"""Analyze warp patterns for failing threads."""

print("Warp analysis:")
print("="*60)

# In CUDA, warps are groups of 32 threads
warp_size = 32

# Failing threads
failing_threads = [10, 17]

for tid in failing_threads:
    warp_id = tid // warp_size
    lane_id = tid % warp_size
    
    print(f"\\nThread {tid}:")
    print(f"  Warp ID: {warp_id}")
    print(f"  Lane ID: {lane_id}")
    print(f"  tid % 7 = {tid % 7}")
    
# All threads in our test
print("\\nAll threads (0-31) by warp:")
print("Warp 0 (threads 0-31):")
for tid in range(32):
    marker = " **FAILS**" if tid in failing_threads else ""
    mod7 = tid % 7
    print(f"  Thread {tid:2d}: lane={tid:2d}, tid%7={mod7}{marker}")

# Check if there is a pattern with lane IDs
print("\\nLane ID patterns:")
for tid in failing_threads:
    lane_id = tid % 32
    print(f"Thread {tid}: lane {lane_id}")
    print(f"  Binary: {bin(lane_id)}")
    print(f"  Lane % 7 = {lane_id % 7}")

# Memory bank conflicts happen when threads in the same warp
# access the same memory bank
print("\\nMemory bank analysis (32 banks):")
stride = 65  # doubles
for tid in range(32):
    offset = tid * stride
    bank = offset % 32
    conflict = " **BANK CONFLICT**" if bank in [10, 17] else ""
    fail = " **FAILS**" if tid in failing_threads else ""
    print(f"Thread {tid:2d}: offset={offset:4d}, bank={bank:2d}{conflict}{fail}")
