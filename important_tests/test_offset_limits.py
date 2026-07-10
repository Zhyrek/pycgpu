# Check if we're hitting offset limitations

# SystemState is the largest array
SYSTEM_STATE_SIZE = 50000  # doubles
bytes_per_double = 8

# Check different thread counts
for num_conditions in [1, 2, 10, 100, 256]:
    threads_allocated = 256 if num_conditions <= 256 else ((num_conditions + 255) // 256) * 256
    
    print(f"\nConditions: {num_conditions}, Threads allocated: {threads_allocated}")
    
    # Last valid thread
    last_valid = num_conditions - 1
    offset_bytes = last_valid * SYSTEM_STATE_SIZE * bytes_per_double
    print(f"  Last valid thread ({last_valid}): offset = {offset_bytes:,} bytes")
    
    # Last allocated thread  
    last_allocated = threads_allocated - 1
    offset_bytes = last_allocated * SYSTEM_STATE_SIZE * bytes_per_double
    print(f"  Last allocated thread ({last_allocated}): offset = {offset_bytes:,} bytes")
    
    if offset_bytes > 2**30:  # 1GB
        print(f"  ⚠️ WARNING: Offset > 1GB - may hit addressing limits on some AMD GPUs")
    if offset_bytes > 2**31:  # 2GB (signed 32-bit limit)
        print(f"  ❌ ERROR: Offset > 2GB - exceeds signed 32-bit addressing!")
    if offset_bytes > 2**32:  # 4GB
        print(f"  ❌ CRITICAL: Offset > 4GB - exceeds 32-bit addressing!")
