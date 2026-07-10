# Check if arrays are allocated correctly for the access patterns

# From gpu_equilibrium.py
MAX_COMPONENTS = 4
MAX_PHASES = 21
MAX_DOF_PER_PHASE = 16
MAX_STATEVARS = 2

# Calculate results array size
results_per_condition = (7 + MAX_COMPONENTS + MAX_PHASES + 
                        (MAX_PHASES * MAX_DOF_PER_PHASE) + 
                        (MAX_PHASES * MAX_COMPONENTS) + MAX_PHASES)

print(f"Results per condition: {results_per_condition} doubles")

# Test with different condition counts
for num_conditions in [1, 2, 10]:
    print(f"\n{num_conditions} condition(s):")
    
    # Python allocates (from gpu_equilibrium.py line 2172):
    results_flat_size = num_conditions * results_per_condition
    print(f"  Python allocates: {results_flat_size} doubles")
    
    # Kernel threads launched
    threads_per_block = 256
    blocks = (num_conditions + threads_per_block - 1) // threads_per_block
    total_threads = blocks * threads_per_block
    print(f"  Threads launched: {total_threads} (blocks={blocks})")
    
    # Last valid access by a working thread
    last_valid_thread = num_conditions - 1
    last_valid_offset = last_valid_thread * results_per_condition + (results_per_condition - 1)
    print(f"  Last valid access: thread {last_valid_thread} at offset {last_valid_offset}")
    
    # Check if this is within bounds
    if last_valid_offset < results_flat_size:
        print(f"  ✓ Within bounds ({last_valid_offset} < {results_flat_size})")
    else:
        print(f"  ✗ OUT OF BOUNDS ({last_valid_offset} >= {results_flat_size})")

# Check initial phase data access
print("\n" + "="*60)
print("Initial phase data access:")

# From gpu_equilibrium.py, initial_phase_data_stride calculation
for num_conditions in [1, 2, 10]:
    # Each condition has phases data
    initial_data_per_condition = MAX_PHASES + MAX_PHASES + (MAX_PHASES * MAX_DOF_PER_PHASE) + (MAX_PHASES * MAX_COMPONENTS) + MAX_COMPONENTS + 1
    
    print(f"\n{num_conditions} condition(s):")
    print(f"  Data per condition: {initial_data_per_condition} doubles")
    
    # Total allocated
    total_allocated = num_conditions * initial_data_per_condition
    print(f"  Total allocated: {total_allocated} doubles")
    
    # Access by thread 1 (condition_idx=1)
    if num_conditions >= 2:
        thread1_offset = 1 * initial_data_per_condition
        print(f"  Thread 1 accesses offset: {thread1_offset}")
        if thread1_offset < total_allocated:
            print(f"  ✓ Within bounds")
        else:
            print(f"  ✗ OUT OF BOUNDS!")
