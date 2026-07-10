# Calculate actual array sizes being allocated

# From the code
MAX_PHASES = 64
MAX_COMPONENTS = 32
MAX_STATEVARS = 8
MAX_DOF_PER_PHASE = 64
MAX_SVD_DIM = 18  # Approximate
MAX_PHASE_MATRIX_DIM = 128
threads = 256

# Calculate sizes
arrays = {
    "A_lstsq_copy": MAX_SVD_DIM * MAX_SVD_DIM,
    "U_lstsq": MAX_SVD_DIM * MAX_SVD_DIM,
    "V_lstsq": MAX_SVD_DIM * MAX_SVD_DIM,
    "hess (per CompsetState)": MAX_PHASE_MATRIX_DIM * MAX_PHASE_MATRIX_DIM,
    "system_states": 50000,
    "delta_ms": MAX_PHASES * MAX_COMPONENTS,
    "phase_compositions": MAX_PHASES * MAX_COMPONENTS,
}

print("Array sizes per thread (doubles):")
for name, size in arrays.items():
    size_bytes = size * 8
    print(f"  {name:30s}: {size:8d} doubles = {size_bytes:10d} bytes = {size_bytes/1024:.1f} KB")

print(f"\nTotal for {threads} threads:")
total = 0
for name, size in arrays.items():
    total_size = size * threads * 8
    total += total_size
    print(f"  {name:30s}: {total_size/(1024*1024):.1f} MB")

print(f"\nTOTAL ALLOCATION: {total/(1024*1024):.1f} MB")

# Check specific problematic calculation
print("\n" + "="*60)
print("POTENTIAL ISSUE: Address calculation for thread 255:")
system_state_size = 50000
offset_for_thread_255 = 255 * system_state_size * 8  # in bytes
print(f"  &arrays[19][255 * 50000] = &arrays[19][{255 * system_state_size}]")
print(f"  Byte offset: {offset_for_thread_255:,} bytes = {offset_for_thread_255/(1024*1024):.1f} MB")

if offset_for_thread_255 > 2**32:
    print("  ⚠️ WARNING: Offset exceeds 32-bit addressing limit!")
