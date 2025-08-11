#!/usr/bin/env python
"""Calculate the actual GPU memory allocation issue."""

# Constants from the code
MAX_COMPONENTS = 32  # From GPU code
MAX_PHASES = 64
MAX_STATEVARS = 8
MAX_FIXED_MOLE_FRACTION_CONDITIONS = 32
MAX_DOF_PER_PHASE = 8
MAX_INTERNAL_CONSTRAINTS = 8

# Calculate dimensions
MAX_SVD_DIM = MAX_COMPONENTS + MAX_PHASES + MAX_STATEVARS + MAX_FIXED_MOLE_FRACTION_CONDITIONS + 2
MAX_PHASE_MATRIX_DIM = MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS
MAX_DOF_SIZE = MAX_STATEVARS + MAX_DOF_PER_PHASE
MAX_EQ_MATRIX_ROWS = 2 * MAX_PHASES + MAX_COMPONENTS + 1
MAX_EQ_MATRIX_COLS = MAX_COMPONENTS + MAX_PHASES + MAX_STATEVARS
MAX_EQ_MATRIX_SIZE = MAX_EQ_MATRIX_ROWS * MAX_EQ_MATRIX_COLS
MAX_EQ_SOLN_LEN = MAX_EQ_MATRIX_COLS

print("GPU Memory Allocation Analysis")
print("=" * 60)

print(f"\nDimension calculations:")
print(f"  MAX_SVD_DIM: {MAX_SVD_DIM}")
print(f"  MAX_PHASE_MATRIX_DIM: {MAX_PHASE_MATRIX_DIM}")
print(f"  MAX_DOF_SIZE: {MAX_DOF_SIZE}")
print(f"  MAX_EQ_MATRIX_ROWS: {MAX_EQ_MATRIX_ROWS}")
print(f"  MAX_EQ_MATRIX_COLS: {MAX_EQ_MATRIX_COLS}")
print(f"  MAX_EQ_MATRIX_SIZE: {MAX_EQ_MATRIX_SIZE}")

# Per-thread allocations from gpu_equilibrium.py
per_thread_arrays = {
    'A_lstsq_copy': MAX_SVD_DIM * MAX_SVD_DIM * 8,
    'U_lstsq': MAX_SVD_DIM * MAX_SVD_DIM * 8,
    'V_lstsq': MAX_SVD_DIM * MAX_SVD_DIM * 8,
    'singular_values_lstsq': MAX_SVD_DIM * 8,
    'superdiag_lstsq': MAX_SVD_DIM * 8,
    'U_inv': MAX_PHASE_MATRIX_DIM * MAX_PHASE_MATRIX_DIM * 8,
    'V_inv': MAX_PHASE_MATRIX_DIM * MAX_PHASE_MATRIX_DIM * 8,
    'singular_values_inv': MAX_PHASE_MATRIX_DIM * 8,
    'superdiag_inv': MAX_PHASE_MATRIX_DIM * 8,
    'work_inv': MAX_PHASE_MATRIX_DIM * MAX_PHASE_MATRIX_DIM * 8,
    'x_dof': MAX_DOF_SIZE * 8,
    'grad': MAX_DOF_SIZE * 8,
    'hess': MAX_DOF_SIZE * MAX_DOF_SIZE * 8,
    'masses': MAX_COMPONENTS * 8,
    'mass_jac': MAX_COMPONENTS * MAX_DOF_SIZE * 8,
    'phase_matrix': MAX_PHASE_MATRIX_DIM * MAX_PHASE_MATRIX_DIM * 8,
    'equilibrium_matrix': MAX_EQ_MATRIX_SIZE * 8,
    'equilibrium_rhs': MAX_EQ_MATRIX_ROWS * 8,
    'eq_soln': MAX_EQ_SOLN_LEN * 8,
}

total_per_thread = sum(per_thread_arrays.values())

print(f"\nPer-thread memory allocations:")
for name, size in sorted(per_thread_arrays.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name:30s}: {size:10,} bytes ({size/1024:.1f} KB)")

print(f"\nTotal per thread: {total_per_thread:,} bytes ({total_per_thread/1024:.1f} KB, {total_per_thread/1024/1024:.2f} MB)")

# Calculate for different numbers of conditions
print(f"\nMemory usage for different numbers of conditions:")
print(f"(assuming 256 threads per block)")

threads_per_block = 256

for n_conditions in [100, 500, 1000, 2000, 5000, 10000, 20000]:
    blocks_needed = (n_conditions + threads_per_block - 1) // threads_per_block
    total_threads = blocks_needed * threads_per_block
    wasted_threads = total_threads - n_conditions
    
    total_memory_mb = (total_threads * total_per_thread) / (1024**2)
    wasted_memory_mb = (wasted_threads * total_per_thread) / (1024**2)
    
    print(f"\n  {n_conditions:,} conditions:")
    print(f"    Blocks: {blocks_needed}, Total threads: {total_threads:,}")
    print(f"    Memory: {total_memory_mb:.1f} MB")
    print(f"    Wasted on padding: {wasted_memory_mb:.1f} MB ({wasted_memory_mb/total_memory_mb*100:.1f}%)")

print(f"\n" + "=" * 60)
print("THE PROBLEM:")
print("=" * 60)
print(f"1. Each thread needs {total_per_thread/1024:.1f} KB of workspace")
print(f"2. The code allocates for ALL threads in the grid, not just active ones")
print(f"3. For large SVD dimensions ({MAX_SVD_DIM}x{MAX_SVD_DIM}), this explodes quickly")
print(f"4. The equilibrium matrix alone is {MAX_EQ_MATRIX_SIZE*8/1024:.1f} KB per thread!")

print(f"\nFor 10,000 conditions on an 8GB GPU:")
blocks = (10000 + threads_per_block - 1) // threads_per_block
total_threads = blocks * threads_per_block
memory_gb = (total_threads * total_per_thread) / (1024**3)
print(f"  Would need {memory_gb:.1f} GB just for work arrays!")
print(f"  This exceeds available GPU memory!")