# Calculate memory requirements for CompsetState arrays

# Typical values (from test cases)
MAX_PHASES = 21
MAX_COMPONENTS = 4  # AL, CU, FE, VA
MAX_PHASE_MATRIX_DIM = 128  # Typical upper bound
THREADS = 1024  # Typical batch size

# Sizes in doubles (8 bytes each)
hess_size_per_compset = MAX_PHASE_MATRIX_DIM * MAX_PHASE_MATRIX_DIM
mass_jac_size_per_compset = MAX_PHASES * MAX_COMPONENTS  
c_component_size_per_compset = MAX_COMPONENTS * MAX_COMPONENTS

# Total per CompsetState
total_per_compset = hess_size_per_compset + mass_jac_size_per_compset + c_component_size_per_compset
total_per_compset_kb = (total_per_compset * 8) / 1024

# Total for all CompsetStates per thread
total_per_thread = MAX_PHASES * total_per_compset
total_per_thread_kb = (total_per_thread * 8) / 1024
total_per_thread_mb = total_per_thread_kb / 1024

# Total for all threads
total_all_threads = THREADS * total_per_thread
total_all_threads_mb = (total_all_threads * 8) / (1024 * 1024)
total_all_threads_gb = total_all_threads_mb / 1024

print(f"Memory Requirements for CompsetState Arrays:")
print(f"============================================")
print(f"Per CompsetState:")
print(f"  hess: {hess_size_per_compset} doubles = {hess_size_per_compset * 8 / 1024:.1f} KB")
print(f"  mass_jac: {mass_jac_size_per_compset} doubles = {mass_jac_size_per_compset * 8 / 1024:.1f} KB")
print(f"  c_component: {c_component_size_per_compset} doubles = {c_component_size_per_compset * 8 / 1024:.1f} KB")
print(f"  Total: {total_per_compset_kb:.1f} KB")
print()
print(f"Per Thread (MAX_PHASES={MAX_PHASES} CompsetStates):")
print(f"  Total: {total_per_thread} doubles = {total_per_thread_mb:.1f} MB")
print()
print(f"For All Threads ({THREADS} threads):")
print(f"  Total: {total_all_threads} doubles = {total_all_threads_gb:.2f} GB")
print()
print("VERDICT: This is TOO MUCH memory!")
print()
print("Optimization Strategy:")
print("=====================")
print("Since CompsetStates are accessed sequentially (one at a time per phase),")
print("we can use a SMALLER pool and index with: thread_idx * MAX_PHASES + phase_idx")
print()
print("However, hess array alone is so large that even ONE per thread would be:")
print(f"  {THREADS} threads * {hess_size_per_compset * 8 / 1024:.1f} KB = {THREADS * hess_size_per_compset * 8 / (1024*1024):.1f} MB")
