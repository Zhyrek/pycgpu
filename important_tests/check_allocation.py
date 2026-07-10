# Check how memory is allocated for small condition counts
num_conditions = 1
threads_per_block = 256
blocks_per_grid = (num_conditions + threads_per_block - 1) // threads_per_block
total_threads = blocks_per_grid * threads_per_block

print(f"Conditions: {num_conditions}")
print(f"Threads per block: {threads_per_block}")
print(f"Blocks: {blocks_per_grid}")
print(f"Total threads allocated for: {total_threads}")
print(f"Memory allocated for {total_threads} threads even though only {num_conditions} needed")

# Test different sizes
for n in [1, 10, 100, 255, 256, 257]:
    blocks = (n + threads_per_block - 1) // threads_per_block
    total = blocks * threads_per_block
    print(f"n={n:3d}: {blocks} blocks, {total} threads allocated")
