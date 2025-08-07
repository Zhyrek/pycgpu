#!/usr/bin/env python
"""Design and test a dynamic padding strategy for avoiding cache conflicts."""

import numpy as np

def calculate_safe_stride(base_size, max_threads=256):
    """
    Calculate a safe padded stride that avoids cache conflicts.
    
    The strategy:
    1. Avoid small prime factors (especially 2, 3, 5, 7) that create patterns
    2. Align to cache line boundaries (64 bytes = 8 doubles)
    3. Ensure no systematic conflicts for common thread counts
    
    Parameters:
    -----------
    base_size : int
        The actual size needed for the data structure (in doubles)
    max_threads : int
        Maximum number of threads that might run concurrently
        
    Returns:
    --------
    int
        Padded size that avoids cache conflicts
    """
    
    # Cache line is typically 64 bytes = 8 doubles
    CACHE_LINE_DOUBLES = 8
    
    # Start with cache-line aligned size
    padded = ((base_size + CACHE_LINE_DOUBLES - 1) // CACHE_LINE_DOUBLES) * CACHE_LINE_DOUBLES
    
    # List of problematic patterns to avoid
    # These are small primes and their products that create stride conflicts
    problematic_factors = [3, 5, 7, 11, 13]
    
    # Also avoid exact multiples of common warp/wavefront sizes
    gpu_sizes = [32, 64]  # NVIDIA warp=32, AMD wavefront=64
    
    def has_conflicts(size, max_threads):
        """Check if a size will cause conflicts for likely thread patterns."""
        conflicts = []
        
        # Check for conflicts with small primes
        for prime in problematic_factors:
            if size % prime == 0:
                # This creates a pattern where every prime-th thread conflicts
                conflicts.append(f"divisible by {prime}")
            
            # Also check if (size % prime) creates a bad pattern
            remainder = size % prime
            if remainder == prime - 1:
                # Almost divisible - can create patterns too
                conflicts.append(f"size % {prime} = {remainder} (near multiple)")
        
        # Check for exact GPU size multiples which can cause bank conflicts
        for gpu_size in gpu_sizes:
            if size % gpu_size == 0:
                conflicts.append(f"exact multiple of {gpu_size}")
        
        # Check specific problematic remainders that we've observed
        # For example, size=83 has 83%7=6, which causes 7-stride conflicts
        if size % 7 == 6:
            conflicts.append("size % 7 = 6 (creates 7-stride pattern)")
        
        return conflicts
    
    # Keep padding until we find a size without conflicts
    max_padding = base_size  # Don't more than double the size
    original_padded = padded
    
    while padded < base_size + max_padding:
        conflicts = has_conflicts(padded, max_threads)
        if not conflicts:
            return padded
        
        # Try next cache line
        padded += CACHE_LINE_DOUBLES
    
    # If we can't find a perfect size, use a prime number padding strategy
    # Prime numbers tend to avoid systematic conflicts
    # Find next prime after our base size
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    candidate = ((base_size + CACHE_LINE_DOUBLES - 1) // CACHE_LINE_DOUBLES) * CACHE_LINE_DOUBLES
    while not is_prime(candidate):
        candidate += CACHE_LINE_DOUBLES
    
    return candidate


def test_padding_strategy():
    """Test the padding strategy with various sizes."""
    
    print("Testing Dynamic Padding Strategy")
    print("=" * 70)
    
    # Test cases representing different system configurations
    test_cases = [
        # (components, phases, description)
        (2, 2, "Binary, 2 phases"),
        (3, 3, "Ternary, 3 phases"),
        (3, 6, "Ternary, 6 phases (AuBi case)"),
        (4, 8, "Quaternary, 8 phases"),
        (5, 10, "Quinary, 10 phases"),
    ]
    
    for n_components, n_phases, description in test_cases:
        # Calculate base size (simplified formula)
        # Based on the SystemSpec structure
        n_statevars = 3
        n_fixed_mole = n_components
        
        # Size calculation from gpu_systemspec_flat.py
        base_size = (
            50 +  # scalar fields
            n_components +  # initial_chemical_potentials
            n_fixed_mole * n_components +  # prescribed_mole_fraction_coefficients
            n_fixed_mole +  # prescribed_mole_fraction_rhs
            n_components +  # free_chemical_potential_indices
            n_statevars +  # free_statevar_indices
            n_components +  # fixed_chemical_potential_indices
            n_statevars +  # fixed_statevar_indices
            n_phases  # fixed_stable_compset_indices
        )
        
        padded_size = calculate_safe_stride(base_size)
        overhead = ((padded_size - base_size) / base_size) * 100
        
        print(f"\n{description}:")
        print(f"  Components: {n_components}, Phases: {n_phases}")
        print(f"  Base size: {base_size} doubles")
        print(f"  Padded size: {padded_size} doubles")
        print(f"  Overhead: {overhead:.1f}%")
        
        # Check for conflicts with common thread counts
        thread_counts = [32, 64, 128, 256]
        for n_threads in thread_counts:
            # Check if threads with certain patterns will conflict
            conflict_found = False
            for spacing in [3, 5, 7, 11]:
                # Check if threads spaced by 'spacing' will have cache conflicts
                if n_threads >= spacing * 2:
                    offset_diff = spacing * padded_size
                    # Check if this creates bad cache patterns
                    if offset_diff % 64 == 0:  # Same cache line offset
                        conflict_found = True
                        print(f"  ⚠️  WARNING: {n_threads} threads with spacing {spacing} may conflict")
                        break
            
            if not conflict_found:
                print(f"  ✓ No conflicts detected for {n_threads} threads")
    
    print("\n" + "=" * 70)
    print("Specific test for AuBi case (83 doubles):")
    base_83 = 83
    padded_83 = calculate_safe_stride(base_83)
    print(f"  Original size: {base_83} doubles")
    print(f"  Padded size: {padded_83} doubles")
    print(f"  This should avoid the 7-stride conflict pattern")
    
    # Verify it actually avoids the conflict
    print(f"\n  Verification for threads with (tid % 7 == 3):")
    for tid in [3, 10, 17, 24, 31]:
        offset = tid * padded_83
        print(f"    Thread {tid:2d}: offset {offset:5d} (% 64 = {offset % 64})")
    
    # Check if the offsets create conflicts
    offsets = [3 * padded_83, 10 * padded_83, 17 * padded_83]
    diff1 = offsets[1] - offsets[0]
    diff2 = offsets[2] - offsets[1]
    print(f"\n  Offset differences: {diff1}, {diff2}")
    print(f"  Are they the same? {diff1 == diff2}")
    print(f"  Diff % 7 = {diff1 % 7}")
    print(f"  Diff % 64 = {diff1 % 64} (cache line conflicts if 0)")


def generate_padding_function():
    """Generate the actual function to be used in the GPU code."""
    
    print("\n" + "=" * 70)
    print("Recommended implementation for gpu_systemspec_flat.py:")
    print("=" * 70)
    
    code = '''
def apply_safe_padding(spec_doubles, verbose=False):
    """
    Apply padding to SystemSpec array to avoid cache conflicts.
    
    Parameters:
    -----------
    spec_doubles : np.ndarray
        The SystemSpec data as a double array
    verbose : bool
        Print padding information
        
    Returns:
    --------
    np.ndarray
        Padded array safe from cache conflicts
    """
    base_size = len(spec_doubles)
    CACHE_LINE_DOUBLES = 8
    
    # Round up to cache line boundary
    padded_size = ((base_size + CACHE_LINE_DOUBLES - 1) // CACHE_LINE_DOUBLES) * CACHE_LINE_DOUBLES
    
    # Avoid problematic patterns
    # Key insight: avoid sizes where (size % small_prime) creates patterns
    # Especially avoid size % 7 = 6, which creates 7-stride conflicts
    
    while padded_size < base_size * 2:  # Don't more than double
        # Check for problematic patterns
        has_conflict = False
        
        # Avoid exact multiples of small primes
        for prime in [3, 5, 7]:
            if padded_size % prime == 0:
                has_conflict = True
                break
        
        # Avoid size % 7 = 6 (the specific AuBi problem)
        if padded_size % 7 == 6:
            has_conflict = True
        
        # Avoid exact multiples of GPU warp size
        if padded_size % 32 == 0:
            has_conflict = True
        
        if not has_conflict:
            break
            
        padded_size += CACHE_LINE_DOUBLES
    
    if verbose and padded_size != base_size:
        print(f"[GPU] Padding SystemSpec from {base_size} to {padded_size} doubles to avoid cache conflicts")
    
    # Create padded array
    padded_array = np.zeros(padded_size, dtype=np.float64)
    padded_array[:base_size] = spec_doubles
    
    return padded_array
'''
    print(code)


if __name__ == "__main__":
    test_padding_strategy()
    generate_padding_function()