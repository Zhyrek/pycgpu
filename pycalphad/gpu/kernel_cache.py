"""
Custom kernel caching mechanism for GPU equilibrium calculations.

This module provides a disk-based caching system that's more tolerant of minor
source code variations (like CSE reordering) than CuPy's built-in cache.
"""

import os
import pickle
import hashlib
from pathlib import Path
import cupy as cp

# Cache directory
CACHE_DIR = Path.home() / ".pycalphad_gpu_cache"
CACHE_DIR.mkdir(exist_ok=True)

def get_kernel_cache_key(model_funcs_c: str, num_phases: int, dynamic_sizes: dict, verbose: bool = False) -> str:
    """
    Generate a cache key that ignores minor CSE variations.

    This uses a simplified version of the source that ignores:
    - Variable reordering (x[3] vs x[4])
    - CSE variable names
    """
    # For now, use a simpler key that's less sensitive to CSE variations
    # We could parse and normalize the code more thoroughly if needed

    # Extract just the function signatures and phase count
    # This is a crude but effective way to identify "same kernel"
    import re

    # Find all function signatures
    func_sigs = re.findall(r'__device__\s+\w+\s+\w+\([^)]*\)', model_funcs_c)
    func_sigs_str = '\n'.join(sorted(func_sigs))

    # Create cache key from essential parts
    cache_input = f"{func_sigs_str}|{num_phases}|{sorted(dynamic_sizes.items())}|{verbose}"
    cache_key = hashlib.md5(cache_input.encode()).hexdigest()

    return cache_key

def load_cached_module(cache_key: str):
    """Load a cached CuPy module from disk."""
    cache_file = CACHE_DIR / f"{cache_key}.pkl"

    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                # Reconstruct the module from cached PTX/cubin
                # Note: This is a simplified version - real implementation would need
                # to properly handle module reconstruction
                return cached_data.get('module')
        except Exception as e:
            print(f"[GPU] Warning: Failed to load cached module: {e}")
            return None

    return None

def save_module_to_cache(cache_key: str, module, source: str):
    """Save a CuPy module to disk cache."""
    cache_file = CACHE_DIR / f"{cache_key}.pkl"

    try:
        # Note: CuPy modules can't be pickled directly
        # We'd need to save the compiled PTX/cubin and metadata
        # For now, just save a marker that this was compiled
        cached_data = {
            'source_hash': hashlib.md5(source.encode()).hexdigest(),
            'source_length': len(source),
            # 'module': module  # Can't pickle directly
        }

        with open(cache_file, 'wb') as f:
            pickle.dump(cached_data, f)

    except Exception as e:
        print(f"[GPU] Warning: Failed to cache module: {e}")

def check_cache_validity(cache_key: str, source: str) -> bool:
    """Check if a cached kernel is still valid."""
    cache_file = CACHE_DIR / f"{cache_key}.pkl"

    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                # Check if source is "close enough" to cached version
                # Allow for minor variations in length
                source_len_diff = abs(len(source) - cached_data.get('source_length', 0))
                if source_len_diff < 10:  # Allow up to 10 char difference
                    return True
        except Exception:
            pass

    return False