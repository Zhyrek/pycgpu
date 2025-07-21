#!/usr/bin/env python3
"""
Force GPU kernel regeneration by clearing caches
"""

import os
import shutil
import cupy

# Clear CuPy kernel cache
print("Clearing CuPy kernel cache...")
try:
    # Try to get cache directory
    cache_dir = os.path.expanduser("~/.cupy/kernel_cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Cleared {cache_dir}")
    
    # Also try common cache locations
    for cache_path in ["~/.cache/cupy", "~/.cupy", "/tmp/cupy-cache*"]:
        expanded = os.path.expanduser(cache_path)
        if "*" in expanded:
            import glob
            for path in glob.glob(expanded):
                if os.path.exists(path):
                    shutil.rmtree(path)
                    print(f"Cleared {path}")
        elif os.path.exists(expanded):
            shutil.rmtree(expanded)
            print(f"Cleared {expanded}")
            
except Exception as e:
    print(f"Error clearing cache: {e}")

# Force Python to reimport modules
print("\nClearing Python module cache...")
import sys
modules_to_clear = [m for m in sys.modules.keys() if 'pycalphad' in m]
for module in modules_to_clear:
    del sys.modules[module]
    print(f"Cleared module: {module}")

print("\nCache clearing complete. GPU kernels will be regenerated on next run.")