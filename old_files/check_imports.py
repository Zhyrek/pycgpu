#!/usr/bin/env python3
"""Check if basic imports work and what's missing"""

import sys
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")

# Check for CuPy
try:
    import cupy
    print(f"✓ CuPy version: {cupy.__version__}")
    print(f"  CUDA available: {cupy.cuda.is_available()}")
except ImportError as e:
    print(f"✗ CuPy not available: {e}")

# Check for NumPy
try:
    import numpy
    print(f"✓ NumPy version: {numpy.__version__}")
except ImportError as e:
    print(f"✗ NumPy not available: {e}")

# Check for pycalphad
try:
    # This might fail if Cython extensions aren't built
    import pycalphad
    print(f"✓ pycalphad version: {pycalphad.__version__}")
except ImportError as e:
    print(f"✗ pycalphad not available: {e}")
    print("  This is expected if Cython extensions haven't been built yet.")

# Check if GPU module exists
try:
    from pycalphad.gpu import gpu_equilibrium
    print("✓ GPU module found")
except ImportError as e:
    print(f"✗ GPU module not available: {e}")

print("\nTo complete the setup:")
print("1. Install build-essential: sudo apt install build-essential")
print("2. Run: source venv_gpu/bin/activate && pip install -e .")
print("3. Install CuPy if needed")
print("4. Run: python test_script.py")