#!/usr/bin/env python3
"""Setup script for pycalphad GPU development environment"""

import os
import sys
import subprocess
import venv

def run_command(cmd, check=True):
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result

def main():
    print("Setting up pycalphad GPU development environment...")
    
    # Check if we're in the pycalphad directory
    if not os.path.exists("setup.py"):
        print("Error: Please run this script from the pycalphad root directory")
        sys.exit(1)
    
    # Create virtual environment if it doesn't exist
    venv_path = "venv_gpu"
    if not os.path.exists(venv_path):
        print("Creating virtual environment...")
        venv.create(venv_path, with_pip=True)
    
    # Determine the pip executable path
    if sys.platform == "win32":
        pip_path = os.path.join(venv_path, "Scripts", "pip")
        python_path = os.path.join(venv_path, "Scripts", "python")
    else:
        pip_path = os.path.join(venv_path, "bin", "pip")
        python_path = os.path.join(venv_path, "bin", "python")
    
    # Upgrade pip
    print("Upgrading pip...")
    run_command(f"{python_path} -m pip install --upgrade pip")
    
    # Install build dependencies
    print("Installing build dependencies...")
    run_command(f"{pip_path} install --upgrade setuptools wheel cython numpy")
    
    # Install pycalphad in development mode
    print("Installing pycalphad in development mode...")
    run_command(f"{pip_path} install -e .")
    
    # Install CuPy
    print("\nPlease select your CUDA version:")
    print("1) CUDA 11.2 - 11.8")
    print("2) CUDA 12.x")
    print("3) ROCm (AMD GPUs)")
    print("4) CPU-only development (no GPU)")
    
    try:
        choice = input("Enter your choice (1-4): ").strip()
    except KeyboardInterrupt:
        print("\nSetup cancelled.")
        sys.exit(0)
    
    cupy_packages = {
        "1": "cupy-cuda11x",
        "2": "cupy-cuda12x",
        "3": "cupy-rocm-5-0"
    }
    
    if choice in cupy_packages:
        print(f"Installing {cupy_packages[choice]}...")
        run_command(f"{pip_path} install {cupy_packages[choice]}", check=False)
    elif choice == "4":
        print("Skipping CuPy installation (CPU-only mode)...")
    else:
        print("Invalid choice. Skipping CuPy installation.")
        print(f"You can install it manually later with: {pip_path} install cupy-cuda11x")
    
    # Verify installation
    print("\nVerifying installation...")
    result = run_command(f"{python_path} -c \"import pycalphad; print(f'pycalphad version: {{pycalphad.__version__}}')\"", check=False)
    if result.returncode == 0:
        print(result.stdout.strip())
    
    # Check if CuPy is available
    cupy_check = """
try:
    import cupy
    print('CuPy is installed and available')
    print(f'CuPy version: {cupy.__version__}')
    print(f'CUDA available: {cupy.cuda.is_available()}')
except ImportError:
    print('CuPy is not installed - GPU acceleration will not be available')
"""
    run_command(f'{python_path} -c "{cupy_check}"', check=False)
    
    print("\nSetup complete! To use this environment in the future, run:")
    if sys.platform == "win32":
        print(f"  {venv_path}\\Scripts\\activate")
    else:
        print(f"  source {venv_path}/bin/activate")
    print("\nTo test the GPU functionality, run:")
    print("  python test_script.py")

if __name__ == "__main__":
    main()