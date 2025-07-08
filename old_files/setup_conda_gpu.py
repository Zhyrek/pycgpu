#!/usr/bin/env python3
"""Setup pycalphad GPU environment using Miniconda"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

def run_command(cmd, check=True, shell=True, capture_output=False):
    """Run a shell command"""
    print(f"Running: {cmd}")
    if capture_output:
        result = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
        if check and result.returncode != 0:
            print(f"Error: {result.stderr}")
            sys.exit(1)
        return result
    else:
        result = subprocess.run(cmd, shell=shell)
        if check and result.returncode != 0:
            sys.exit(1)
        return result

def main():
    print("=== Setting up pycalphad GPU environment with Miniconda ===")
    
    # Check if we're in the pycalphad directory
    if not os.path.exists("setup.py"):
        print("Error: Please run this script from the pycalphad root directory")
        sys.exit(1)
    
    conda_prefix = os.path.expanduser("~/miniconda3")
    env_name = "pycalphad-gpu"
    
    # Check if Miniconda is installed
    if not os.path.exists(conda_prefix):
        print("Installing Miniconda...")
        miniconda_url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        urllib.request.urlretrieve(miniconda_url, "miniconda.sh")
        run_command(f"bash miniconda.sh -b -p {conda_prefix}")
        os.remove("miniconda.sh")
    else:
        print(f"Miniconda already installed at {conda_prefix}")
    
    # Set up conda paths
    conda_sh = f"{conda_prefix}/etc/profile.d/conda.sh"
    conda_exe = f"{conda_prefix}/bin/conda"
    
    # Check if environment exists
    result = run_command(f"{conda_exe} env list", capture_output=True)
    if f"{env_name} " in result.stdout:
        print(f"Removing existing {env_name} environment...")
        run_command(f"{conda_exe} env remove -n {env_name} -y")
    
    # Create new environment
    print(f"Creating new conda environment: {env_name}")
    run_command(f"{conda_exe} create -n {env_name} python=3.12 -y")
    
    # Get environment python path
    env_python = f"{conda_prefix}/envs/{env_name}/bin/python"
    env_pip = f"{conda_prefix}/envs/{env_name}/bin/pip"
    
    # Install packages
    print("Installing CUDA toolkit and CuPy...")
    run_command(f"{conda_exe} install -n {env_name} -c conda-forge -c nvidia cuda-toolkit=12.1 cupy -y")
    
    print("Installing build dependencies...")
    run_command(f"{conda_exe} install -n {env_name} -c conda-forge numpy scipy matplotlib cython setuptools wheel pip -y")
    
    print("Installing pycalphad dependencies...")
    run_command(f"{env_pip} install symengine xarray tinydb pint pyparsing pytest pytest-cov")
    
    print("Installing pycalphad in development mode...")
    run_command(f"{env_pip} install -e .")
    
    # Create activation script
    activate_script = "activate_gpu_env.sh"
    with open(activate_script, "w") as f:
        f.write(f"""#!/bin/bash
# Quick activation script for pycalphad GPU environment

if [ -f "{conda_sh}" ]; then
    source "{conda_sh}"
    conda activate {env_name}
    echo "Activated {env_name} environment"
    echo "Python: $(which python)"
    echo "CUDA toolkit is available via conda"
else
    echo "Error: Miniconda not found at {conda_prefix}"
    exit 1
fi
""")
    os.chmod(activate_script, 0o755)
    
    # Test installation
    print("\n=== Testing installation ===")
    test_code = """
import sys
print(f'Python: {sys.version}')
print(f'Python path: {sys.executable}')

try:
    import numpy
    print(f'✓ NumPy {numpy.__version__}')
except ImportError:
    print('✗ NumPy not found')

try:
    import cupy
    print(f'✓ CuPy {cupy.__version__}')
    print(f'  CUDA available: {cupy.cuda.is_available()}')
    if cupy.cuda.is_available():
        print(f'  CUDA runtime version: {cupy.cuda.runtime.runtimeGetVersion()}')
        # Test kernel compilation
        kernel = cupy.RawKernel(r'''
        extern "C" __global__
        void test(float* x) {
            x[threadIdx.x] = threadIdx.x;
        }
        ''', 'test')
        print('  ✓ Kernel compilation works')
except ImportError:
    print('✗ CuPy not found')
except Exception as e:
    print(f'✗ CuPy error: {e}')

try:
    import pycalphad
    print(f'✓ pycalphad {pycalphad.__version__}')
except ImportError:
    print('✗ pycalphad not found')
"""
    
    run_command(f'{env_python} -c "{test_code}"', check=False)
    
    print("\n=== Setup complete! ===")
    print(f"\nTo activate this environment in the future, use one of:")
    print(f"  1. source activate_gpu_env.sh")
    print(f"  2. source {conda_sh} && conda activate {env_name}")
    print(f"\nTo test GPU functionality:")
    print(f"  python test_script.py")
    print(f"\nNote: The conda environment includes CUDA toolkit 12.1,")
    print(f"so you don't need to install it separately.")

if __name__ == "__main__":
    main()