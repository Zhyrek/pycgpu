#!/bin/bash
# Setup script for pycalphad GPU development using Miniconda

set -e  # Exit on error

echo "=== Setting up pycalphad GPU environment with Miniconda ==="

# Check if we're in the pycalphad directory
if [ ! -f "setup.py" ]; then
    echo "Error: Please run this script from the pycalphad root directory"
    exit 1
fi

CONDA_PREFIX="$HOME/miniconda3"
ENV_NAME="pycalphad-gpu"

# Check if Miniconda is already installed
if [ ! -d "$CONDA_PREFIX" ]; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $CONDA_PREFIX
    rm miniconda.sh
else
    echo "Miniconda already installed at $CONDA_PREFIX"
fi

# Initialize conda for bash
echo "Initializing conda..."
source "$CONDA_PREFIX/etc/profile.d/conda.sh"

# Remove existing environment if it exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Removing existing $ENV_NAME environment..."
    conda env remove -n $ENV_NAME -y
fi

# Create new environment with Python 3.12, CUDA toolkit, and CuPy
echo "Creating new conda environment: $ENV_NAME"
echo "This will install Python 3.12, CUDA toolkit 12.1, and CuPy..."

conda create -n $ENV_NAME python=3.12 -y

# Activate the environment
echo "Activating environment..."
conda activate $ENV_NAME

# Install CUDA toolkit and CuPy from conda-forge
echo "Installing CUDA toolkit and CuPy..."
conda install -c conda-forge -c nvidia cuda-toolkit=12.1 cupy -y

# Install build dependencies
echo "Installing build dependencies..."
conda install -c conda-forge numpy scipy matplotlib cython setuptools wheel pip -y

# Install pycalphad dependencies
echo "Installing pycalphad dependencies..."
pip install symengine xarray tinydb pint pyparsing pytest pytest-cov

# Install pycalphad in development mode
echo "Installing pycalphad in development mode..."
pip install -e .

# Create activation script
ACTIVATE_SCRIPT="activate_gpu_env.sh"
cat > $ACTIVATE_SCRIPT << 'EOF'
#!/bin/bash
# Quick activation script for pycalphad GPU environment

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate pycalphad-gpu
    echo "Activated pycalphad-gpu environment"
    echo "Python: $(which python)"
    echo "CUDA toolkit is available via conda"
else
    echo "Error: Miniconda not found at $HOME/miniconda3"
    exit 1
fi
EOF
chmod +x $ACTIVATE_SCRIPT

# Test the installation
echo ""
echo "=== Testing installation ==="
python -c "
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
        extern \"C\" __global__
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
"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To activate this environment in the future, use one of:"
echo "  1. source activate_gpu_env.sh"
echo "  2. conda activate $ENV_NAME"
echo ""
echo "To test GPU functionality:"
echo "  python test_script.py"
echo ""
echo "Note: The conda environment includes CUDA toolkit 12.1,"
echo "so you don't need to install it separately."