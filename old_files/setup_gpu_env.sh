#!/bin/bash
# Setup script for pycalphad GPU development environment

echo "Setting up pycalphad GPU development environment..."

# Check if we're in the pycalphad directory
if [ ! -f "setup.py" ]; then
    echo "Error: Please run this script from the pycalphad root directory"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv_gpu" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv_gpu
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv_gpu/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install build dependencies
echo "Installing build dependencies..."
pip install --upgrade setuptools wheel cython numpy

# Install pycalphad in development mode
echo "Installing pycalphad in development mode..."
pip install -e .

# Install CuPy
echo "Installing CuPy..."
echo "Please select your CUDA version:"
echo "1) CUDA 11.2 - 11.8"
echo "2) CUDA 12.x"
echo "3) ROCm (AMD GPUs)"
echo "4) CPU-only development (no GPU)"
read -p "Enter your choice (1-4): " cuda_choice

case $cuda_choice in
    1)
        echo "Installing CuPy for CUDA 11.x..."
        pip install cupy-cuda11x
        ;;
    2)
        echo "Installing CuPy for CUDA 12.x..."
        pip install cupy-cuda12x
        ;;
    3)
        echo "Installing CuPy for ROCm..."
        pip install cupy-rocm-5-0
        ;;
    4)
        echo "Skipping CuPy installation (CPU-only mode)..."
        ;;
    *)
        echo "Invalid choice. Skipping CuPy installation."
        echo "You can install it manually later with: pip install cupy-cuda11x"
        ;;
esac

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import pycalphad; print(f'pycalphad version: {pycalphad.__version__}')"

# Check if CuPy is available
python -c "
try:
    import cupy
    print('CuPy is installed and available')
    print(f'CuPy version: {cupy.__version__}')
    print(f'CUDA available: {cupy.cuda.is_available()}')
except ImportError:
    print('CuPy is not installed - GPU acceleration will not be available')
"

echo ""
echo "Setup complete! To use this environment in the future, run:"
echo "  source venv_gpu/bin/activate"
echo ""
echo "To test the GPU functionality, run:"
echo "  python test_script.py"