#!/bin/bash
# Script to run Python with proper CUDA environment setup

# Common CUDA installation locations
CUDA_PATHS=(
    "/usr/local/cuda"
    "/usr/local/cuda-12"
    "/usr/local/cuda-12.0"
    "/usr/local/cuda-12.1"
    "/usr/local/cuda-12.2"
    "/usr/local/cuda-12.3"
    "/usr/local/cuda-12.4"
    "/usr/local/cuda-12.5"
    "/usr/local/cuda-12.6"
    "/usr/local/cuda-12.7"
    "/usr/local/cuda-12.8"
    "$HOME/cuda"
)

# Find CUDA installation
CUDA_HOME=""
for path in "${CUDA_PATHS[@]}"; do
    if [ -d "$path" ]; then
        CUDA_HOME="$path"
        break
    fi
done

if [ -z "$CUDA_HOME" ]; then
    echo "Warning: CUDA installation not found in common locations."
    echo "Attempting to run without setting CUDA paths..."
else
    echo "Found CUDA at: $CUDA_HOME"
    export CUDA_HOME
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
fi

# WSL2 specific CUDA library path
if [ -d "/usr/lib/wsl/lib" ]; then
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
fi

# Activate virtual environment
source venv_gpu/bin/activate

# Run the command
echo "Running: $@"
exec "$@"