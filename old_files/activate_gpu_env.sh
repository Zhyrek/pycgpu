#!/bin/bash
# Quick activation script for pycalphad GPU environment

if [ -f "/home/scott/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/home/scott/miniconda3/etc/profile.d/conda.sh"
    conda activate pycalphad-gpu
    echo "Activated pycalphad-gpu environment"
    echo "Python: $(which python)"
    echo "CUDA toolkit is available via conda"
else
    echo "Error: Miniconda not found at /home/scott/miniconda3"
    exit 1
fi
