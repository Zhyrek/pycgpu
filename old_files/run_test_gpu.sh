#!/bin/bash
# Quick script to run test with GPU environment

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pycalphad-gpu
export CUPY_NVCC_FLAGS='-allow-unsupported-compiler'
python test_script.py