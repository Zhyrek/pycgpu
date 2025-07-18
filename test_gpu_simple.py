#!/usr/bin/env python
"""Simple test of GPU after SVD changes."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import shutil

# Clear all caches
cache_dir = os.path.expanduser('~/.cache/pycalphad_gpu')
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

# Also clear any __pycache__ directories
for root, dirs, files in os.walk('.'):
    if '__pycache__' in dirs:
        shutil.rmtree(os.path.join(root, '__pycache__'))

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("Testing GPU with SVD improvements")
print("-" * 40)

# Try GPU
try:
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    gpu_gm = result_gpu.GM.values.flatten()[0]
    print(f"GPU GM: {gpu_gm:.2f} J/mol")
    print("GPU calculation succeeded!")
except Exception as e:
    print(f"GPU failed: {type(e).__name__}")
    print(f"Error: {str(e)[:200]}...")  # First 200 chars of error