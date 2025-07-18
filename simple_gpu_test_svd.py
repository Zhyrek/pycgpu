#!/usr/bin/env python
"""Simple GPU vs CPU test after SVD fix."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

# Suppress debug output
os.environ['PYCALPHAD_DEBUG'] = '0'

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Clear cache
cache_dir = os.path.expanduser('~/.cache/pycalphad_gpu')
if os.path.exists(cache_dir):
    import shutil
    shutil.rmtree(cache_dir)

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test X(TI)=0.9, T=600K
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("Testing GPU after SVD fix")
print("Condition: X(TI)=0.9, T=600K")
print("-" * 40)

# CPU
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
cpu_gm = result_cpu.GM.values.flatten()[0]
print(f"CPU GM: {cpu_gm:.2f} J/mol")

# GPU
try:
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    gpu_gm = result_gpu.GM.values.flatten()[0]
    print(f"GPU GM: {gpu_gm:.2f} J/mol")
    print(f"Error: {abs(gpu_gm - cpu_gm):.2f} J/mol")
except Exception as e:
    print(f"GPU failed: {type(e).__name__}")