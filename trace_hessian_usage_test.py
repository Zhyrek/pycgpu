#!/usr/bin/env python
import os
import sys
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Enable debug
os.environ['PYCALPHAD_DEBUG_CATEGORIES'] = 'HESSIAN'

# Create test case
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

conditions = {v.X('TI'): 0.01, v.T: 1000, v.P: 101325}

print("=== Tracing Hessian Usage ===")

# Run GPU only
try:
    gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True)
    print("✓ GPU completed")
except Exception as e:
    print(f"✗ GPU failed: {e}")
