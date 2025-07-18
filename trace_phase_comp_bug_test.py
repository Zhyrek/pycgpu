
import os
import sys
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

print("=== Tracing Phase Composition Bug ===\n")

conditions = {v.X('TI'): 0.01, v.T: 1000, v.P: 101325}

try:
    gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True)
    print(f"\nGPU completed: X(TI) = {gpu_result.X.sel(component='TI').values.flatten()[0]:.6f}")
except Exception as e:
    print(f"GPU failed: {e}")
