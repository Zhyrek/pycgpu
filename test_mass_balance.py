
import os
import sys
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

conditions = {v.X('TI'): 0.01, v.T: 1000, v.P: 101325}

print("Testing mass balance debug...")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True)
gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()[0]
print(f"\nFinal GPU X(TI) = {gpu_x_ti:.8f}")
print(f"Target X(TI) = 0.01000000")
print(f"Error = {gpu_x_ti - 0.01:.2e}")
