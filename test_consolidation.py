
import os
import sys
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

conditions = {v.X('TI'): 0.01, v.T: 1000, v.P: 101325}

print("Testing GPU consolidation...")
result = equilibrium(dbf, comps, phases, conditions, gpu=True)
print(f"\nGPU X(TI) = {result.X.sel(component='TI').values.flatten()[0]:.8f}")
