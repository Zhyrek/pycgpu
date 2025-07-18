
import os
import sys
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

conditions = {v.X('TI'): 0.01, v.T: 1000, v.P: 101325}

print("Testing GPU with miscibility gap fix...")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True)
gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()[0]

result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False)
cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()[0]

print(f"GPU X(TI) = {gpu_x_ti:.8f}")
print(f"CPU X(TI) = {cpu_x_ti:.8f}")
print(f"Difference: {abs(gpu_x_ti - cpu_x_ti):.2e}")

if abs(gpu_x_ti - cpu_x_ti) < 1e-6:
    print("✓ SUCCESS: GPU and CPU match!")
else:
    print(f"✗ FAIL: Still differ by {100*abs(gpu_x_ti - cpu_x_ti)/cpu_x_ti:.2f}%")
