#\!/usr/bin/env python
"""Compare CPU and GPU equilibrium results."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("Comparing CPU vs GPU equilibrium calculations...")
print("=" * 50)

# Run CPU calculation
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()[0]

# Run GPU calculation
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()[0]

print(f"\nTarget X(TI): 0.9000000000")
print(f"CPU X(TI):    {cpu_x_ti:.10f}")
print(f"GPU X(TI):    {gpu_x_ti:.10f}")
print(f"Difference:   {abs(gpu_x_ti - cpu_x_ti):.10e}")

if abs(gpu_x_ti - cpu_x_ti) < 1e-6:
    print("\n✅ SUCCESS: GPU and CPU agree within tolerance!")
else:
    print(f"\n❌ FAILED: GPU and CPU differ by {abs(gpu_x_ti - cpu_x_ti):.10e}")

# Also check that both are close to the target
if abs(cpu_x_ti - 0.9) < 1e-6 and abs(gpu_x_ti - 0.9) < 1e-6:
    print("✅ Both CPU and GPU converged to the correct value!")
else:
    if abs(cpu_x_ti - 0.9) >= 1e-6:
        print(f"❌ CPU error: {cpu_x_ti - 0.9:.10e}")
    if abs(gpu_x_ti - 0.9) >= 1e-6:
        print(f"❌ GPU error: {gpu_x_ti - 0.9:.10e}")
