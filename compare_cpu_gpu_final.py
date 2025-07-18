#!/usr/bin/env python
"""Compare CPU and GPU results directly."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("DIRECT CPU vs GPU COMPARISON")
print("=" * 60)

# Run CPU
print("\nCPU Result:")
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()
cpu_np = result_cpu.NP.values.flatten()
cpu_phases = result_cpu.Phase.values.flatten()

print(f"Number of phases: {sum(1 for np in cpu_np if np > 1e-12)}")
for i, (np_val, x_ti, phase) in enumerate(zip(cpu_np, cpu_x_ti, cpu_phases)):
    if np_val > 1e-12:
        print(f"  {phase}: NP={np_val:.6f}, X(TI)={x_ti:.6f}")

cpu_overall = sum(np_val * x_ti for np_val, x_ti in zip(cpu_np, cpu_x_ti) if np_val > 1e-12)
print(f"Overall X(TI): {cpu_overall:.10f}")

# Run GPU
print("\nGPU Result:")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()
gpu_np = result_gpu.NP.values.flatten()
gpu_phases = result_gpu.Phase.values.flatten()

print(f"Number of phases: {sum(1 for np in gpu_np if np > 1e-12)}")
for i, (np_val, x_ti, phase) in enumerate(zip(gpu_np, gpu_x_ti, gpu_phases)):
    if np_val > 1e-12:
        print(f"  {phase}: NP={np_val:.6f}, X(TI)={x_ti:.6f}")

gpu_overall = sum(np_val * x_ti for np_val, x_ti in zip(gpu_np, gpu_x_ti) if np_val > 1e-12)
print(f"Overall X(TI): {gpu_overall:.10f}")

print("\nDIFFERENCE:")
print(f"CPU overall X(TI): {cpu_overall:.10f}")
print(f"GPU overall X(TI): {gpu_overall:.10f}")
print(f"Absolute difference: {abs(cpu_overall - gpu_overall):.10f}")
print(f"Match within 0.1%: {'YES' if abs(cpu_overall - gpu_overall) < 0.001 else 'NO'}")

if abs(cpu_overall - gpu_overall) > 0.001:
    print("\nERROR: CPU and GPU results differ significantly!")
    print("The GPU is NOT correctly reproducing the CPU behavior.")