#!/usr/bin/env python
"""Direct comparison of specific values for X(TI)=0.1, T=600K."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import sys
import io
import os

# Temporarily enable debug output to trace calculations
os.environ['PYCALPHAD_DEBUG'] = '1'

# Load database and set up calculation
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test condition: X(TI)=0.1, T=600K
conditions = {v.X('TI'): 0.1, v.T: 600, v.P: 101325}

print("="*80)
print("DIRECT VALUE COMPARISON: X(TI)=0.1, T=600K")
print("="*80)

# Run calculations and save output
print("\nCPU CALCULATION:")
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
cpu_gm = result_cpu.GM.values.flatten()[0]
cpu_phases = result_cpu.Phase.values.flatten()
cpu_np = result_cpu.NP.values.flatten()
cpu_y = result_cpu.Y.values.flatten()
cpu_mu = result_cpu.MU.values.flatten()

print(f"  GM: {cpu_gm:.6f} J/mol")
print(f"  Chemical potentials: {cpu_mu}")
active_cpu = [(phase, np, i) for i, (phase, np) in enumerate(zip(cpu_phases, cpu_np)) if np > 1e-6]
for phase, np, idx in active_cpu:
    y_start = idx * 2  # Each phase has 2 site fractions
    print(f"  {phase}: NP={np:.6f}, Y=[{cpu_y[y_start]:.6f}, {cpu_y[y_start+1]:.6f}]")

print("\nGPU CALCULATION:")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
gpu_gm = result_gpu.GM.values.flatten()[0]
gpu_phases = result_gpu.Phase.values.flatten()
gpu_np = result_gpu.NP.values.flatten()
gpu_y = result_gpu.Y.values.flatten()
gpu_mu = result_gpu.MU.values.flatten()

print(f"  GM: {gpu_gm:.6f} J/mol")
print(f"  Chemical potentials: {gpu_mu}")
active_gpu = [(phase, np, i) for i, (phase, np) in enumerate(zip(gpu_phases, gpu_np)) if np > 1e-6]
for phase, np, idx in active_gpu:
    y_start = idx * 2  # Each phase has 2 site fractions
    print(f"  {phase}: NP={np:.6f}, Y=[{gpu_y[y_start]:.6f}, {gpu_y[y_start+1]:.6f}]")

print("\nDIFFERENCES:")
print(f"  GM difference: {gpu_gm - cpu_gm:.6f} J/mol")
print(f"  Chemical potential differences: {gpu_mu - cpu_mu}")

# Check if both converged to single phase
if len(active_cpu) == 1 and len(active_gpu) == 1:
    print("\nBoth converged to single phase")
    cpu_phase, cpu_np_val, cpu_idx = active_cpu[0]
    gpu_phase, gpu_np_val, gpu_idx = active_gpu[0]
    
    # Compare site fractions
    cpu_y_vals = [cpu_y[cpu_idx*2], cpu_y[cpu_idx*2+1]]
    gpu_y_vals = [gpu_y[gpu_idx*2], gpu_y[gpu_idx*2+1]]
    
    print(f"  CPU Y: {cpu_y_vals}")
    print(f"  GPU Y: {gpu_y_vals}")
    print(f"  Y differences: {[g-c for g,c in zip(gpu_y_vals, cpu_y_vals)]}")
    
    # Calculate energies using the same site fractions
    from pycalphad import Model
    mod = Model(dbf, comps, 'BCC_A2')
    
    # Energy at CPU site fractions
    cpu_energy = float(mod.GM.subs({
        v.T: 600, 
        v.P: 101325,
        v.Y('BCC_A2', 0, 'NB'): cpu_y_vals[0],
        v.Y('BCC_A2', 0, 'TI'): cpu_y_vals[1]
    }))
    
    # Energy at GPU site fractions  
    gpu_energy = float(mod.GM.subs({
        v.T: 600,
        v.P: 101325,
        v.Y('BCC_A2', 0, 'NB'): gpu_y_vals[0],
        v.Y('BCC_A2', 0, 'TI'): gpu_y_vals[1]
    }))
    
    print(f"\nSymbolic energy calculation:")
    print(f"  Energy at CPU Y: {cpu_energy:.6f} J/mol")
    print(f"  Energy at GPU Y: {gpu_energy:.6f} J/mol")
    print(f"  Energy difference: {gpu_energy - cpu_energy:.6f} J/mol")