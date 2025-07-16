#!/usr/bin/env python
"""Detailed trace to find where CPU and GPU calculations diverge."""

from pycalphad import Database, equilibrium
from pycalphad.core.debug_output import reset_debug_session, close_debug_output
import numpy as np

# Load TDB
tdb = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['LIQUID', 'BCC_A2']

# Single condition
conditions = {
    'T': 1000,
    'P': 101325,
    'X(TI)': 0.5
}

print("="*80)
print("DETAILED CPU vs GPU DIVERGENCE TRACE")
print("="*80)

# Run CPU with verbose output
print("\n" + "="*60)
print("CPU DETAILED OUTPUT")
print("="*60)
reset_debug_session()
eq_cpu = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=True)

# Run GPU with verbose output
print("\n" + "="*60)
print("GPU DETAILED OUTPUT")  
print("="*60)
reset_debug_session()
eq_gpu = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=True)

# Compare results
print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)
print(f"CPU Final GM: {eq_cpu.GM.values.item():.15e}")
print(f"GPU Final GM: {eq_gpu.GM.values.item():.15e}")
print(f"Absolute Difference: {abs(eq_cpu.GM.values.item() - eq_gpu.GM.values.item()):.15e}")
print(f"Relative Error: {abs(eq_cpu.GM.values.item() - eq_gpu.GM.values.item()) / abs(eq_cpu.GM.values.item()) * 100:.15e}%")

# Check phase amounts
print("\n--- Phase Amounts ---")
cpu_np = eq_cpu.NP.values.flatten()
gpu_np = eq_gpu.NP.values.flatten()
print(f"CPU NP: {cpu_np}")
print(f"GPU NP: {gpu_np}")
print(f"NP Difference: {np.abs(cpu_np - gpu_np)}")

# Check site fractions if available
if hasattr(eq_cpu, 'Y') and hasattr(eq_gpu, 'Y'):
    print("\n--- Site Fractions ---")
    cpu_y = eq_cpu.Y.values.flatten()
    gpu_y = eq_gpu.Y.values.flatten()
    print(f"CPU Y: {cpu_y}")
    print(f"GPU Y: {gpu_y}")
    print(f"Y Difference: {np.abs(cpu_y - gpu_y)}")