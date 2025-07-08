#!/usr/bin/env python
"""Test single condition with CPU and GPU."""

from pycalphad import Database, equilibrium
from pycalphad.core.debug_output import reset_debug_session, close_debug_output

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

print("="*60)
print("CPU Calculation")
print("="*60)
reset_debug_session()
eq_cpu = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=False)
print(f"CPU Final GM: {eq_cpu.GM.values[0]}")
print(f"CPU Shape: {eq_cpu.GM.shape}")

print("\n" + "="*60)
print("GPU Calculation")
print("="*60) 
reset_debug_session()
eq_gpu = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=False)
print(f"GPU Final GM: {eq_gpu.GM.values[0]}")
print(f"GPU Shape: {eq_gpu.GM.shape}")

print("\n" + "="*60)
print("Comparison")
print("="*60)
print(f"GM Difference: {abs(eq_cpu.GM.values[0] - eq_gpu.GM.values[0])}")
print(f"Relative Error: {abs(eq_cpu.GM.values[0] - eq_gpu.GM.values[0]) / abs(eq_cpu.GM.values[0]) * 100:.2e}%")