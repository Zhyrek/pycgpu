#!/usr/bin/env python
"""Test to verify the consolidation issue between CPU and GPU."""

from pycalphad import Database, equilibrium
from pycalphad.core.debug_output import reset_debug_session, close_debug_output
import numpy as np
import sys
import io

# Load TDB
tdb = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['LIQUID', 'BCC_A2']

# Test the specific divergent case
conditions = {
    'T': 1000,
    'P': 101325,
    'X(TI)': 0.01
}

print("="*80)
print("TESTING PHASE CONSOLIDATION: T=1000K, X(TI)=0.01")
print("="*80)

# Run CPU calculation
print("\nCPU Calculation:")
print("-" * 40)
cpu_output = io.StringIO()
old_stdout = sys.stdout
sys.stdout = cpu_output
reset_debug_session()
eq_cpu = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=True)
sys.stdout = old_stdout
cpu_text = cpu_output.getvalue()

# Check for consolidation in CPU output
if "CONSOLIDATING phases" in cpu_text:
    print("CPU: Phase consolidation occurred")
    # Extract the consolidation message
    import re
    consol_match = re.search(r"\[CPU DEBUG\] CONSOLIDATING phases (\d+) and (\d+): max_diff=([\d.]+)", cpu_text)
    if consol_match:
        phase1, phase2, max_diff = consol_match.groups()
        print(f"  Consolidated phases {phase1} and {phase2}, max composition diff: {float(max_diff):.6f}")
else:
    print("CPU: No phase consolidation")

# Run GPU calculation
print("\nGPU Calculation:")
print("-" * 40)
gpu_output = io.StringIO()
sys.stdout = gpu_output
reset_debug_session()
eq_gpu = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=True)
sys.stdout = old_stdout
gpu_text = gpu_output.getvalue()

# Check for consolidation in GPU output
if "remove_and_consolidate_phases called" in gpu_text:
    print("GPU: Consolidation function was called")
    # Check if actual consolidation happened
    if "Should consolidate: YES" in gpu_text:
        print("GPU: Phase consolidation occurred")
    elif "Should consolidate: NO" in gpu_text:
        print("GPU: Consolidation check performed but phases not consolidated")
    else:
        print("GPU: Consolidation function called but no consolidation messages found")
else:
    print("GPU: Consolidation function was NOT called")

# Check for the duplicate phase warning
if "duplicate phase type" in gpu_text:
    print("\nGPU: WARNING - Duplicate phase type detected (immiscibility gap)")

# Compare final results
print("\nFinal Results:")
print("-" * 40)
cpu_gm = eq_cpu.GM.values.item()
gpu_gm = eq_gpu.GM.values.item()
print(f"CPU GM: {cpu_gm:.6f} J/mol")
print(f"GPU GM: {gpu_gm:.6f} J/mol")
print(f"Difference: {abs(cpu_gm - gpu_gm):.6f} J/mol")

# Check final phase compositions
print("\nFinal Phase Amounts:")
cpu_np = eq_cpu.NP.values.flatten()
gpu_np = eq_gpu.NP.values.flatten()
cpu_phases = eq_cpu.Phase.values.flatten()
gpu_phases = eq_gpu.Phase.values.flatten()

print("CPU:")
for i, (phase, amount) in enumerate(zip(cpu_phases, cpu_np)):
    if amount > 1e-6:
        print(f"  {phase}: {amount:.6f}")

print("GPU:")
for i, (phase, amount) in enumerate(zip(gpu_phases, gpu_np)):
    if amount > 1e-6:
        print(f"  {phase}: {amount:.6f}")

print("\n" + "="*80)
print("ANALYSIS:")
print("="*80)
print("The divergence is caused by different phase consolidation behavior:")
print("- CPU consolidates two BCC_A2 phases when compositions are within 1e-4")
print("- GPU may not be performing the same consolidation")
print("- This leads to different convergence paths and final results")