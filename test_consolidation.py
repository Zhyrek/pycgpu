
#!/usr/bin/env python
"""Test the specific consolidation issue with two BCC phases."""

from pycalphad import Database, equilibrium
import numpy as np
import sys
import io

# Load TDB
tdb = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['LIQUID', 'BCC_A2']

# Test a specific condition that shows the consolidation issue
T = 1200
X_TI = 0.9
P = 101325

conditions = {
    'T': T,
    'P': P,
    'X(TI)': X_TI
}

print("="*80)
print("TESTING CONSOLIDATION ISSUE")
print("="*80)
print(f"Conditions: T={T}K, X(TI)={X_TI}, P={P}")
print("="*80)

# Capture output to see debug messages
print("\n*** CPU CALCULATION ***\n")
cpu_output = io.StringIO()
old_stdout = sys.stdout
sys.stdout = cpu_output
eq_cpu = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=True)
sys.stdout = old_stdout
cpu_log = cpu_output.getvalue()

# Look for consolidation messages in CPU output
if "[CPU DEBUG] CONSOLIDATING" in cpu_log:
    print("CPU consolidation detected:")
    for line in cpu_log.split('\n'):
        if "CONSOLIDATING" in line or "consolidation" in line:
            print(f"  {line}")

print("\n*** GPU CALCULATION ***\n")
gpu_output = io.StringIO()
sys.stdout = gpu_output
eq_gpu = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=True)
sys.stdout = old_stdout
gpu_log = gpu_output.getvalue()

# Look for consolidation messages in GPU output
if "Checking phases" in gpu_log and "for consolidation" in gpu_log:
    print("GPU consolidation checks:")
    for line in gpu_log.split('\n'):
        if "consolidation" in line or "Should consolidate" in line:
            print(f"  {line}")

# Compare results
print("\n" + "="*80)
print("RESULTS COMPARISON")
print("="*80)

# Extract phase amounts and compositions
cpu_phases = eq_cpu.Phase.values.flatten()
gpu_phases = eq_gpu.Phase.values.flatten()
cpu_np = eq_cpu.NP.values.flatten()
gpu_np = eq_gpu.NP.values.flatten()

print("\nCPU Results:")
cpu_stable_count = 0
for i, (phase, amount) in enumerate(zip(cpu_phases, cpu_np)):
    if amount > 1e-10:
        cpu_stable_count += 1
        # Get composition for this phase
        x_vals = eq_cpu.X.where(eq_cpu.Phase == phase, drop=True).values
        if x_vals.ndim > 1:
            x_vals = x_vals.flatten()
        x_nb = x_vals[0] if len(x_vals) > 0 else 0.0
        x_ti = x_vals[1] if len(x_vals) > 1 else 0.0
        print(f"  {phase}: amount={amount:.15e}, X(NB)={x_nb:.6f}, X(TI)={x_ti:.6f}")
print(f"  Total stable phases: {cpu_stable_count}")

print("\nGPU Results:")
gpu_stable_count = 0
for i, (phase, amount) in enumerate(zip(gpu_phases, gpu_np)):
    if amount > 1e-10:
        gpu_stable_count += 1
        # Get composition for this phase
        x_vals = eq_gpu.X.where(eq_gpu.Phase == phase, drop=True).values
        if x_vals.ndim > 1:
            x_vals = x_vals.flatten()
        x_nb = x_vals[0] if len(x_vals) > 0 else 0.0
        x_ti = x_vals[1] if len(x_vals) > 1 else 0.0
        print(f"  {phase}: amount={amount:.15e}, X(NB)={x_nb:.6f}, X(TI)={x_ti:.6f}")
print(f"  Total stable phases: {gpu_stable_count}")

print(f"\nGM Difference: {abs(eq_cpu.GM.values.item() - eq_gpu.GM.values.item()):.6e}")
print(f"Phase Count Difference: CPU={cpu_stable_count}, GPU={gpu_stable_count}")

if cpu_stable_count != gpu_stable_count:
    print("\n*** CONSOLIDATION ISSUE DETECTED ***")
    print("GPU is not consolidating phases like CPU does!")
