#!/usr/bin/env python
"""Trace numerical precision loss during consolidation."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import sys
import io

# Load database and set up calculation
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test single-phase condition that requires consolidation
conditions = {v.X('TI'): 0.1, v.T: 600, v.P: 101325}

print("Tracing consolidation numerical precision")
print("="*80)
print(f"Condition: X(TI)={conditions[v.X('TI')]}, T={conditions[v.T]}K")
print()

# First, run CPU calculation to get reference
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
cpu_gm = float(result_cpu.GM.values)
print(f"CPU Reference: GM = {cpu_gm:.15f} J/mol")
print()

# Now run GPU with verbose output to trace consolidation
print("GPU Calculation with consolidation trace:")
print("-"*60)

# Capture GPU output
gpu_output = io.StringIO()
old_stdout = sys.stdout
sys.stdout = gpu_output
try:
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)
finally:
    sys.stdout = old_stdout

gpu_text = gpu_output.getvalue()
gpu_gm = float(result_gpu.GM.values)

# Look for key consolidation events
lines = gpu_text.split('\n')
consolidation_found = False
pre_consolidation_gm = None
post_consolidation_gm = None

for i, line in enumerate(lines):
    # Look for GM values before and after consolidation
    if "GM:" in line or "gm:" in line.lower():
        if "before" in line.lower() or "pre" in line.lower():
            try:
                # Extract GM value
                parts = line.split()
                for j, part in enumerate(parts):
                    if "GM" in part.upper() and j+1 < len(parts):
                        val = parts[j+1].replace(',', '').replace(':', '')
                        try:
                            pre_consolidation_gm = float(val)
                        except:
                            pass
            except:
                pass
        elif "after" in line.lower() or "post" in line.lower():
            try:
                # Extract GM value
                parts = line.split()
                for j, part in enumerate(parts):
                    if "GM" in part.upper() and j+1 < len(parts):
                        val = parts[j+1].replace(',', '').replace(':', '')
                        try:
                            post_consolidation_gm = float(val)
                        except:
                            pass
            except:
                pass
    
    # Look for consolidation events
    if "Consolidated phases" in line:
        consolidation_found = True
        print(f"CONSOLIDATION EVENT: {line}")
        # Print context around consolidation
        for j in range(max(0, i-3), min(len(lines), i+5)):
            if "phase_amt" in lines[j] or "NP" in lines[j] or "GM" in lines[j].upper():
                print(f"  Context [{j-i:+d}]: {lines[j]}")

print()
print("Results:")
print("-"*60)
print(f"GPU Final: GM = {gpu_gm:.15f} J/mol")
print(f"CPU Final: GM = {cpu_gm:.15f} J/mol")
print(f"Error: {gpu_gm - cpu_gm:.15e} J/mol")
print(f"Relative error: {abs((gpu_gm - cpu_gm)/cpu_gm)*100:.10f}%")

if consolidation_found:
    print("\nConsolidation was performed")
else:
    print("\nNo consolidation found (unexpected for this condition)")

# Look for phase fractions in the output
print("\nPhase fractions from GPU output:")
for line in lines:
    if "final phase amounts:" in line.lower() or "phase_amt[" in line:
        print(f"  {line}")
    if "Phase BCC_A2" in line and "amount=" in line:
        print(f"  {line}")

# Compare phase results
print("\nFinal phase comparison:")
cpu_phases = result_cpu.Phase.values.flatten()
cpu_np = result_cpu.NP.values.flatten()
gpu_phases = result_gpu.Phase.values.flatten()
gpu_np = result_gpu.NP.values.flatten()

for i, (cp, gp, ca, ga) in enumerate(zip(cpu_phases, gpu_phases, cpu_np, gpu_np)):
    if cp and ca > 1e-6:
        print(f"  CPU: {cp} amount={ca:.6f}")
    if gp and ga > 1e-6:
        print(f"  GPU: {gp} amount={ga:.6f}")

# Look for specific numerical operations during consolidation
print("\nSearching for numerical operations during consolidation:")
for line in lines:
    if any(keyword in line.lower() for keyword in ["fmax", "normalize", "sum", "total_amt", "1e-8"]):
        print(f"  {line}")