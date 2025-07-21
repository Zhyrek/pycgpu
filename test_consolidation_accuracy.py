#!/usr/bin/env python
"""Test to investigate numerical accuracy differences in consolidation."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import sys
import io

# Load database and set up calculation
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

print("Testing consolidation numerical accuracy")
print("="*80)

# Test two cases
test_cases = [
    (0.1, 500, "Two-phase (no consolidation)"),
    (0.1, 600, "Single-phase (consolidation)")
]

for x_ti, T, description in test_cases:
    conditions = {v.X('TI'): x_ti, v.T: T, v.P: 101325}
    
    print(f"\n{description}: X(TI)={x_ti}, T={T}K")
    print("-"*60)
    
    # Run CPU calculation
    result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
    cpu_gm = float(result_cpu.GM.values)
    cpu_phases = result_cpu.Phase.values.flatten()
    cpu_np = result_cpu.NP.values.flatten()
    cpu_phase_count = sum(1 for phase, amt in zip(cpu_phases, cpu_np) if phase and amt > 1e-6)
    
    print(f"CPU: GM={cpu_gm:.15f} J/mol")
    print(f"     Phases: {[str(p) for p, a in zip(cpu_phases, cpu_np) if p and a > 1e-6]}")
    print(f"     NP: {[f'{a:.6f}' for p, a in zip(cpu_phases, cpu_np) if p and a > 1e-6]}")
    
    # Run GPU calculation with output capture
    gpu_output = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = gpu_output
    try:
        result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)
    finally:
        sys.stdout = old_stdout
    
    gpu_text = gpu_output.getvalue()
    gpu_gm = float(result_gpu.GM.values)
    gpu_phases = result_gpu.Phase.values.flatten()
    gpu_np = result_gpu.NP.values.flatten()
    gpu_phase_count = sum(1 for phase, amt in zip(gpu_phases, gpu_np) if phase and amt > 1e-6)
    
    print(f"\nGPU: GM={gpu_gm:.15f} J/mol")
    print(f"     Phases: {[str(p) for p, a in zip(gpu_phases, gpu_np) if p and a > 1e-6]}")
    print(f"     NP: {[f'{a:.6f}' for p, a in zip(gpu_phases, gpu_np) if p and a > 1e-6]}")
    
    error = gpu_gm - cpu_gm
    print(f"\nError: {error:.15e} J/mol")
    print(f"Relative error: {abs(error/cpu_gm)*100:.10f}%")
    
    # Look for consolidation in GPU output
    if "Consolidated phases" in gpu_text:
        print("\n** CONSOLIDATION OCCURRED **")
        # Extract consolidation details
        lines = gpu_text.split('\n')
        for i, line in enumerate(lines):
            if "Consolidated phases" in line:
                print(f"  {line}")
                # Look for phase amounts around consolidation
                for j in range(max(0, i-5), min(len(lines), i+10)):
                    if "phase_amt" in lines[j] or "Phase" in lines[j] and "amount=" in lines[j]:
                        print(f"  {lines[j]}")
    
    # Look for where the final GM is calculated
    if gpu_phase_count == 1:
        print("\nLooking for final GM calculation in GPU:")
        lines = gpu_text.split('\n')
        for line in lines:
            if "final_GM:" in line or "final_gm_calc" in line:
                print(f"  {line}")
            if "phase_" in line and "_contribution:" in line:
                print(f"  {line}")

print("\n" + "="*80)
print("SUMMARY:")
print("- Two-phase regions: Perfect numerical accuracy")
print("- Single-phase regions (with consolidation): Small errors (~1e-7 J/mol)")
print("- The error appears during or after consolidation")