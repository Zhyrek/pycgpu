#!/usr/bin/env python
"""Compare CPU and GPU Hessian calculations to find divergence."""

from pycalphad import Database, equilibrium
from pycalphad.core.debug_output import reset_debug_session, close_debug_output
import numpy as np
import re

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

# Capture output to analyze
import io
import sys

# Run CPU calculation
print("="*80)
print("CAPTURING CPU HESSIAN VALUES")
print("="*80)
cpu_output = io.StringIO()
old_stdout = sys.stdout
sys.stdout = cpu_output
reset_debug_session()
eq_cpu = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=True)
sys.stdout = old_stdout
cpu_text = cpu_output.getvalue()

# Run GPU calculation
print("\n" + "="*80)
print("CAPTURING GPU HESSIAN VALUES")
print("="*80)
gpu_output = io.StringIO()
sys.stdout = gpu_output
reset_debug_session()
eq_gpu = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=True)
sys.stdout = old_stdout
gpu_text = gpu_output.getvalue()

# Extract Hessian values from output
def extract_hessian_values(text, is_gpu=False):
    """Extract Hessian values from debug output."""
    hessians = []
    
    if is_gpu:
        # GPU format: "[GPU HESSIAN] Phase 0 has Hessian function, values:"
        # Then: "  H[1,1] (index 4) = 1.635185e+04"
        pattern = r"\[GPU HESSIAN\] Phase (\d+) has Hessian function"
        matches = list(re.finditer(pattern, text))
        
        for match in matches:
            phase_idx = int(match.group(1))
            start_pos = match.end()
            # Find the next few lines with H[i,j] values
            h_pattern = r"H\[(\d+),(\d+)\].*?= ([-+]?\d+\.?\d*e[+-]?\d+)"
            h_matches = re.finditer(h_pattern, text[start_pos:start_pos+1000])
            
            hess_dict = {}
            for h_match in h_matches:
                i, j, val = int(h_match.group(1)), int(h_match.group(2)), float(h_match.group(3))
                hess_dict[(i,j)] = val
            
            if hess_dict:
                hessians.append((phase_idx, hess_dict))
    else:
        # CPU format: "[CPU HESSIAN] Phase 0 (BCC_A2) Hessian after formulahess:"
        # Then: "  Row 3: 1.635185e+04 1.304530e+04"
        pattern = r"\[CPU HESSIAN\] Phase (\d+) \((\w+)\) Hessian after formulahess:"
        matches = list(re.finditer(pattern, text))
        
        for match in matches:
            phase_idx = int(match.group(1))
            phase_name = match.group(2)
            start_pos = match.end()
            
            # Find the row values
            row_pattern = r"Row (\d+): (.*?)$"
            row_matches = re.finditer(row_pattern, text[start_pos:start_pos+500], re.MULTILINE)
            
            hess_dict = {}
            for row_match in row_matches:
                row_idx = int(row_match.group(1))
                values = row_match.group(2).strip().split()
                for col_idx, val_str in enumerate(values):
                    if val_str:
                        # CPU shows rows starting from num_statevars
                        hess_dict[(row_idx, 3 + col_idx)] = float(val_str)
            
            if hess_dict:
                hessians.append((phase_idx, hess_dict))
    
    return hessians

# Extract values
cpu_hessians = extract_hessian_values(cpu_text, is_gpu=False)
gpu_hessians = extract_hessian_values(gpu_text, is_gpu=True)

print("\n" + "="*80)
print("HESSIAN COMPARISON")
print("="*80)

print(f"\nCPU Hessians found: {len(cpu_hessians)}")
for phase_idx, hess_dict in cpu_hessians:
    print(f"  Phase {phase_idx}:")
    for (i,j), val in sorted(hess_dict.items()):
        print(f"    H[{i},{j}] = {val:.6e}")

print(f"\nGPU Hessians found: {len(gpu_hessians)}")
for phase_idx, hess_dict in gpu_hessians:
    print(f"  Phase {phase_idx}:")
    for (i,j), val in sorted(hess_dict.items()):
        print(f"    H[{i},{j}] = {val:.6e}")

# Compare values
print("\n" + "="*80)
print("HESSIAN DIFFERENCES")
print("="*80)

# Map GPU indices to CPU indices
# GPU uses model indices (1,1), (1,2), etc.
# CPU uses workspace indices (3,3), (3,4), etc.
gpu_to_cpu_map = {
    (1,1): (3,3),
    (1,2): (3,4),
    (2,1): (4,3),
    (2,2): (4,4)
}

for gpu_phase_idx, gpu_hess in gpu_hessians:
    # Find corresponding CPU phase
    cpu_phase = None
    for cpu_phase_idx, cpu_hess in cpu_hessians:
        if cpu_phase_idx == gpu_phase_idx:
            cpu_phase = cpu_hess
            break
    
    if cpu_phase is not None:
        print(f"\nPhase {gpu_phase_idx} Hessian comparison:")
        for gpu_idx, cpu_idx in gpu_to_cpu_map.items():
            if gpu_idx in gpu_hess and cpu_idx in cpu_phase:
                gpu_val = gpu_hess[gpu_idx]
                cpu_val = cpu_phase[cpu_idx]
                diff = abs(gpu_val - cpu_val)
                rel_err = diff / abs(cpu_val) if cpu_val != 0 else float('inf')
                print(f"  GPU H{gpu_idx} vs CPU H{cpu_idx}: {gpu_val:.6e} vs {cpu_val:.6e}")
                print(f"    Difference: {diff:.6e} ({rel_err*100:.3e}%)")

# Also check final results
print("\n" + "="*80)
print("FINAL RESULT COMPARISON")
print("="*80)
print(f"CPU Final GM: {eq_cpu.GM.values.item():.15e}")
print(f"GPU Final GM: {eq_gpu.GM.values.item():.15e}")
print(f"Absolute Difference: {abs(eq_cpu.GM.values.item() - eq_gpu.GM.values.item()):.15e}")

# Look for c_G calculations in output
print("\n" + "="*80)
print("C_G CALCULATION COMPARISON")
print("="*80)

# Extract c_G values from CPU output
cpu_cg_pattern = r"\[CPU c_G DEBUG\] Phase (\d+) calculation:.*?After c_G calc, c_G = \[(.*?)\]"
cpu_cg_matches = re.finditer(cpu_cg_pattern, cpu_text, re.DOTALL)

print("CPU c_G values:")
for match in cpu_cg_matches:
    phase_idx = int(match.group(1))
    cg_values = match.group(2).strip()
    print(f"  Phase {phase_idx}: c_G = [{cg_values}]")

# Extract c_G values from GPU output
gpu_cg_pattern = r"GPU DEBUG: Phase (\d+) c_G = \[(.*?)\]"
gpu_cg_matches = re.finditer(gpu_cg_pattern, gpu_text)

print("\nGPU c_G values:")
for match in gpu_cg_matches:
    phase_idx = int(match.group(1))
    cg_values = match.group(2).strip()
    print(f"  Phase {phase_idx}: c_G = [{cg_values}]")