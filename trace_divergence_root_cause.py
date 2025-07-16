#!/usr/bin/env python
"""Trace the root cause of CPU vs GPU divergence for T=1000K, X(TI)=0.01."""

from pycalphad import Database, equilibrium
from pycalphad.core.debug_output import reset_debug_session, close_debug_output
import numpy as np
import re
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
print("TRACING ROOT CAUSE OF DIVERGENCE: T=1000K, X(TI)=0.01")
print("="*80)

# Capture CPU output
cpu_output = io.StringIO()
old_stdout = sys.stdout
sys.stdout = cpu_output
reset_debug_session()
eq_cpu = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=True)
sys.stdout = old_stdout
cpu_text = cpu_output.getvalue()

# Capture GPU output
gpu_output = io.StringIO()
sys.stdout = gpu_output
reset_debug_session()
eq_gpu = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=True)
sys.stdout = old_stdout
gpu_text = gpu_output.getvalue()

# Extract key values for comparison
def extract_values(text, is_gpu=False):
    """Extract numerical values from debug output."""
    values = {}
    
    # Initial energy calculations
    if is_gpu:
        # GPU: "[GPU]   phase_0_energy: -4.954985273049657e+04"
        pattern = r"\[GPU\]\s+phase_(\d+)_energy:\s+([-\d.e+]+)"
        matches = re.findall(pattern, text)
        for phase_str, energy_str in matches[:2]:  # First 2 phases only
            values[f'initial_phase_{phase_str}_energy'] = float(energy_str)
    else:
        # CPU: "[CPU] Phase 0 energy: -4.954985273049655e+04"
        pattern = r"\[CPU\] Phase (\d+) energy:\s+([-\d.e+]+)"
        matches = re.findall(pattern, text)
        for phase_str, energy_str in matches[:2]:
            values[f'initial_phase_{phase_str}_energy'] = float(energy_str)
    
    # Site fractions at start
    if is_gpu:
        # GPU: "[GPU]   phase_site_fractions: [0.995151, 0.004849]"
        pattern = r"\[GPU\]\s+phase_site_fractions:\s+\[([-\d., ]+)\]"
        matches = re.findall(pattern, text)
        for i, sf_str in enumerate(matches[:2]):
            sf_vals = [float(x.strip()) for x in sf_str.split(',')]
            values[f'initial_phase_{i}_site_fractions'] = sf_vals
    else:
        # CPU: "Site fractions: [0.9915249 0.0084751]"
        pattern = r"Site fractions:\s+\[([-\d. ]+)\]"
        matches = re.findall(pattern, text)
        for i, sf_str in enumerate(matches[:2]):
            sf_vals = [float(x) for x in sf_str.split()]
            values[f'initial_phase_{i}_site_fractions'] = sf_vals
    
    # Hessian values
    if is_gpu:
        # GPU: "Site fraction Hessian block:\n    [0] 8.355014e+03 (idx=18) 1.304530e+04"
        pattern = r"Site fraction Hessian block:.*?\[0\]\s+([-\d.e+]+).*?\[1\]\s+([-\d.e+]+).*?"
        matches = re.finditer(pattern, text, re.DOTALL)
        for i, match in enumerate(matches):
            if i < 2:  # First 2 phases
                h00 = float(match.group(1))
                values[f'phase_{i}_hessian_0_0'] = h00
    else:
        # CPU: "[CPU HESSIAN] Phase 0 (BCC_A2) Hessian after formulahess:\n  Row 3: 8.355014e+03 1.304530e+04"
        pattern = r"\[CPU HESSIAN\] Phase (\d+).*?Row 3:\s+([-\d.e+]+)\s+([-\d.e+]+)"
        matches = re.findall(pattern, text, re.DOTALL)
        for phase_str, h00, h01 in matches[:2]:
            phase_idx = int(phase_str)
            values[f'phase_{phase_idx}_hessian_0_0'] = float(h00)
    
    # Gradient values
    if is_gpu:
        # GPU: "gradient values: [-41045.66313885 -67182.86755817]"
        pattern = r"gradient values:\s+\[([-\d.e+, ]+)\]"
        matches = re.findall(pattern, text)
        for i, grad_str in enumerate(matches[:2]):
            grads = [float(x.strip()) for x in grad_str.split(',') if x.strip()]
            if len(grads) >= 2:
                values[f'phase_{i}_gradient_0'] = grads[0]
                values[f'phase_{i}_gradient_1'] = grads[1]
    else:
        # CPU: "gradient values: [-41045.66313885 -67182.86755817]"
        pattern = r"gradient values:\s+\[([-\d.e+ ]+)\]"
        matches = re.findall(pattern, text)
        for i, grad_str in enumerate(matches[:2]):
            grads = [float(x) for x in grad_str.split()]
            if len(grads) >= 2:
                values[f'phase_{i}_gradient_0'] = grads[0]
                values[f'phase_{i}_gradient_1'] = grads[1]
    
    # c_G values
    if is_gpu:
        # GPU: "c_G[0] = -1.540250896139424e-02"
        pattern = r"c_G\[0\]\s+=\s+([-\d.e+]+)"
        matches = re.findall(pattern, text)
        for i, cg_str in enumerate(matches[:2]):
            values[f'phase_{i}_c_G_0'] = float(cg_str)
    else:
        # CPU: "After c_G calc, c_G = [-0.01540251  0.01540251]"
        pattern = r"After c_G calc, c_G = \[([-\d.e+ ]+)\]"
        matches = re.findall(pattern, text)
        for i, cg_str in enumerate(matches[:2]):
            cg_vals = [float(x) for x in cg_str.split()]
            if len(cg_vals) > 0:
                values[f'phase_{i}_c_G_0'] = cg_vals[0]
    
    return values

# Extract values
cpu_values = extract_values(cpu_text, is_gpu=False)
gpu_values = extract_values(gpu_text, is_gpu=True)

print("\nCOMPARING KEY CALCULATION POINTS:")
print("-" * 80)

# Compare initial energies
print("\n1. INITIAL PHASE ENERGIES (should be identical with same site fractions):")
for phase in [0, 1]:
    key = f'initial_phase_{phase}_energy'
    if key in cpu_values and key in gpu_values:
        cpu_val = cpu_values[key]
        gpu_val = gpu_values[key]
        diff = abs(cpu_val - gpu_val)
        print(f"   Phase {phase}: CPU={cpu_val:.15e}, GPU={gpu_val:.15e}, Diff={diff:.15e}")
        if diff > 1e-10:
            print(f"   *** DIVERGENCE FOUND IN INITIAL ENERGY ***")

# Compare site fractions
print("\n2. INITIAL SITE FRACTIONS:")
for phase in [0, 1]:
    key = f'initial_phase_{phase}_site_fractions'
    if key in cpu_values and key in gpu_values:
        cpu_sf = cpu_values[key]
        gpu_sf = gpu_values[key]
        print(f"   Phase {phase}:")
        print(f"     CPU: {cpu_sf}")
        print(f"     GPU: {gpu_sf}")
        if len(cpu_sf) == len(gpu_sf):
            for i, (c, g) in enumerate(zip(cpu_sf, gpu_sf)):
                diff = abs(c - g)
                if diff > 1e-10:
                    print(f"     *** SF[{i}] DIFFERS: {diff:.15e} ***")

# Compare Hessian values
print("\n3. HESSIAN VALUES (should be identical with same inputs):")
for phase in [0, 1]:
    key = f'phase_{phase}_hessian_0_0'
    if key in cpu_values and key in gpu_values:
        cpu_val = cpu_values[key]
        gpu_val = gpu_values[key]
        diff = abs(cpu_val - gpu_val)
        print(f"   Phase {phase} H[0,0]: CPU={cpu_val:.6e}, GPU={gpu_val:.6e}, Diff={diff:.6e}")
        if diff > 1e-10:
            print(f"   *** HESSIAN DIVERGENCE ***")

# Compare gradients
print("\n4. GRADIENT VALUES (should be identical with same inputs):")
for phase in [0, 1]:
    for grad_idx in [0, 1]:
        key = f'phase_{phase}_gradient_{grad_idx}'
        if key in cpu_values and key in gpu_values:
            cpu_val = cpu_values[key]
            gpu_val = gpu_values[key]
            diff = abs(cpu_val - gpu_val)
            print(f"   Phase {phase} grad[{grad_idx}]: CPU={cpu_val:.6e}, GPU={gpu_val:.6e}, Diff={diff:.6e}")
            if diff > 1e-10:
                print(f"   *** GRADIENT DIVERGENCE ***")

# Compare c_G values
print("\n5. C_G VALUES (critical for convergence):")
for phase in [0, 1]:
    key = f'phase_{phase}_c_G_0'
    if key in cpu_values and key in gpu_values:
        cpu_val = cpu_values[key]
        gpu_val = gpu_values[key]
        diff = abs(cpu_val - gpu_val)
        print(f"   Phase {phase} c_G[0]: CPU={cpu_val:.15e}, GPU={gpu_val:.15e}, Diff={diff:.15e}")
        if diff > 1e-10:
            print(f"   *** C_G DIVERGENCE ***")

# Look for the duplicate phase warning
if "duplicate phase type" in gpu_text:
    print("\n*** GPU WARNING ABOUT DUPLICATE PHASE TYPE DETECTED ***")
    print("This indicates the GPU is handling immiscibility gaps differently than CPU")