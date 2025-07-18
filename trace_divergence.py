#!/usr/bin/env python
"""Trace CPU vs GPU calculation to find first divergence point."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Problematic condition
conditions = {v.X('TI'): 0.5, v.T: 700, v.P: 101325}

print("TRACING DIVERGENCE: X(TI)=0.5, T=700K")
print("=" * 50)

# First, check the starting point inputs
print("Input conditions:")
print(f"  X(TI): {conditions[v.X('TI')]}")
print(f"  T: {conditions[v.T]} K")
print(f"  P: {conditions[v.P]} Pa")

# Run CPU calculation and extract debug output
print("\nRunning CPU calculation...")
import subprocess
import tempfile

# Create script to capture CPU debug output
with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write(f"""
import sys
sys.path.insert(0, '{os.getcwd()}')
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {{v.X('TI'): 0.5, v.T: 700, v.P: 101325}}

print("=== CPU CALCULATION ===")
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
print(f"CPU_FINAL_GM: {{result_cpu.GM.values.flatten()[0]:.6f}}")
print(f"CPU_FINAL_PHASES: {{result_cpu.Phase.values.flatten()}}")
""")
    cpu_script = f.name

# Create script to capture GPU debug output  
with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write(f"""
import sys
sys.path.insert(0, '{os.getcwd()}')
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {{v.X('TI'): 0.5, v.T: 700, v.P: 101325}}

print("=== GPU CALCULATION ===")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
print(f"GPU_FINAL_GM: {{result_gpu.GM.values.flatten()[0]:.6f}}")
print(f"GPU_FINAL_PHASES: {{result_gpu.Phase.values.flatten()}}")
""")
    gpu_script = f.name

try:
    # Run CPU and capture all output
    print("Capturing CPU output...")
    cpu_result = subprocess.run(['python', cpu_script], capture_output=True, text=True, timeout=120)
    
    print("Capturing GPU output...")  
    gpu_result = subprocess.run(['python', gpu_script], capture_output=True, text=True, timeout=120)
    
    # Extract final values
    import re
    
    cpu_gm_match = re.search(r'CPU_FINAL_GM: ([-\d.]+)', cpu_result.stdout)
    gpu_gm_match = re.search(r'GPU_FINAL_GM: ([-\d.]+)', gpu_result.stdout)
    
    cpu_phases_match = re.search(r'CPU_FINAL_PHASES: \[(.*?)\]', cpu_result.stdout)
    gpu_phases_match = re.search(r'GPU_FINAL_PHASES: \[(.*?)\]', gpu_result.stdout)
    
    if cpu_gm_match and gpu_gm_match:
        cpu_gm = float(cpu_gm_match.group(1))
        gpu_gm = float(gpu_gm_match.group(1))
        
        print(f"\nFINAL RESULTS:")
        print(f"  CPU GM: {cpu_gm:.6f} J/mol")
        print(f"  GPU GM: {gpu_gm:.6f} J/mol")
        print(f"  Difference: {abs(cpu_gm - gpu_gm):.6f} J/mol")
        
        if cpu_phases_match and gpu_phases_match:
            cpu_phases_str = cpu_phases_match.group(1)
            gpu_phases_str = gpu_phases_match.group(1)
            print(f"  CPU phases: [{cpu_phases_str}]")
            print(f"  GPU phases: [{gpu_phases_str}]")
    
    # Look for specific debug patterns in the output to find divergence
    print(f"\nLOOKING FOR DIVERGENCE POINTS...")
    
    # Check CPU Hessian values
    cpu_hess_matches = re.findall(r'hess\[3,3\] = ([\d.e+-]+)', cpu_result.stderr)
    cpu_hess_44_matches = re.findall(r'hess\[4,4\] = ([\d.e+-]+)', cpu_result.stderr)
    
    # Check GPU Hessian values  
    gpu_hess_matches = re.findall(r'hess.*?\[0\] ([\d.e+-]+).*?idx=18', gpu_result.stderr)
    gpu_hess_44_matches = re.findall(r'hess.*?\[1\].*?idx=24.*?([\d.e+-]+)', gpu_result.stderr)
    
    if cpu_hess_matches and gpu_hess_matches:
        print(f"\nHESSIAN COMPARISON (first few values):")
        for i, (cpu_val, gpu_val) in enumerate(zip(cpu_hess_matches[:3], gpu_hess_matches[:3])):
            cpu_h = float(cpu_val)
            gpu_h = float(gpu_val)
            print(f"  Hess[3,3] #{i}: CPU={cpu_h:.6f}, GPU={gpu_h:.6f}, diff={abs(cpu_h-gpu_h):.6f}")
    
    # Check for energy values
    cpu_energy_matches = re.findall(r'Energy: ([-\d.e+-]+)', cpu_result.stderr)
    gpu_energy_matches = re.findall(r'phase_\d+_energy: ([-\d.e+-]+)', gpu_result.stderr)
    
    if cpu_energy_matches and gpu_energy_matches:
        print(f"\nENERGY COMPARISON (first few values):")
        for i, (cpu_energy, gpu_energy) in enumerate(zip(cpu_energy_matches[:3], gpu_energy_matches[:3])):
            cpu_e = float(cpu_energy)
            gpu_e = float(gpu_energy)
            print(f"  Energy #{i}: CPU={cpu_e:.6f}, GPU={gpu_e:.6f}, diff={abs(cpu_e-gpu_e):.6f}")
    
    # Look for starting point differences
    cpu_starting_matches = re.findall(r'phase_amounts\[0\]=([\d.e+-]+)', cpu_result.stderr)
    gpu_starting_matches = re.findall(r'phase_amounts\[0\]=([\d.e+-]+)', gpu_result.stderr)
    
    if cpu_starting_matches and gpu_starting_matches:
        print(f"\nSTARTING POINT COMPARISON:")
        cpu_start = float(cpu_starting_matches[0])
        gpu_start = float(gpu_starting_matches[0])
        print(f"  Initial phase_amounts[0]: CPU={cpu_start:.6f}, GPU={gpu_start:.6f}, diff={abs(cpu_start-gpu_start):.6f}")
    
    # Check site fractions if available
    cpu_site_matches = re.findall(r'site_fractions.*?([\d.e+-]+)', cpu_result.stderr)
    gpu_site_matches = re.findall(r'site_fractions.*?([\d.e+-]+)', gpu_result.stderr)
    
    print(f"\nRAW DEBUG OUTPUT SIZES:")
    print(f"  CPU stderr: {len(cpu_result.stderr)} characters")
    print(f"  GPU stderr: {len(gpu_result.stderr)} characters")
    print(f"  CPU stdout: {len(cpu_result.stdout)} characters")
    print(f"  GPU stdout: {len(gpu_result.stdout)} characters")
    
    # Save full outputs to files for detailed analysis
    with open('/tmp/cpu_debug.txt', 'w') as f:
        f.write("=== CPU STDERR ===\n")
        f.write(cpu_result.stderr)
        f.write("\n=== CPU STDOUT ===\n")
        f.write(cpu_result.stdout)
    
    with open('/tmp/gpu_debug.txt', 'w') as f:
        f.write("=== GPU STDERR ===\n")
        f.write(gpu_result.stderr)
        f.write("\n=== GPU STDOUT ===\n")
        f.write(gpu_result.stdout)
    
    print(f"\nFull debug outputs saved to /tmp/cpu_debug.txt and /tmp/gpu_debug.txt")
    print(f"Run 'grep -n \"specific_pattern\" /tmp/cpu_debug.txt /tmp/gpu_debug.txt' to find divergence")
    
finally:
    os.unlink(cpu_script)
    os.unlink(gpu_script)

print("\nTo find exact divergence point, examine:")
print("1. Initial starting point values")  
print("2. First iteration energy calculations")
print("3. Hessian matrix values")
print("4. Phase stability decisions")