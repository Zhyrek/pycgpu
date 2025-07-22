#!/usr/bin/env python
"""Extract and compare CPU vs GPU iterations from debug output."""

import re

with open('full_debug_output.txt', 'r') as f:
    content = f.read()

# Extract CPU iteration data
cpu_iterations = []
gpu_iterations = []

# Find CPU TRACE sections
cpu_trace_pattern = r'\[CPU TRACE\] ===== AFTER ITERATION (\d+) =====.*?\[CPU TRACE\] ===== END ITERATION \d+ ====='
cpu_traces = re.findall(cpu_trace_pattern, content, re.DOTALL)

# Find GPU TRACE sections  
gpu_trace_pattern = r'\[GPU TRACE\] ===== AFTER ITERATION (\d+) =====.*?\[GPU TRACE\] ===== END ITERATION \d+ ====='
gpu_traces = re.findall(gpu_trace_pattern, content, re.DOTALL)

# Also look for key values
print("=== ITERATION 0 COMPARISON ===\n")

# CPU Iteration 0
print("CPU ITERATION 0:")
cpu_iter0 = re.search(r'\[CPU TRACE\] ===== AFTER ITERATION 0 =====.*?\[CPU TRACE\] ===== END ITERATION 0 =====', content, re.DOTALL)
if cpu_iter0:
    trace = cpu_iter0.group(0)
    # Extract key values
    chem_pot = re.search(r'Chemical potentials: \[(.*?)\]', trace)
    if chem_pot:
        print(f"  Chemical potentials: {chem_pot.group(1)}")
    
    # Phase info
    phases = re.findall(r'Phase (\d+) \((\w+)\):.*?phase_amt.*?: ([\d.e+-]+).*?energy: ([\d.e+-]+).*?Site fractions: \[([\d., ]+)\]', trace, re.DOTALL)
    for phase_num, phase_name, amt, energy, site_frac in phases:
        print(f"  Phase {phase_num} ({phase_name}):")
        print(f"    Amount: {amt}")
        print(f"    Energy: {energy}")
        print(f"    Site fractions: [{site_frac}]")

print("\nGPU ITERATION 0:")
gpu_iter0 = re.search(r'\[GPU TRACE\] ===== AFTER ITERATION 0 =====.*?\[GPU TRACE\] ===== END ITERATION 0 =====', content, re.DOTALL)
if gpu_iter0:
    trace = gpu_iter0.group(0)
    # Extract key values
    chem_pot = re.search(r'Chemical potentials: \[(.*?)\]', trace)
    if chem_pot:
        print(f"  Chemical potentials: {chem_pot.group(1)}")
    
    # Phase info
    phases = re.findall(r'Phase (\d+) \((\w+)\):.*?phase_amt.*?: ([\d.e+-]+).*?energy: ([\d.e+-]+).*?Site fractions: \[([\d., ]+)\]', trace, re.DOTALL)
    for phase_num, phase_name, amt, energy, site_frac in phases:
        print(f"  Phase {phase_num} ({phase_name}):")
        print(f"    Amount: {amt}")
        print(f"    Energy: {energy}")
        print(f"    Site fractions: [{site_frac}]")

# Look for equilibrium solutions
print("\n=== EQUILIBRIUM SOLUTIONS ===")
cpu_eq_soln = re.findall(r'CPU.*?Equilibrium solution.*?: \[([\d.e+-, ]+)\]', content)
gpu_eq_soln = re.findall(r'GPU.*?Equilibrium solution at iteration (\d+).*?: \[([\d.e+-, ]+)\]', content)

print("\nCPU Equilibrium solutions:")
for i, soln in enumerate(cpu_eq_soln[:5]):  # First 5
    print(f"  Iteration {i}: [{soln}]")

print("\nGPU Equilibrium solutions:")  
for iter_num, soln in gpu_eq_soln[:5]:  # First 5
    print(f"  Iteration {iter_num}: [{soln}]")