#!/usr/bin/env python
"""Capture ALL deltas and values from CPU and GPU at divergence iteration."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import sys
import io
import re

print("CAPTURING ALL VALUES AND DELTAS AT DIVERGENCE ITERATION")
print("=" * 80)

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

# Capture ALL output including debug prints
class OutputCapture:
    def __init__(self):
        self.output = []
        self.original_stdout = sys.stdout
        
    def write(self, text):
        self.output.append(text)
        self.original_stdout.write(text)
        
    def flush(self):
        self.original_stdout.flush()
        
    def get_output(self):
        return ''.join(self.output)

# Run CPU with capture
print("\n" + "="*80)
print("CPU CALCULATION - CAPTURING ALL OUTPUT")
print("="*80)
cpu_capture = OutputCapture()
sys.stdout = cpu_capture

try:
    result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=True)
except:
    pass

sys.stdout = cpu_capture.original_stdout
cpu_output = cpu_capture.get_output()

# Run GPU with capture  
print("\n" + "="*80)
print("GPU CALCULATION - CAPTURING ALL OUTPUT")
print("="*80)
gpu_capture = OutputCapture()
sys.stdout = gpu_capture

try:
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)
except:
    pass

sys.stdout = gpu_capture.original_stdout
gpu_output = gpu_capture.get_output()

# Parse and print ALL values at iteration 0 and 1
print("\n" + "="*80)
print("EXTRACTED VALUES - ITERATION 0")
print("="*80)

print("\nCPU ITERATION 0:")
# Find all numerical values after "iteration 0" in CPU output
cpu_iter0_section = re.search(r'iteration 0(.*?)iteration 1', cpu_output, re.DOTALL)
if cpu_iter0_section:
    # Extract all delta values
    cpu_deltas = re.findall(r'delta[^=]*=\s*([-\d.e+]+)', cpu_iter0_section.group(1))
    print("  All deltas found:")
    for i, delta in enumerate(cpu_deltas):
        print(f"    delta[{i}] = {delta}")
    
    # Extract all phase amounts
    cpu_phase_amts = re.findall(r'phase_amt[^=]*=\s*([-\d.e+]+)', cpu_iter0_section.group(1))
    print("  All phase amounts:")
    for i, amt in enumerate(cpu_phase_amts):
        print(f"    phase_amt[{i}] = {amt}")
        
    # Extract all X values
    cpu_x_values = re.findall(r'X[^=]*=\s*\[([-\d.e+, ]+)\]', cpu_iter0_section.group(1))
    print("  All X compositions:")
    for i, x in enumerate(cpu_x_values):
        print(f"    X[{i}] = [{x}]")
        
    # Extract c_G values
    cpu_cg_values = re.findall(r'c_G[^=]*=\s*\[([-\d.e+, ]+)\]', cpu_iter0_section.group(1))
    print("  All c_G values:")
    for i, cg in enumerate(cpu_cg_values):
        print(f"    c_G[{i}] = [{cg}]")

print("\nGPU ITERATION 0:")
# Find all numerical values after "iteration 0" in GPU output
gpu_iter0_section = re.search(r'Iteration 0(.*?)Iteration 1', gpu_output, re.DOTALL)
if gpu_iter0_section:
    # Extract all delta values
    gpu_deltas = re.findall(r'delta[^=]*=\s*([-\d.e+]+)', gpu_iter0_section.group(1))
    print("  All deltas found:")
    for i, delta in enumerate(gpu_deltas):
        print(f"    delta[{i}] = {delta}")
        
    # Extract phase amounts
    gpu_phase_amts = re.findall(r'phase_amt[^=]*=\s*([-\d.e+]+)', gpu_iter0_section.group(1))
    print("  All phase amounts:")
    for i, amt in enumerate(gpu_phase_amts):
        print(f"    phase_amt[{i}] = {amt}")
        
    # Extract all NP values
    gpu_np_values = re.findall(r'NP[^=]*=\s*([-\d.e+]+)', gpu_iter0_section.group(1))
    print("  All NP values:")
    for i, np_val in enumerate(gpu_np_values):
        print(f"    NP[{i}] = {np_val}")
        
    # Extract c_G values
    gpu_cg_values = re.findall(r'c_G\[(\d+)\]\s*=\s*([-\d.e+]+)', gpu_iter0_section.group(1))
    print("  All c_G values:")
    for idx, val in gpu_cg_values:
        print(f"    c_G[{idx}] = {val}")

print("\n" + "="*80)
print("EXTRACTED VALUES - ITERATION 1")  
print("="*80)

print("\nCPU ITERATION 1:")
# Find all numerical values after "iteration 1" in CPU output
cpu_iter1_section = re.search(r'iteration 1(.*?)iteration 2', cpu_output, re.DOTALL)
if cpu_iter1_section:
    # Extract all delta values
    cpu_deltas = re.findall(r'delta[^=]*=\s*([-\d.e+]+)', cpu_iter1_section.group(1))
    print("  All deltas found:")
    for i, delta in enumerate(cpu_deltas):
        print(f"    delta[{i}] = {delta}")
        
    # Extract solution vector
    cpu_soln = re.findall(r'equilibrium_soln\[(.*?)\]', cpu_iter1_section.group(1))
    print("  Solution vector elements:")
    for elem in cpu_soln:
        print(f"    equilibrium_soln[{elem}]")
        
    # Extract RHS values
    cpu_rhs = re.findall(r'RHS[^=]*=\s*([-\d.e+]+)', cpu_iter1_section.group(1))
    print("  All RHS values:")
    for i, rhs in enumerate(cpu_rhs):
        print(f"    RHS[{i}] = {rhs}")

print("\nGPU ITERATION 1:")
# Find all numerical values after "iteration 1" in GPU output  
gpu_iter1_section = re.search(r'Iteration 1(.*?)(?:Iteration 2|converged)', gpu_output, re.DOTALL)
if gpu_iter1_section:
    # Extract all delta values
    gpu_deltas = re.findall(r'delta[^=]*=\s*([-\d.e+]+)', gpu_iter1_section.group(1))
    print("  All deltas found:")
    for i, delta in enumerate(gpu_deltas):
        print(f"    delta[{i}] = {delta}")
        
    # Extract solution values
    gpu_soln = re.findall(r'x\[(\d+)\]\s*=\s*([-\d.e+]+)', gpu_iter1_section.group(1))
    print("  Solution vector:")
    for idx, val in gpu_soln:
        print(f"    x[{idx}] = {val}")
        
    # Extract singular values
    gpu_svd = re.findall(r'singular_values\[(\d+)\]\s*=\s*([-\d.e+]+)', gpu_iter1_section.group(1))
    print("  Singular values:")
    for idx, val in gpu_svd:
        print(f"    singular_values[{idx}] = {val}")