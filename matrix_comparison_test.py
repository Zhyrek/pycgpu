#!/usr/bin/env python
"""Extract and compare equilibrium matrices from CPU and GPU output."""

import subprocess
import sys
import re
import numpy as np

# Test script that prints matrices
test_script = '''
import sys
from pycalphad import Database, equilibrium

# Load TDB
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['LIQUID', 'BCC_A2']

# Single condition
conditions = {
    'T': 1000,
    'X_TI': 0.5
}

# Run with GPU flag from command line
gpu_flag = sys.argv[1] == 'gpu'
print(f"Running {'GPU' if gpu_flag else 'CPU'} solver...")

# Run equilibrium
eq_result = equilibrium(dbf, comps, phases, conditions, gpu=gpu_flag, verbose=True, 
                       calc_opts={'pdens': 100})

print(f"\\nFinal GM: {eq_result.GM.values[0]}")
print(f"Phase amounts: {eq_result.NP.values}")
'''

# Run both solvers and capture output
print("Running CPU solver...")
cpu_proc = subprocess.run([sys.executable, '-c', test_script, 'cpu'], 
                         capture_output=True, text=True)

print("Running GPU solver...")
gpu_proc = subprocess.run([sys.executable, '-c', test_script, 'gpu'], 
                         capture_output=True, text=True)

# Combine stdout and stderr
cpu_output = cpu_proc.stdout + cpu_proc.stderr
gpu_output = gpu_proc.stdout + gpu_proc.stderr

# Function to extract matrix from output
def extract_matrix(output, label):
    """Extract matrix data following a label."""
    lines = output.split('\n')
    matrix_lines = []
    in_matrix = False
    
    for i, line in enumerate(lines):
        if label in line and 'matrix' in line.lower():
            in_matrix = True
            continue
        
        if in_matrix:
            if 'Row' in line and ':' in line:
                # Extract numbers from row
                parts = line.split(':')
                if len(parts) > 1:
                    numbers = re.findall(r'[+-]?\d*\.?\d+[eE]?[+-]?\d*', parts[1])
                    if numbers:
                        matrix_lines.append([float(n) for n in numbers])
            elif not line.strip() or '[' in line:
                # End of matrix
                if matrix_lines:
                    break
    
    return matrix_lines

# Extract GPU matrix
print("\nExtracting GPU equilibrium matrix...")
gpu_matrix_lines = []
for line in gpu_output.split('\n'):
    if 'GPU MATRIX DEBUG' in line or 'Full equilibrium matrix' in line:
        print(f"Found: {line}")
    if line.startswith('[GPU]   Row') and ':' in line:
        # Parse the row
        parts = line.split(':', 1)
        if len(parts) > 1:
            numbers = re.findall(r'[+-]?\d*\.?\d+[eE]?[+-]?\d*', parts[1])
            if numbers:
                gpu_matrix_lines.append([float(n) for n in numbers])
                print(f"  Extracted row with {len(numbers)} values")

# Extract RHS vectors
print("\nExtracting RHS vectors...")
gpu_rhs = []
for line in gpu_output.split('\n'):
    if line.startswith('[GPU]   RHS[') and ':' in line:
        match = re.search(r'RHS\[\d+\]: ([+-]?\d*\.?\d+[eE]?[+-]?\d*)', line)
        if match:
            gpu_rhs.append(float(match.group(1)))

# Print findings
print(f"\nGPU Matrix shape: {len(gpu_matrix_lines)} x {len(gpu_matrix_lines[0]) if gpu_matrix_lines else 0}")
print(f"GPU RHS length: {len(gpu_rhs)}")

if gpu_matrix_lines:
    print("\nGPU Equilibrium Matrix:")
    for i, row in enumerate(gpu_matrix_lines):
        print(f"  Row {i}: {row}")
        
if gpu_rhs:
    print("\nGPU RHS Vector:")
    for i, val in enumerate(gpu_rhs):
        print(f"  RHS[{i}]: {val:+.15e}")

# Extract key numerical values
print("\n" + "="*60)
print("KEY NUMERICAL VALUES")
print("="*60)

# Final GM values
cpu_gm = re.search(r'Final GM: ([+-]?\d*\.?\d+[eE]?[+-]?\d*)', cpu_output)
gpu_gm = re.search(r'Final GM: ([+-]?\d*\.?\d+[eE]?[+-]?\d*)', gpu_output)

if cpu_gm and gpu_gm:
    cpu_gm_val = float(cpu_gm.group(1))
    gpu_gm_val = float(gpu_gm.group(1))
    print(f"CPU Final GM: {cpu_gm_val:.15e}")
    print(f"GPU Final GM: {gpu_gm_val:.15e}")
    print(f"Difference: {abs(cpu_gm_val - gpu_gm_val):.15e}")
    print(f"Relative error: {abs(cpu_gm_val - gpu_gm_val) / abs(cpu_gm_val) * 100:.2e}%")

# Check for specific debugging output patterns
print("\n" + "="*60)
print("CHECKING FOR PHASE INFORMATION")
print("="*60)

# Look for phase removal info in GPU output
for line in gpu_output.split('\n'):
    if 'phase removal' in line.lower() or 'removing phase' in line.lower():
        print(f"GPU: {line}")
    if 'phase_0' in line or 'phase_1' in line:
        if any(key in line for key in ['energy', 'composition', 'amount', 'NP']):
            print(f"GPU: {line}")