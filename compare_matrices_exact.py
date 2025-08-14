#!/usr/bin/env python
"""Extract and compare CPU and GPU equilibrium matrices exactly."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['PYCALPHAD_DEBUG_MODE'] = '1'

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
import subprocess
import re
warnings.filterwarnings("ignore")

def extract_matrix_from_output(output, pattern):
    """Extract matrix and RHS from debug output."""
    lines = output.split('\n')
    matrix_data = []
    rhs_data = []
    
    for i, line in enumerate(lines):
        if pattern in line:
            # Found the matrix output, parse the dimensions
            match = re.search(r'rows=(\d+), cols=(\d+)', line)
            if match:
                rows = int(match.group(1))
                cols = int(match.group(2))
                
                # Parse the next 'rows' lines for matrix data
                for j in range(1, rows + 1):
                    if i + j < len(lines):
                        row_line = lines[i + j]
                        # Extract matrix values and RHS
                        if 'Row' in row_line and 'RHS:' in row_line:
                            # Split at RHS
                            parts = row_line.split('| RHS:')
                            if len(parts) == 2:
                                # Extract matrix values
                                matrix_part = parts[0].split(':')[1] if ':' in parts[0] else parts[0]
                                values = re.findall(r'[+-]?\d+\.?\d*e[+-]?\d+', matrix_part)
                                matrix_data.append([float(v) for v in values])
                                
                                # Extract RHS value
                                rhs_val = re.findall(r'[+-]?\d+\.?\d*e[+-]?\d+', parts[1])
                                if rhs_val:
                                    rhs_data.append(float(rhs_val[0]))
                
                if matrix_data:
                    return np.array(matrix_data), np.array(rhs_data)
    
    return None, None

def run_equilibrium_capture_output(dbf, comps, phases, conditions, gpu=False):
    """Run equilibrium and capture stdout/stderr."""
    import io
    from contextlib import redirect_stdout, redirect_stderr
    
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
        result = equilibrium(dbf, comps, phases, conditions, gpu=gpu, verbose=True)
    
    return stdout_capture.getvalue() + stderr_capture.getvalue(), result

def main():
    dbf = Database('Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['LIQUID']
    
    conditions = {
        v.X('AL'): 0.25,
        v.X('CU'): 0.25,
        v.T: 2000,
        v.P: 101325
    }
    
    print("=" * 80)
    print("EXACT MATRIX COMPARISON")
    print("=" * 80)
    print(f"Phases: {phases}")
    print(f"Prescribed: X(AL)={conditions[v.X('AL')]:.3f}, X(CU)={conditions[v.X('CU')]:.3f}")
    print()
    
    # Run CPU 
    print("Running CPU calculation...")
    cpu_output, cpu_result = run_equilibrium_capture_output(dbf, comps, phases, conditions, gpu=False)
    
    # Run GPU
    print("Running GPU calculation...")
    gpu_output, gpu_result = run_equilibrium_capture_output(dbf, comps, phases, conditions, gpu=True)
    
    # Extract matrices from iteration 0
    print("\n" + "=" * 80)
    print("ITERATION 0 COMPARISON")
    print("=" * 80)
    
    cpu_matrix, cpu_rhs = extract_matrix_from_output(cpu_output, "[EQUILIBRIUM_MATRIX_OUTPUT] CPU Iteration 0")
    gpu_matrix, gpu_rhs = extract_matrix_from_output(gpu_output, "[EQUILIBRIUM_MATRIX_OUTPUT] GPU Iteration 0")
    
    if cpu_matrix is not None and gpu_matrix is not None:
        print(f"\nCPU Matrix shape: {cpu_matrix.shape}")
        print(f"GPU Matrix shape: {gpu_matrix.shape}")
        
        if cpu_matrix.shape == gpu_matrix.shape:
            print("\n--- Matrix Element Comparison ---")
            max_diff = 0.0
            for i in range(cpu_matrix.shape[0]):
                for j in range(cpu_matrix.shape[1]):
                    diff = abs(cpu_matrix[i,j] - gpu_matrix[i,j])
                    if diff > 1e-15:  # Only show non-zero differences
                        print(f"  [{i},{j}]: CPU={cpu_matrix[i,j]:+.9e}, GPU={gpu_matrix[i,j]:+.9e}, diff={diff:.2e}")
                    max_diff = max(max_diff, diff)
            
            print(f"\nMax matrix element difference: {max_diff:.2e}")
            
            print("\n--- RHS Comparison ---")
            max_rhs_diff = 0.0
            for i in range(len(cpu_rhs)):
                diff = abs(cpu_rhs[i] - gpu_rhs[i])
                if diff > 1e-10:  # Only show significant differences
                    print(f"  Row {i}: CPU={cpu_rhs[i]:+.9e}, GPU={gpu_rhs[i]:+.9e}, diff={diff:.2e}")
                max_rhs_diff = max(max_rhs_diff, diff)
            
            print(f"\nMax RHS difference: {max_rhs_diff:.2e}")
            
            # Check if matrices are effectively identical
            if max_diff < 1e-10 and max_rhs_diff < 1e-10:
                print("\n✓ Matrices are IDENTICAL (within numerical tolerance)")
            else:
                print("\n✗ Matrices DIFFER")
                
            # Now check what the solution should be
            print("\n" + "=" * 80)
            print("SOLVING THE SYSTEM")
            print("=" * 80)
            
            # Solve both systems with numpy to see what we should get
            print("\nSolving CPU matrix with numpy.linalg.lstsq:")
            cpu_soln, residuals, rank, s = np.linalg.lstsq(cpu_matrix, cpu_rhs, rcond=1e-16)
            print(f"  Solution: {cpu_soln}")
            print(f"  Rank: {rank}/{min(cpu_matrix.shape)}")
            print(f"  Singular values: {s}")
            
            print("\nSolving GPU matrix with numpy.linalg.lstsq:")
            gpu_soln, residuals, rank, s = np.linalg.lstsq(gpu_matrix, gpu_rhs, rcond=1e-16)
            print(f"  Solution: {gpu_soln}")
            print(f"  Rank: {rank}/{min(gpu_matrix.shape)}")
            print(f"  Singular values: {s}")
            
            print("\nDifference in solutions:")
            for i in range(len(cpu_soln)):
                print(f"  x[{i}]: {cpu_soln[i] - gpu_soln[i]:+.2e}")
                
        else:
            print("ERROR: Matrix dimensions don't match!")
    else:
        print("ERROR: Could not extract matrices from output")
    
    # Also extract the actual solutions reported by CPU/GPU
    print("\n" + "=" * 80)
    print("ACTUAL SOLVER OUTPUTS")
    print("=" * 80)
    
    # Look for equilibrium solution in output
    cpu_soln_match = re.search(r'equilibrium_soln\[0\]:\s*([-\d.e+]+)', cpu_output)
    gpu_soln_match = re.search(r'Equilibrium solution.*?:\s*\[([-\d.e+, ]+)\]', gpu_output)
    
    if cpu_soln_match:
        print(f"\nCPU reported first solution element: {cpu_soln_match.group(1)}")
    if gpu_soln_match:
        gpu_vals = [float(x.strip()) for x in gpu_soln_match.group(1).split(',')]
        print(f"GPU reported solution: {gpu_vals}")

if __name__ == "__main__":
    main()