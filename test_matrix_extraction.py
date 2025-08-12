#!/usr/bin/env python
"""Extract and compare CPU and GPU equilibrium matrices at iteration 0."""

from pycalphad import Database, equilibrium, variables as v
import numpy as np
import io
import sys
import re
import os

def run_comparison():
    """Run both CPU and GPU and extract matrices."""
    
    tdb = Database('Al-Cu-Fe.tdb')
    conditions = {v.T: 1200, v.P: 101325, v.X('CU'): 0.3, v.X('FE'): 0.2}
    
    print("=" * 80)
    print("EQUILIBRIUM MATRIX COMPARISON - ITERATION 0")
    print("Al-Cu-Fe system: T=1200K, X(CU)=0.3, X(FE)=0.2")
    print("=" * 80)
    
    # Run CPU with DEBUG_MODE enabled
    print("\nCPU CALCULATION:")
    print("-" * 40)
    
    # Set DEBUG_MODE environment variable for CPU
    os.environ['DEBUG_MODE'] = '1'
    
    # Capture CPU output
    cpu_output = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = cpu_output
    
    eq_cpu = equilibrium(tdb, ['AL', 'CU', 'FE', 'VA'], ['LIQUID'], conditions, gpu=False, verbose=False)
    
    sys.stdout = old_stdout
    cpu_text = cpu_output.getvalue()
    cpu_gm = float(eq_cpu.GM.values.flatten()[0])
    
    # Extract CPU matrix from output
    cpu_matrix_pattern = r"\[CPU EQUILIBRIUM MATRIX\] Iteration 0.*?\n((?:.*?Row.*?\n)*)"
    cpu_match = re.search(cpu_matrix_pattern, cpu_text, re.MULTILINE | re.DOTALL)
    
    if cpu_match:
        print("CPU equilibrium matrix at iteration 0:")
        matrix_lines = cpu_match.group(1).strip().split('\n')
        for line in matrix_lines:
            print(line)
    else:
        print("Could not find CPU matrix in output")
        print("First 1000 chars of CPU output:")
        print(cpu_text[:1000])
    
    print(f"\nCPU GM: {cpu_gm:.8f} J/mol")
    
    # Run GPU with verbose output
    print("\nGPU CALCULATION:")
    print("-" * 40)
    
    gpu_output = io.StringIO()
    sys.stdout = gpu_output
    
    eq_gpu = equilibrium(tdb, ['AL', 'CU', 'FE', 'VA'], ['LIQUID'], conditions, gpu=True, verbose=True)
    
    sys.stdout = old_stdout
    gpu_text = gpu_output.getvalue()
    gpu_gm = float(eq_gpu.GM.values.flatten()[0])
    
    # Extract GPU matrix from output
    gpu_matrix_pattern = r"\[GPU EQUILIBRIUM MATRIX\] Iteration 0.*?\n((?:Row.*?\n)*)"
    gpu_match = re.search(gpu_matrix_pattern, gpu_text, re.MULTILINE | re.DOTALL)
    
    if gpu_match:
        print("GPU equilibrium matrix at iteration 0:")
        matrix_lines = gpu_match.group(1).strip().split('\n')
        for line in matrix_lines:
            print("  " + line)
    else:
        print("Could not find GPU matrix in output")
    
    print(f"\nGPU GM: {gpu_gm:.8f} J/mol")
    
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"CPU GM: {cpu_gm:.8f} J/mol")
    print(f"GPU GM: {gpu_gm:.8f} J/mol")
    print(f"Difference: {abs(cpu_gm - gpu_gm):.8f} J/mol")
    
    # Parse and compare matrix values
    if cpu_match and gpu_match:
        print("\nMATRIX VALUE COMPARISON:")
        print("-" * 40)
        
        # Parse CPU matrix
        cpu_rows = []
        for line in cpu_match.group(1).strip().split('\n'):
            if 'Row' in line:
                # Extract values between : and |
                values_part = line.split(':')[1].split('|')[0]
                values = [float(x.strip()) for x in values_part.split()]
                rhs = float(line.split('RHS:')[1].strip())
                cpu_rows.append((values, rhs))
        
        # Parse GPU matrix
        gpu_rows = []
        for line in gpu_match.group(1).strip().split('\n'):
            if 'Row' in line:
                values_part = line.split(':')[1].split('|')[0]
                values = [float(x.strip()) for x in values_part.split()]
                rhs = float(line.split('RHS:')[1].strip())
                gpu_rows.append((values, rhs))
        
        # Compare dimensions
        print(f"CPU matrix: {len(cpu_rows)} rows, {len(cpu_rows[0][0]) if cpu_rows else 0} cols")
        print(f"GPU matrix: {len(gpu_rows)} rows, {len(gpu_rows[0][0]) if gpu_rows else 0} cols")
        
        # Compare values
        if len(cpu_rows) == len(gpu_rows):
            max_diff = 0.0
            for i, (cpu_row, gpu_row) in enumerate(zip(cpu_rows, gpu_rows)):
                cpu_vals, cpu_rhs = cpu_row
                gpu_vals, gpu_rhs = gpu_row
                
                if len(cpu_vals) == len(gpu_vals):
                    for j, (cv, gv) in enumerate(zip(cpu_vals, gpu_vals)):
                        diff = abs(cv - gv)
                        if diff > max_diff:
                            max_diff = diff
                            print(f"  Max diff at [{i},{j}]: CPU={cv:.6e}, GPU={gv:.6e}, diff={diff:.6e}")
                
                rhs_diff = abs(cpu_rhs - gpu_rhs)
                if rhs_diff > 1e-10:
                    print(f"  RHS diff at row {i}: CPU={cpu_rhs:.6e}, GPU={gpu_rhs:.6e}, diff={rhs_diff:.6e}")
            
            print(f"\nMaximum matrix element difference: {max_diff:.6e}")
        else:
            print("\nMatrices have different dimensions - cannot compare values")

if __name__ == "__main__":
    run_comparison()