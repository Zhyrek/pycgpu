#!/usr/bin/env python
"""Print equilibrium matrices for CPU and GPU at first iteration."""

from pycalphad import Database, equilibrium, variables as v
import numpy as np
import io
import sys
import re

def extract_gpu_matrix(gpu_output):
    """Extract GPU matrix from debug output."""
    
    # Look for the complete equilibrium matrix
    matrix_pattern = r"GPU DEBUG: Complete equilibrium matrix \((\d+)x(\d+)\) at iteration (\d+):(.*?)(?=GPU DEBUG:|$)"
    matrix_matches = list(re.finditer(matrix_pattern, gpu_output, re.DOTALL))
    
    if matrix_matches:
        match = matrix_matches[0]
        rows = int(match.group(1))
        cols = int(match.group(2))
        iteration = int(match.group(3))
        matrix_text = match.group(4)
        
        print(f"GPU EQUILIBRIUM MATRIX ({rows}x{cols}) - Iteration {iteration}:")
        print("=" * 60)
        
        # Extract matrix rows
        row_pattern = r"Row (\d+): (.*?)(?=\n|$)"
        row_matches = re.findall(row_pattern, matrix_text)
        
        for row_num, row_data in row_matches:
            values = [float(x.strip()) for x in row_data.split()]
            row_str = " ".join([f"{val:12.6e}" for val in values])
            print(f"Row {row_num}: [{row_str}]")
        
        return True
    
    # Alternative pattern - look for individual matrix elements
    elem_pattern = r"GPU DEBUG: eq_matrix\[(\d+),(\d+)\] = (.*?) \("
    elem_matches = list(re.finditer(elem_pattern, gpu_output))
    
    if elem_matches:
        print("GPU EQUILIBRIUM MATRIX (from individual elements):")
        print("=" * 60)
        
        # Group by row
        matrix_dict = {}
        max_row = 0
        max_col = 0
        
        for match in elem_matches:
            row = int(match.group(1))
            col = int(match.group(2))
            value = float(match.group(3))
            matrix_dict[(row, col)] = value
            max_row = max(max_row, row)
            max_col = max(max_col, col)
        
        # Print matrix
        for row in range(max_row + 1):
            row_values = []
            for col in range(max_col + 1):
                if (row, col) in matrix_dict:
                    row_values.append(matrix_dict[(row, col)])
                else:
                    row_values.append(0.0)
            
            row_str = " ".join([f"{val:12.6e}" for val in row_values])
            print(f"Row {row}: [{row_str}]")
        
        return True
    
    return False

def extract_gpu_rhs(gpu_output):
    """Extract GPU RHS vector."""
    
    rhs_pattern = r"GPU DEBUG: RHS vector at iteration (\d+): \[(.*?)\]"
    rhs_matches = list(re.finditer(rhs_pattern, gpu_output, re.DOTALL))
    
    if rhs_matches:
        match = rhs_matches[0]
        iteration = int(match.group(1))
        rhs_text = match.group(2)
        
        print(f"\nGPU RHS VECTOR - Iteration {iteration}:")
        print("-" * 40)
        
        rhs_values = [float(x.strip()) for x in rhs_text.split(',')]
        for i, val in enumerate(rhs_values):
            print(f"RHS[{i}]: {val:12.6e}")
        
        return True
    
    return False

def print_matrices():
    """Print CPU and GPU equilibrium matrices."""
    
    tdb = Database('Al-Cu-Fe.tdb')
    conditions = {v.T: 1200, v.P: 101325, v.X('CU'): 0.3, v.X('FE'): 0.2}
    
    print("EQUILIBRIUM MATRIX COMPARISON")
    print("Al-Cu-Fe ternary system: T=1200K, X(CU)=0.3, X(FE)=0.2")
    print("=" * 80)
    
    # CPU - we can't easily extract the matrix, so just note the result
    print("\nCPU EQUILIBRIUM:")
    print("=" * 40)
    eq_cpu = equilibrium(tdb, ['AL', 'CU', 'FE', 'VA'], ['LIQUID'], conditions, gpu=False)
    cpu_gm = float(eq_cpu.GM.values.flatten()[0])
    print(f"CPU GM Result: {cpu_gm:.8f} J/mol")
    print("(CPU matrix not directly accessible)")
    
    print("\n" + "=" * 80)
    print("GPU EQUILIBRIUM MATRIX DEBUG:")
    print("=" * 80)
    
    # GPU with debug output
    gpu_output = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = gpu_output
    
    try:
        eq_gpu = equilibrium(tdb, ['AL', 'CU', 'FE', 'VA'], ['LIQUID'], conditions, gpu=True, verbose=True)
        sys.stdout = old_stdout
        gpu_text = gpu_output.getvalue()
        gpu_gm = float(eq_gpu.GM.values.flatten()[0])
        
        print(f"GPU GM Result: {gpu_gm:.8f} J/mol")
        print(f"Difference: {abs(cpu_gm - gpu_gm):.8f} J/mol")
        
        print("\n" + "-" * 80)
        
        # Extract and print GPU matrix
        if not extract_gpu_matrix(gpu_text):
            print("GPU matrix not found in debug output")
        
        # Extract and print GPU RHS
        if not extract_gpu_rhs(gpu_text):
            print("GPU RHS vector not found in debug output")
        
    except Exception as e:
        sys.stdout = old_stdout
        print(f"Error: {e}")

if __name__ == "__main__":
    print_matrices()