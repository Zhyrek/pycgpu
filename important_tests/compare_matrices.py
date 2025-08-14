#!/usr/bin/env python
"""Compare CPU and GPU equilibrium matrices for a failing condition."""

import sys
import os
import subprocess
import re
import numpy as np

def extract_matrix_from_output(output, matrix_type="CPU"):
    """Extract matrix values from the output."""
    lines = output.split('\n')
    matrix_rows = []
    rhs_values = []
    
    for line in lines:
        if f"[EQUILIBRIUM_MATRIX_OUTPUT] {matrix_type} Iteration 0" in line:
            # Extract dimensions
            match = re.search(r'rows=(\d+), cols=(\d+)', line)
            if match:
                rows = int(match.group(1))
                cols = int(match.group(2))
        elif "Row" in line and "RHS:" in line:
            # Parse the row
            parts = line.split('|')
            if len(parts) == 2:
                # Extract matrix values
                matrix_part = parts[0].split(':')[1].strip()
                values = [float(x) for x in matrix_part.split()]
                matrix_rows.append(values)
                
                # Extract RHS value
                rhs_part = parts[1].strip()
                rhs_val = float(rhs_part.split(':')[1].strip())
                rhs_values.append(rhs_val)
    
    if matrix_rows:
        return np.array(matrix_rows), np.array(rhs_values)
    return None, None

def main():
    """Run the comparison for the failing condition."""
    
    # Run the print_equilibrium_matrix script
    cmd = [
        'python', 'print_equilibrium_matrix.py',
        '../Al-Cu-Fe.tdb',
        'AL,CU,FE,VA',
        'LIQUID,FCC_A1,BCC_A2,BCC_B2,L12,ALCU_THETA,AL13FE4,AL5FE2',
        '--T=900', '--X_AL=0.2', '--X_CU=0.5'
    ]
    
    print("Running equilibrium matrix comparison for X(AL)=0.2, X(CU)=0.5, T=900K")
    print("=" * 80)
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd='/mnt/c/users/scott/Documents/pycalphad/important_tests')
    output = result.stdout
    
    # Extract CPU and GPU matrices
    cpu_matrix, cpu_rhs = extract_matrix_from_output(output, "CPU")
    gpu_matrix, gpu_rhs = extract_matrix_from_output(output, "GPU")
    
    if cpu_matrix is None or gpu_matrix is None:
        print("Failed to extract matrices from output")
        print("\nRaw output:")
        print(output[:2000])
        return
    
    print(f"Matrix dimensions: {cpu_matrix.shape}")
    print()
    
    # Compare matrices element by element
    print("Matrix comparison (absolute differences):")
    print("-" * 80)
    
    matrix_diff = np.abs(gpu_matrix - cpu_matrix)
    max_diff = np.max(matrix_diff)
    
    print("Matrix differences:")
    for i in range(matrix_diff.shape[0]):
        for j in range(matrix_diff.shape[1]):
            if matrix_diff[i, j] > 1e-15:
                print(f"  Element [{i},{j}]: CPU={cpu_matrix[i,j]:+.6e}, GPU={gpu_matrix[i,j]:+.6e}, Diff={matrix_diff[i,j]:.6e}")
    
    if max_diff < 1e-10:
        print("  ✓ All matrix elements match to within 1e-10")
    else:
        print(f"  ✗ Maximum matrix difference: {max_diff:.6e}")
    
    print("\nRHS comparison:")
    rhs_diff = np.abs(gpu_rhs - cpu_rhs)
    max_rhs_diff = np.max(rhs_diff)
    
    for i in range(len(rhs_diff)):
        if rhs_diff[i] > 1e-15:
            print(f"  RHS[{i}]: CPU={cpu_rhs[i]:+.6e}, GPU={gpu_rhs[i]:+.6e}, Diff={rhs_diff[i]:.6e}")
    
    if max_rhs_diff < 1e-10:
        print("  ✓ All RHS values match to within 1e-10")
    else:
        print(f"  ✗ Maximum RHS difference: {max_rhs_diff:.6e}")
    
    print("\n" + "=" * 80)
    print("SUMMARY:")
    
    if max_diff < 1e-10 and max_rhs_diff < 1e-10:
        print("✓ INITIAL SOLVER STATE IS IN PERFECT AGREEMENT")
        print("  The equilibrium matrix and RHS at iteration 0 are identical.")
        print("  This confirms the divergence happens during solver iterations,")
        print("  not in the initial setup.")
    else:
        print("✗ INITIAL SOLVER STATE HAS DIFFERENCES")
        print(f"  Maximum matrix difference: {max_diff:.6e}")
        print(f"  Maximum RHS difference: {max_rhs_diff:.6e}")
    
    # Also check for numerical precision issues
    print("\nNumerical analysis:")
    print(f"  Matrix condition number: {np.linalg.cond(cpu_matrix):.2e}")
    
    # Check for near-singular matrix
    try:
        cpu_det = np.linalg.det(cpu_matrix)
        gpu_det = np.linalg.det(gpu_matrix)
        print(f"  CPU matrix determinant: {cpu_det:.6e}")
        print(f"  GPU matrix determinant: {gpu_det:.6e}")
        
        if abs(cpu_det) < 1e-10:
            print("  ⚠ WARNING: Matrix is nearly singular!")
    except:
        pass
    
    print("=" * 80)

if __name__ == "__main__":
    main()