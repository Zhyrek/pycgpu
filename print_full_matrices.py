#!/usr/bin/env python
"""Print the complete equilibrium matrices for CPU and GPU at the first iteration."""

from pycalphad import Database, equilibrium, variables as v
import numpy as np
import io
import sys
import re

def extract_cpu_matrix():
    """Extract CPU matrix by patching numpy.linalg.solve."""
    
    # Patch numpy.linalg.solve to capture matrix
    import numpy.linalg
    original_solve = numpy.linalg.solve
    captured_matrix = []
    captured_rhs = []
    
    def patched_solve(a, b):
        nonlocal captured_matrix, captured_rhs
        # Only capture first call (iteration 0)
        if len(captured_matrix) == 0 and a.shape[0] <= 10:
            captured_matrix.append(np.copy(a))
            captured_rhs.append(np.copy(b))
            print("\n[CPU MATRIX CAPTURED] Equilibrium matrix at first solver call:")
            print(f"  Matrix shape: {a.shape}")
            for i in range(a.shape[0]):
                row_str = "  Row {}: ".format(i)
                for j in range(a.shape[1]):
                    row_str += "{:+.6e} ".format(a[i,j])
                row_str += "| RHS: {:+.6e}".format(b[i])
                print(row_str)
        return original_solve(a, b)
    
    numpy.linalg.solve = patched_solve
    
    # Run CPU calculation
    tdb = Database('Al-Cu-Fe.tdb')
    conditions = {v.T: 1200, v.P: 101325, v.X('CU'): 0.3, v.X('FE'): 0.2}
    
    eq_cpu = equilibrium(tdb, ['AL', 'CU', 'FE', 'VA'], ['LIQUID'], conditions, gpu=False, verbose=False)
    cpu_gm = float(eq_cpu.GM.values.flatten()[0])
    
    # Restore original solve
    numpy.linalg.solve = original_solve
    
    return cpu_gm, captured_matrix, captured_rhs

def extract_gpu_matrix():
    """Extract GPU matrix from debug output."""
    
    tdb = Database('Al-Cu-Fe.tdb')
    conditions = {v.T: 1200, v.P: 101325, v.X('CU'): 0.3, v.X('FE'): 0.2}
    
    # Capture GPU output
    gpu_output = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = gpu_output
    
    eq_gpu = equilibrium(tdb, ['AL', 'CU', 'FE', 'VA'], ['LIQUID'], conditions, gpu=True, verbose=True)
    sys.stdout = old_stdout
    gpu_text = gpu_output.getvalue()
    gpu_gm = float(eq_gpu.GM.values.flatten()[0])
    
    # Extract matrix from GPU debug output
    # Look for the complete matrix at iteration 0
    matrix_pattern = r"\[GPU EQUILIBRIUM MATRIX\] Complete matrix at iteration 0.*?\n((?:.*?\n)*?)(?=\[GPU|$)"
    matrix_match = re.search(matrix_pattern, gpu_text, re.MULTILINE | re.DOTALL)
    
    gpu_matrix = None
    gpu_rhs = None
    
    if matrix_match:
        matrix_text = matrix_match.group(0)
        print("\n[GPU MATRIX CAPTURED] Found equilibrium matrix in debug output")
        
        # Parse the matrix rows
        row_pattern = r"Row (\d+): (.*?)\| RHS: (.*?)(?:\n|$)"
        row_matches = re.findall(row_pattern, matrix_text)
        
        if row_matches:
            print(f"  Found {len(row_matches)} rows")
            for row_num, row_data, rhs in row_matches:
                values = [float(x.strip()) for x in row_data.split()]
                rhs_val = float(rhs.strip())
                row_str = "  Row {}: ".format(row_num)
                for val in values:
                    row_str += "{:+.6e} ".format(val)
                row_str += "| RHS: {:+.6e}".format(rhs_val)
                print(row_str)
    else:
        # Try alternative pattern for iteration 0
        alt_pattern = r"\[GPU EQUILIBRIUM MATRIX\] Iteration 0.*?\n((?:.*?\n)*?)system_amount"
        alt_match = re.search(alt_pattern, gpu_text, re.MULTILINE | re.DOTALL)
        
        if alt_match:
            matrix_text = alt_match.group(0)
            print("\n[GPU MATRIX CAPTURED] Found equilibrium matrix (alternative pattern)")
            
            # Parse the matrix rows
            row_pattern = r"Row (\d+): (.*?)\| RHS: (.*?)(?:\n|$)"
            row_matches = re.findall(row_pattern, matrix_text)
            
            if row_matches:
                print(f"  Found {len(row_matches)} rows")
                for row_num, row_data, rhs in row_matches:
                    values = [float(x.strip()) for x in row_data.split()]
                    rhs_val = float(rhs.strip())
                    row_str = "  Row {}: ".format(row_num)
                    for val in values:
                        row_str += "{:+.6e} ".format(val)
                    row_str += "| RHS: {:+.6e}".format(rhs_val)
                    print(row_str)
        else:
            print("\n[GPU MATRIX] Could not find matrix in debug output")
            print("First 2000 chars of output:")
            print(gpu_text[:2000])
    
    return gpu_gm

def main():
    print("=" * 80)
    print("EQUILIBRIUM MATRIX COMPARISON - FIRST ITERATION")
    print("Al-Cu-Fe system: T=1200K, X(CU)=0.3, X(FE)=0.2")
    print("=" * 80)
    
    # Get CPU matrix
    print("\nExtracting CPU matrix...")
    cpu_gm, cpu_matrix, cpu_rhs = extract_cpu_matrix()
    print(f"\nCPU GM: {cpu_gm:.8f} J/mol")
    
    # Get GPU matrix  
    print("\nExtracting GPU matrix...")
    gpu_gm = extract_gpu_matrix()
    print(f"\nGPU GM: {gpu_gm:.8f} J/mol")
    
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"CPU GM: {cpu_gm:.8f} J/mol")
    print(f"GPU GM: {gpu_gm:.8f} J/mol")
    print(f"Difference: {abs(cpu_gm - gpu_gm):.8f} J/mol")

if __name__ == "__main__":
    main()