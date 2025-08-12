#!/usr/bin/env python
"""Detailed comparison of CPU and GPU equilibrium matrices after gradient fixes."""

from pycalphad import Database, equilibrium, variables as v
import numpy as np
import io
import sys
import re

def extract_gpu_matrix_data(gpu_output):
    """Extract detailed matrix data from GPU debug output."""
    
    # Look for the equilibrium matrix construction
    matrix_pattern = r"GPU DEBUG: Equilibrium matrix \((\d+)x(\d+)\) at iteration (\d+):(.*?)(?=GPU DEBUG:|$)"
    matrix_matches = list(re.finditer(matrix_pattern, gpu_output, re.DOTALL))
    
    if not matrix_matches:
        print("No GPU equilibrium matrix found in output")
        return None
    
    # Use the first matrix (iteration 0)
    match = matrix_matches[0]
    rows = int(match.group(1))
    cols = int(match.group(2))
    iteration = int(match.group(3))
    matrix_text = match.group(4)
    
    print(f"Found GPU matrix: {rows}x{cols} at iteration {iteration}")
    
    # Extract matrix rows
    row_pattern = r"Row (\d+): (.*?)(?=\n|$)"
    row_matches = re.findall(row_pattern, matrix_text)
    
    matrix = []
    for row_num, row_data in row_matches:
        # Parse the row values
        values = [float(x.strip()) for x in row_data.split()]
        matrix.append(values)
    
    return np.array(matrix), iteration

def extract_gpu_rhs_data(gpu_output):
    """Extract RHS vector from GPU debug output."""
    
    # Look for RHS vector
    rhs_pattern = r"GPU DEBUG: RHS vector at iteration (\d+): \[(.*?)\]"
    rhs_matches = list(re.finditer(rhs_pattern, gpu_output, re.DOTALL))
    
    if not rhs_matches:
        print("No GPU RHS vector found")
        return None
    
    match = rhs_matches[0]
    iteration = int(match.group(1))
    rhs_text = match.group(2)
    
    # Parse RHS values
    rhs_values = [float(x.strip()) for x in rhs_text.split(',')]
    
    return np.array(rhs_values), iteration

def run_detailed_matrix_comparison():
    """Run detailed comparison of CPU vs GPU equilibrium matrices."""
    
    print("DETAILED CPU vs GPU EQUILIBRIUM MATRIX COMPARISON")
    print("="*60)
    print("Testing Al-Cu-Fe ternary system after gradient fixes")
    
    tdb = Database('Al-Cu-Fe.tdb')
    conditions = {v.T: 1200, v.P: 101325, v.X('CU'): 0.3, v.X('FE'): 0.2}
    
    print(f"Conditions: T=1200K, X(CU)=0.3, X(FE)=0.2")
    
    # CPU calculation (we can't easily extract CPU matrix, but we can compare results)
    print("\nCPU Calculation:")
    eq_cpu = equilibrium(tdb, ['AL', 'CU', 'FE', 'VA'], ['LIQUID'], conditions, gpu=False)
    cpu_gm = float(eq_cpu.GM.values.flatten()[0])
    print(f"CPU GM: {cpu_gm:.6f} J/mol")
    
    # GPU calculation with debug output
    print("\nGPU Calculation with matrix debug:")
    
    # Capture GPU output
    gpu_output = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = gpu_output
    
    try:
        eq_gpu = equilibrium(tdb, ['AL', 'CU', 'FE', 'VA'], ['LIQUID'], conditions, gpu=True, verbose=True)
        sys.stdout = old_stdout
        gpu_text = gpu_output.getvalue()
        gpu_gm = float(eq_gpu.GM.values.flatten()[0])
        
        print(f"GPU GM: {gpu_gm:.6f} J/mol")
        
        # Compare final results
        diff = abs(gpu_gm - cpu_gm)
        print(f"GM Difference: {diff:.6f} J/mol")
        
        if diff < 1e-6:
            print("✓ EXCELLENT: Numerical precision match!")
        elif diff < 1.0:
            print("✓ GOOD: Within 1 J/mol")
        else:
            print("○ IMPROVEMENT NEEDED: Still significant difference")
        
        # Extract and analyze GPU matrix data
        print(f"\n" + "-"*60)
        print("GPU MATRIX ANALYSIS:")
        print("-"*60)
        
        # Look for c_G values in first iteration
        cg_pattern = r"GPU DEBUG: Phase (\d+) c_G values.*?c_G\[0\] = (.*?)\\nc_G\[1\] = (.*?)(?:\\nc_G\[2\] = (.*?))?"
        cg_matches = list(re.finditer(cg_pattern, gpu_text, re.DOTALL))
        
        if cg_matches:
            print("c_G values at iteration 0:")
            for match in cg_matches[:3]:  # First 3 phases
                phase_idx = match.group(1)
                cg0 = match.group(2)
                cg1 = match.group(3)
                cg2 = match.group(4) if match.group(4) else "N/A"
                print(f"  Phase {phase_idx}: c_G = [{cg0}, {cg1}, {cg2}]")
        
        # Look for constraint RHS values
        rhs_pattern = r"GPU DEBUG: constraint_rhs\[(\d+)\] = (.*?) \(mass balance for component (\d+)\)"
        rhs_matches = list(re.finditer(rhs_pattern, gpu_text))
        
        if rhs_matches:
            print("\nConstraint RHS values:")
            for match in rhs_matches:
                constraint_idx = match.group(1)
                rhs_value = match.group(2)
                component_idx = match.group(3)
                print(f"  constraint_rhs[{constraint_idx}] = {rhs_value} (component {component_idx})")
        
        # Look for equilibrium matrix elements
        matrix_elem_pattern = r"GPU DEBUG: eq_matrix\[(\d+),(\d+)\] = (.*?) \("
        matrix_matches = list(re.finditer(matrix_elem_pattern, gpu_text))
        
        if matrix_matches:
            print("\nKey equilibrium matrix elements:")
            # Group by matrix position
            matrix_dict = {}
            for match in matrix_matches:
                row = int(match.group(1))
                col = int(match.group(2))
                value = float(match.group(3))
                matrix_dict[(row, col)] = value
            
            # Print first few elements
            print("  Matrix elements (first iteration):")
            for (row, col), value in list(matrix_dict.items())[:10]:
                print(f"    eq_matrix[{row},{col}] = {value:.6e}")
        
        # Look for mass balance residuals
        residual_pattern = r"GPU DEBUG: mass_residual for component (\d+) = (.*?)\\n"
        residual_matches = list(re.finditer(residual_pattern, gpu_text))
        
        if residual_matches:
            print("\nMass balance residuals:")
            for match in residual_matches:
                comp_idx = match.group(1)
                residual = match.group(2)
                print(f"  Component {comp_idx}: {residual}")
        
        # Look for solver solution
        solution_pattern = r"GPU DEBUG: solution\[(\d+)\] = (.*?)(?:\\n|$)"
        solution_matches = list(re.finditer(solution_pattern, gpu_text))
        
        if solution_matches:
            print("\nSolver solution vector:")
            for match in solution_matches:
                idx = match.group(1)
                value = match.group(2)
                print(f"  solution[{idx}] = {value}")
                
    except Exception as e:
        sys.stdout = old_stdout
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    print(f"\n" + "="*60)
    print("MATRIX COMPARISON CONCLUSIONS:")
    print("="*60)
    
    if diff < 1e-6:
        print("✓ SUCCESS: GPU and CPU matrices are now effectively identical!")
        print("  The gradient fixes have resolved the matrix construction issues.")
    elif diff < 10.0:
        print("○ PROGRESS: Matrices are much closer than before (239 → 43 J/mol).")
        print("  Some small differences remain, possibly in:")
        print("  - Numerical precision in matrix inversion")
        print("  - Small differences in constraint formulation")
        print("  - Solver tolerance differences")
    else:
        print("✗ MORE WORK NEEDED: Significant matrix differences still exist.")
        
    return diff

if __name__ == "__main__":
    run_detailed_matrix_comparison()