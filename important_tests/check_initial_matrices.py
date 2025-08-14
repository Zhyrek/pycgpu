#!/usr/bin/env python
"""Check if initial equilibrium matrices match between CPU and GPU."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
import re

def get_matrix_from_output(output):
    """Extract equilibrium matrix from debug output."""
    lines = output.split('\n')
    
    # Look for iteration 0 matrix
    matrix_lines = []
    in_matrix = False
    
    for line in lines:
        if 'Iteration 0' in line and 'EQUILIBRIUM_MATRIX' in line:
            in_matrix = True
            continue
        if in_matrix:
            if 'Row' in line:
                # Extract the row values
                match = re.search(r'Row \d+: ([-+\d.e ]+) \| RHS: ([-+\d.e]+)', line)
                if match:
                    row_vals = match.group(1).strip()
                    rhs_val = match.group(2).strip()
                    matrix_lines.append((row_vals, rhs_val))
            elif matrix_lines and 'Row' not in line:
                # End of matrix
                break
    
    return matrix_lines

def compare_matrices(cpu_matrix, gpu_matrix):
    """Compare CPU and GPU matrices."""
    if len(cpu_matrix) != len(gpu_matrix):
        return False, f"Different sizes: CPU={len(cpu_matrix)}, GPU={len(gpu_matrix)}"
    
    max_diff = 0.0
    diff_location = ""
    
    for i, (cpu_row, gpu_row) in enumerate(zip(cpu_matrix, gpu_matrix)):
        cpu_vals, cpu_rhs = cpu_row
        gpu_vals, gpu_rhs = gpu_row
        
        # Compare row values
        cpu_numbers = [float(x) for x in cpu_vals.split()]
        gpu_numbers = [float(x) for x in gpu_vals.split()]
        
        for j, (cv, gv) in enumerate(zip(cpu_numbers, gpu_numbers)):
            diff = abs(cv - gv)
            if diff > max_diff:
                max_diff = diff
                diff_location = f"Row {i}, Col {j}"
        
        # Compare RHS
        cpu_rhs_val = float(cpu_rhs)
        gpu_rhs_val = float(gpu_rhs)
        diff = abs(cpu_rhs_val - gpu_rhs_val)
        if diff > max_diff:
            max_diff = diff
            diff_location = f"Row {i}, RHS"
    
    if max_diff < 1e-10:
        return True, "Matrices match"
    else:
        return False, f"Max diff={max_diff:.3e} at {diff_location}"

def test_condition(x_al, x_cu, temp, desc):
    """Test a single condition and compare matrices."""
    
    # Create test script
    test_code = f"""
import sys
import os
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')
os.environ['PYCALPHAD_DEBUG_MODE'] = '1'

from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings('ignore')

dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']
phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2']

conditions = {{
    v.X('AL'): {x_al},
    v.X('CU'): {x_cu},
    v.T: {temp},
    v.P: 101325
}}

# CPU calculation
print("=== CPU MATRIX ===")
result = equilibrium(dbf, comps, phases, conditions,
                    calc_opts={{'pdens': 50}},
                    gpu=False, verbose=False)
print(f"CPU GM: {{result.GM.values.item():.2f}}")

# GPU calculation  
print("=== GPU MATRIX ===")
result = equilibrium(dbf, comps, phases, conditions,
                    calc_opts={{'pdens': 50}},
                    gpu=True, verbose=True)
print(f"GPU GM: {{result.GM.values.item():.2f}}")
"""
    
    # Run and capture output
    try:
        result = subprocess.run(
            [sys.executable, '-c', test_code],
            capture_output=True,
            text=True,
            timeout=30,
            cwd='/mnt/c/users/scott/Documents/pycalphad'
        )
        
        output = result.stdout + result.stderr
        
        # Extract matrices
        cpu_section = output[output.find("=== CPU MATRIX ==="):output.find("=== GPU MATRIX ===")]
        gpu_section = output[output.find("=== GPU MATRIX ==="):]
        
        cpu_matrix = get_matrix_from_output(cpu_section)
        gpu_matrix = get_matrix_from_output(gpu_section)
        
        # Compare
        if cpu_matrix and gpu_matrix:
            match, msg = compare_matrices(cpu_matrix, gpu_matrix)
            return match, msg, len(cpu_matrix)
        else:
            return None, "Could not extract matrices", 0
            
    except Exception as e:
        return None, str(e), 0

def main():
    """Check initial matrices for all failing conditions."""
    
    print("=" * 80)
    print("INITIAL EQUILIBRIUM MATRIX COMPARISON")
    print("=" * 80)
    
    # Test the known failing conditions
    test_cases = [
        (0.10, 0.50, 900, "Low Al, High Cu"),
        (0.20, 0.50, 900, "Med Al, High Cu ***"),
        (0.30, 0.50, 900, "High Al, High Cu"),
        (0.40, 0.40, 600, "Equal Al-Cu at 600K"),
        (0.50, 0.40, 600, "High Al at 600K"),
        (0.70, 0.20, 900, "Very High Al"),
        (0.20, 0.40, 1200, "High temp case 1"),
        (0.30, 0.60, 1200, "High temp case 2"),
        (0.70, 0.10, 1200, "High temp case 3"),
    ]
    
    print("\nChecking if initial matrices (iteration 0) match between CPU and GPU:")
    print("-" * 80)
    print("Condition                    | Matrix Size | Match? | Details")
    print("-----------------------------|-------------|--------|------------------")
    
    for x_al, x_cu, temp, desc in test_cases:
        x_fe = 1.0 - x_al - x_cu
        
        match, msg, size = test_condition(x_al, x_cu, temp, desc)
        
        if match is None:
            status = "ERROR"
            details = msg[:30]
        elif match:
            status = "✓ YES"
            details = "Identical"
        else:
            status = "✗ NO"
            details = msg
        
        print(f"{desc:28s} | {size:11d} | {status:6s} | {details}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
If all initial matrices match:
- The divergence occurs during solver iterations, not setup
- Both solvers start from identical conditions
- Numerical instabilities during iteration cause different convergence

If some don't match:
- There may be differences in how constraints are set up
- Or differences in initial phase selection
""")

if __name__ == "__main__":
    main()