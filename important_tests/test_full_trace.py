#!/usr/bin/env python
"""Comprehensive trace comparison between CPU and GPU for failing condition."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import subprocess
import tempfile

def capture_debug_output(dbf, comps, phases, conditions, gpu=False):
    """Capture full debug output from equilibrium calculation."""
    
    # Create a test script that will capture stdout
    test_code = f"""
import sys
import os
sys.path.insert(0, '{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}')

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

# Enable maximum debugging
os.environ['PYCALPHAD_DEBUG_MODE'] = '1'

dbf = Database('{os.path.abspath('../Al-Cu-Fe.tdb')}')
comps = {comps}
phases = {phases}
conditions = {{
    v.X('AL'): {conditions[v.X('AL')]},
    v.X('CU'): {conditions[v.X('CU')]},
    v.T: {conditions[v.T]},
    v.P: {conditions[v.P]}
}}

result = equilibrium(dbf, comps, phases, conditions,
                    calc_opts={{'pdens': 50}},
                    gpu={gpu}, verbose=True)

print("\\nFINAL GM VALUE: {{:.15e}}".format(result.GM.values.item()))
print("FINAL NP VALUES:", result.NP.values.flatten())
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_code)
        temp_file = f.name
    
    try:
        # Run the script and capture output
        result = subprocess.run([sys.executable, temp_file], 
                              capture_output=True, text=True, timeout=30)
        output = result.stdout + result.stderr
    finally:
        os.unlink(temp_file)
    
    return output

def parse_matrix_output(output):
    """Parse equilibrium matrix from debug output."""
    matrices = []
    lines = output.split('\n')
    
    i = 0
    while i < len(lines):
        if 'EQUILIBRIUM_MATRIX_OUTPUT' in lines[i] or '[CPU EQUILIBRIUM MATRIX]' in lines[i]:
            # Found a matrix
            matrix_data = {'iteration': None, 'rows': [], 'rhs': []}
            
            # Extract iteration number
            if 'Iteration' in lines[i]:
                parts = lines[i].split('Iteration')
                if len(parts) > 1:
                    iter_str = parts[1].split()[0]
                    matrix_data['iteration'] = int(iter_str)
            
            # Read matrix rows
            i += 1
            while i < len(lines) and 'Row' in lines[i]:
                row_line = lines[i]
                # Parse the row values and RHS
                if '|' in row_line and 'RHS:' in row_line:
                    parts = row_line.split('|')
                    if len(parts) >= 2:
                        row_part = parts[0].split(':')[1] if ':' in parts[0] else parts[0]
                        rhs_part = parts[1].split('RHS:')[1] if 'RHS:' in parts[1] else parts[1]
                        
                        # Extract numbers from row
                        row_values = []
                        for val in row_part.split():
                            try:
                                row_values.append(float(val))
                            except:
                                pass
                        
                        # Extract RHS value
                        try:
                            rhs_value = float(rhs_part.strip())
                        except:
                            rhs_value = None
                        
                        if row_values:
                            matrix_data['rows'].append(row_values)
                            matrix_data['rhs'].append(rhs_value)
                
                i += 1
            
            if matrix_data['rows']:
                matrices.append(matrix_data)
        else:
            i += 1
    
    return matrices

def compare_values(cpu_val, gpu_val, name, tolerance=1e-10):
    """Compare two values and report differences."""
    if cpu_val is None or gpu_val is None:
        if cpu_val != gpu_val:
            print(f"  ✗ {name}: CPU={cpu_val}, GPU={gpu_val} (one is None)")
            return False
        return True
    
    try:
        cpu_float = float(cpu_val)
        gpu_float = float(gpu_val)
        diff = abs(cpu_float - gpu_float)
        rel_diff = diff / max(abs(cpu_float), abs(gpu_float), 1e-15)
        
        if diff > tolerance:
            print(f"  ✗ {name}: CPU={cpu_float:.15e}, GPU={gpu_float:.15e}, Diff={diff:.15e}, RelDiff={rel_diff:.15e}")
            return False
        elif diff > 0:
            print(f"  ~ {name}: CPU={cpu_float:.15e}, GPU={gpu_float:.15e}, Diff={diff:.15e}")
        return True
    except:
        if cpu_val != gpu_val:
            print(f"  ✗ {name}: CPU={cpu_val}, GPU={gpu_val}")
            return False
        return True

def main():
    """Full trace comparison of failing condition."""
    
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2']
    
    # The exact failing condition
    conditions = {
        v.X('AL'): 0.2,
        v.X('CU'): 0.5,
        v.T: 900,
        v.P: 101325
    }
    
    print("=" * 100)
    print("COMPREHENSIVE CPU vs GPU TRACE COMPARISON")
    print("=" * 100)
    print(f"\nCondition: X(AL)={conditions[v.X('AL')]}, X(CU)={conditions[v.X('CU')]}, T={conditions[v.T]}K")
    print(f"Phases: {phases}")
    print("\nCapturing debug output...")
    
    # Capture full debug output
    print("\nRunning CPU calculation with full debug...")
    cpu_output = capture_debug_output(dbf, comps, phases, conditions, gpu=False)
    
    print("Running GPU calculation with full debug...")
    gpu_output = capture_debug_output(dbf, comps, phases, conditions, gpu=True)
    
    # Save raw outputs for inspection
    with open('cpu_debug_output.txt', 'w') as f:
        f.write(cpu_output)
    with open('gpu_debug_output.txt', 'w') as f:
        f.write(gpu_output)
    
    print("\nRaw debug outputs saved to cpu_debug_output.txt and gpu_debug_output.txt")
    
    # Parse matrices
    print("\n" + "=" * 100)
    print("PARSING EQUILIBRIUM MATRICES")
    print("=" * 100)
    
    cpu_matrices = parse_matrix_output(cpu_output)
    gpu_matrices = parse_matrix_output(gpu_output)
    
    print(f"\nFound {len(cpu_matrices)} CPU matrices and {len(gpu_matrices)} GPU matrices")
    
    # Compare each matrix iteration
    print("\n" + "=" * 100)
    print("MATRIX-BY-MATRIX COMPARISON")
    print("=" * 100)
    
    max_iterations = max(len(cpu_matrices), len(gpu_matrices))
    
    for iter_idx in range(max_iterations):
        print(f"\n{'='*50}")
        print(f"ITERATION {iter_idx}")
        print(f"{'='*50}")
        
        if iter_idx >= len(cpu_matrices):
            print("  ✗ CPU has no matrix for this iteration (terminated early)")
            continue
        if iter_idx >= len(gpu_matrices):
            print("  ✗ GPU has no matrix for this iteration (terminated early)")
            continue
        
        cpu_mat = cpu_matrices[iter_idx]
        gpu_mat = gpu_matrices[iter_idx]
        
        # Compare dimensions
        cpu_rows = len(cpu_mat['rows'])
        cpu_cols = len(cpu_mat['rows'][0]) if cpu_mat['rows'] else 0
        gpu_rows = len(gpu_mat['rows'])
        gpu_cols = len(gpu_mat['rows'][0]) if gpu_mat['rows'] else 0
        
        print(f"\nMatrix dimensions: CPU={cpu_rows}x{cpu_cols}, GPU={gpu_rows}x{gpu_cols}")
        
        if cpu_rows != gpu_rows or cpu_cols != gpu_cols:
            print("  ✗ DIMENSION MISMATCH!")
            continue
        
        # Compare each element
        print("\nComparing matrix elements:")
        all_match = True
        
        for row_idx in range(cpu_rows):
            print(f"\n  Row {row_idx}:")
            cpu_row = cpu_mat['rows'][row_idx]
            gpu_row = gpu_mat['rows'][row_idx]
            
            for col_idx in range(min(len(cpu_row), len(gpu_row))):
                if not compare_values(cpu_row[col_idx], gpu_row[col_idx], 
                                    f"    [{row_idx},{col_idx}]", tolerance=1e-12):
                    all_match = False
            
            # Compare RHS
            if not compare_values(cpu_mat['rhs'][row_idx], gpu_mat['rhs'][row_idx],
                                f"    RHS[{row_idx}]", tolerance=1e-12):
                all_match = False
        
        if all_match:
            print(f"\n  ✓ All values match for iteration {iter_idx}")
    
    # Extract and compare other values from output
    print("\n" + "=" * 100)
    print("COMPARING OTHER DEBUG VALUES")
    print("=" * 100)
    
    # Look for phase amounts, chemical potentials, etc.
    cpu_lines = cpu_output.split('\n')
    gpu_lines = gpu_output.split('\n')
    
    # Extract GM values
    print("\n" + "-" * 50)
    print("FINAL GM VALUES:")
    print("-" * 50)
    
    cpu_gm = None
    gpu_gm = None
    
    for line in cpu_lines:
        if 'FINAL GM VALUE:' in line:
            cpu_gm = float(line.split(':')[1].strip())
            
    for line in gpu_lines:
        if 'FINAL GM VALUE:' in line:
            gpu_gm = float(line.split(':')[1].strip())
    
    if cpu_gm and gpu_gm:
        compare_values(cpu_gm, gpu_gm, "Final GM", tolerance=1e-6)
    
    # Extract phase amounts
    print("\n" + "-" * 50)
    print("FINAL PHASE AMOUNTS:")
    print("-" * 50)
    
    cpu_np = None
    gpu_np = None
    
    for line in cpu_lines:
        if 'FINAL NP VALUES:' in line:
            np_str = line.split(':', 1)[1].strip()
            # Parse the numpy array string
            np_str = np_str.replace('[', '').replace(']', '')
            cpu_np = [float(x) for x in np_str.split() if x and x != 'nan']
    
    for line in gpu_lines:
        if 'FINAL NP VALUES:' in line:
            np_str = line.split(':', 1)[1].strip()
            np_str = np_str.replace('[', '').replace(']', '')
            gpu_np = [float(x) for x in np_str.split() if x and x != 'nan']
    
    if cpu_np and gpu_np:
        for i, (cpu_val, gpu_val) in enumerate(zip(cpu_np, gpu_np)):
            compare_values(cpu_val, gpu_val, f"NP[{i}] ({phases[i] if i < len(phases) else 'unknown'})")
    
    # Look for any chemical potential values
    print("\n" + "-" * 50)
    print("CHEMICAL POTENTIALS (if found):")
    print("-" * 50)
    
    for line in cpu_lines:
        if 'chemical_potential' in line.lower() or 'mu[' in line.lower():
            print(f"CPU: {line.strip()}")
    
    for line in gpu_lines:
        if 'chemical_potential' in line.lower() or 'mu[' in line.lower():
            print(f"GPU: {line.strip()}")
    
    # Look for convergence information
    print("\n" + "-" * 50)
    print("CONVERGENCE INFORMATION:")
    print("-" * 50)
    
    for line in cpu_lines:
        if 'converge' in line.lower() or 'residual' in line.lower():
            if 'CPU' not in line:  # Avoid reprinting our own messages
                print(f"CPU: {line.strip()}")
    
    for line in gpu_lines:
        if 'converge' in line.lower() or 'residual' in line.lower():
            if 'GPU' not in line and '[GPU]' not in line:
                print(f"GPU: {line.strip()}")
    
    print("\n" + "=" * 100)
    print("TRACE COMPARISON COMPLETE")
    print("Check cpu_debug_output.txt and gpu_debug_output.txt for full raw output")
    print("=" * 100)

if __name__ == "__main__":
    main()