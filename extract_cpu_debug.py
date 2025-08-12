#!/usr/bin/env python
"""Extract CPU matrix by adding debug prints to the source."""

import os
import subprocess

def check_cpu_solver_source():
    """Check if we can find the CPU solver source files."""
    
    print("SEARCHING FOR CPU SOLVER SOURCE")
    print("="*40)
    
    # Try to find eqsolver.pyx
    try:
        import pycalphad
        pycalphad_path = os.path.dirname(pycalphad.__file__)
        print(f"Pycalphad installed at: {pycalphad_path}")
        
        # Look for source files
        for root, dirs, files in os.walk(pycalphad_path):
            for file in files:
                if 'eqsolver' in file and ('.pyx' in file or '.py' in file):
                    full_path = os.path.join(root, file)
                    print(f"Found: {full_path}")
                    
                    # Try to read a few lines
                    try:
                        with open(full_path, 'r') as f:
                            lines = f.readlines()[:10]
                        print(f"  First few lines: {len(lines)} lines")
                        for i, line in enumerate(lines[:3]):
                            print(f"  {i+1}: {line.strip()}")
                    except:
                        print("  (Could not read file)")
        
    except Exception as e:
        print(f"Error: {e}")

def try_direct_import():
    """Try to directly import and inspect the solver."""
    
    print("\nTRYING DIRECT SOLVER IMPORT")
    print("="*40)
    
    try:
        # Import the solver directly
        from pycalphad.core import eqsolver
        
        print(f"Eqsolver module: {eqsolver}")
        print(f"Eqsolver file: {eqsolver.__file__ if hasattr(eqsolver, '__file__') else 'No file attr'}")
        
        # Look for solve functions
        functions = [attr for attr in dir(eqsolver) if 'solve' in attr.lower()]
        print(f"Solve functions: {functions}")
        
        # Try to inspect solve_eq_at_conditions
        if hasattr(eqsolver, 'solve_eq_at_conditions'):
            func = eqsolver.solve_eq_at_conditions
            print(f"solve_eq_at_conditions: {func}")
            print(f"Function type: {type(func)}")
            
            # Check if it has source
            import inspect
            try:
                source = inspect.getsource(func)
                print(f"Source length: {len(source)} characters")
                print("First few lines:")
                for i, line in enumerate(source.split('\n')[:5]):
                    print(f"  {i+1}: {line}")
            except:
                print("Cannot get source (likely Cython compiled)")
        
    except Exception as e:
        print(f"Error: {e}")

def manual_cpu_matrix_calculation():
    """Manually calculate what the CPU matrix should be."""
    
    print("\nMANUAL CPU MATRIX ESTIMATION")
    print("="*40)
    
    print("Based on GPU debug output, we can estimate CPU matrix structure:")
    print("For Al-Cu-Fe LIQUID at T=1200K, X(CU)=0.3, X(FE)=0.2")
    print()
    
    print("The equilibrium matrix should have the general form:")
    print("  [H11  H12  H13  1 ]")
    print("  [H21  H22  H23  1 ]") 
    print("  [H31  H32  H33  1 ]")
    print("  [ 1    1    1   0 ]")
    print()
    
    print("Where H_ij are Hessian elements of the Gibbs energy")
    print("The last row/column represent site fraction constraints")
    print()
    
    print("From GPU output, we saw Hessian values around:")
    print("  H11 ~ 2.8e4,  H12 ~ -4.7e4, H13 ~ -6.8e4")
    print("  H21 ~ -4.7e4, H22 ~ 1.6e4,  H23 ~ 2.4e4") 
    print("  H31 ~ -6.8e4, H32 ~ 2.4e4,  H33 ~ 4.6e4")
    print()
    
    print("If CPU and GPU matrices were identical, GM difference would be ~0")
    print("262.7 J/mol difference suggests CPU matrix has different values")
    print()
    
    print("Possible CPU matrix differences:")
    print("1. Different Hessian calculation (numerical vs analytical)")
    print("2. Different constraint formulation")
    print("3. Different phase fraction weighting")
    print("4. Different chemical potential initialization")

if __name__ == "__main__":
    check_cpu_solver_source()
    try_direct_import() 
    manual_cpu_matrix_calculation()