#!/usr/bin/env python
"""Extract CPU matrix by patching numpy/scipy."""

from pycalphad import Database, equilibrium, variables as v
import numpy as np

def patch_linear_algebra():
    """Patch linear algebra to capture matrix operations."""
    
    # Patch numpy.linalg.solve
    import numpy.linalg
    original_solve = numpy.linalg.solve
    
    def patched_solve(a, b):
        print(f"\n=== CPU MATRIX SOLVE ===")
        print(f"Matrix shape: {a.shape}")
        if a.shape[0] <= 8:
            print("Matrix A:")
            for i, row in enumerate(a):
                row_str = " ".join([f"{val:12.6e}" for val in row])
                print(f"  Row {i}: [{row_str}]")
            print("RHS B:")
            for i, val in enumerate(b):
                print(f"  B[{i}]: {val:12.6e}")
        
        result = original_solve(a, b)
        
        if len(result) <= 8:
            print("Solution X:")
            for i, val in enumerate(result):
                print(f"  X[{i}]: {val:12.6e}")
        print("=== END MATRIX SOLVE ===\n")
        
        return result
    
    numpy.linalg.solve = patched_solve
    
    # Also patch scipy if available
    try:
        import scipy.linalg
        original_scipy_solve = scipy.linalg.solve
        
        def patched_scipy_solve(a, b, **kwargs):
            print(f"\n=== CPU SCIPY MATRIX SOLVE ===")
            print(f"Matrix shape: {a.shape}")
            if a.shape[0] <= 8:
                print("Matrix A:")
                for i, row in enumerate(a):
                    row_str = " ".join([f"{val:12.6e}" for val in row])
                    print(f"  Row {i}: [{row_str}]")
                print("RHS B:")
                for i, val in enumerate(b):
                    print(f"  B[{i}]: {val:12.6e}")
            
            result = original_scipy_solve(a, b, **kwargs)
            
            if len(result) <= 8:
                print("Solution X:")
                for i, val in enumerate(result):
                    print(f"  X[{i}]: {val:12.6e}")
            print("=== END SCIPY SOLVE ===\n")
            
            return result
        
        scipy.linalg.solve = patched_scipy_solve
        
    except ImportError:
        pass

def get_cpu_matrix():
    """Get CPU equilibrium matrix values."""
    
    print("EXTRACTING CPU EQUILIBRIUM MATRIX")
    print("="*50)
    
    # Apply patches
    patch_linear_algebra()
    
    # Run CPU calculation
    tdb = Database('Al-Cu-Fe.tdb')
    conditions = {v.T: 1200, v.P: 101325, v.X('CU'): 0.3, v.X('FE'): 0.2}
    
    print("Running CPU equilibrium...")
    eq_cpu = equilibrium(tdb, ['AL', 'CU', 'FE', 'VA'], ['LIQUID'], conditions, gpu=False)
    cpu_gm = float(eq_cpu.GM.values.flatten()[0])
    
    print(f"CPU GM: {cpu_gm:.8f} J/mol")

if __name__ == "__main__":
    get_cpu_matrix()