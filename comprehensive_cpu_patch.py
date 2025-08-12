#!/usr/bin/env python
"""More comprehensive CPU matrix extraction."""

from pycalphad import Database, equilibrium, variables as v
import numpy as np

def patch_all_linear_algebra():
    """Patch every possible linear algebra function."""
    
    # Patch numpy
    import numpy.linalg
    original_funcs = {}
    
    for func_name in ['solve', 'lstsq', 'inv', 'pinv']:
        if hasattr(numpy.linalg, func_name):
            original_funcs[f'numpy_{func_name}'] = getattr(numpy.linalg, func_name)
            
            def make_patch(name, original):
                def patched(*args, **kwargs):
                    print(f"\n=== {name} CALLED ===")
                    if len(args) > 0 and hasattr(args[0], 'shape'):
                        print(f"First arg shape: {args[0].shape}")
                        if args[0].shape[0] <= 10:
                            print(f"Matrix A ({name}):")
                            for i, row in enumerate(args[0]):
                                row_str = " ".join([f"{val:12.6e}" for val in row])
                                print(f"  Row {i}: [{row_str}]")
                    
                    result = original(*args, **kwargs)
                    print(f"=== {name} COMPLETE ===\n")
                    return result
                return patched
            
            setattr(numpy.linalg, func_name, make_patch(f"numpy.linalg.{func_name}", original_funcs[f'numpy_{func_name}']))
    
    # Patch scipy if available
    try:
        import scipy.linalg
        
        for func_name in ['solve', 'lstsq', 'inv', 'pinv', 'lu_solve', 'cho_solve']:
            if hasattr(scipy.linalg, func_name):
                original_funcs[f'scipy_{func_name}'] = getattr(scipy.linalg, func_name)
                
                def make_scipy_patch(name, original):
                    def patched(*args, **kwargs):
                        print(f"\n=== {name} CALLED ===")
                        if len(args) > 0 and hasattr(args[0], 'shape'):
                            print(f"First arg shape: {args[0].shape}")
                            if args[0].shape[0] <= 10:
                                print(f"Matrix A ({name}):")
                                for i, row in enumerate(args[0]):
                                    row_str = " ".join([f"{val:12.6e}" for val in row])
                                    print(f"  Row {i}: [{row_str}]")
                        
                        result = original(*args, **kwargs)
                        print(f"=== {name} COMPLETE ===\n")
                        return result
                    return patched
                
                setattr(scipy.linalg, func_name, make_scipy_patch(f"scipy.linalg.{func_name}", original_funcs[f'scipy_{func_name}']))
        
    except ImportError:
        print("SciPy not available")
    
    # Try to patch BLAS/LAPACK calls
    try:
        import scipy.linalg.lapack
        
        for func_name in ['dgesv', 'dgels', 'dgelss']:
            if hasattr(scipy.linalg.lapack, func_name):
                original_funcs[f'lapack_{func_name}'] = getattr(scipy.linalg.lapack, func_name)
                
                def make_lapack_patch(name, original):
                    def patched(*args, **kwargs):
                        print(f"\n=== LAPACK {name} CALLED ===")
                        result = original(*args, **kwargs)
                        print(f"=== LAPACK {name} COMPLETE ===\n")
                        return result
                    return patched
                
                setattr(scipy.linalg.lapack, func_name, make_lapack_patch(f"lapack.{func_name}", original_funcs[f'lapack_{func_name}']))
                
    except ImportError:
        print("LAPACK not available")

def patch_solver_class():
    """Try to patch the Solver class."""
    
    try:
        from pycalphad.core.solver import Solver
        
        # Save original methods
        if hasattr(Solver, 'solve'):
            original_solve = Solver.solve
            
            def patched_solver_solve(self, *args, **kwargs):
                print("\n=== SOLVER.SOLVE CALLED ===")
                print(f"Solver type: {type(self)}")
                print(f"Args: {len(args)}")
                
                result = original_solve(self, *args, **kwargs)
                
                print("=== SOLVER.SOLVE COMPLETE ===\n")
                return result
            
            Solver.solve = patched_solver_solve
            print("Patched Solver.solve")
            
    except Exception as e:
        print(f"Could not patch Solver: {e}")

def run_comprehensive_extraction():
    """Run comprehensive matrix extraction."""
    
    print("COMPREHENSIVE CPU MATRIX EXTRACTION")
    print("="*50)
    
    # Apply all patches
    patch_all_linear_algebra()
    patch_solver_class()
    
    print("All patches applied. Running CPU equilibrium...")
    print("-" * 50)
    
    # Run CPU calculation
    tdb = Database('Al-Cu-Fe.tdb')
    conditions = {v.T: 1200, v.P: 101325, v.X('CU'): 0.3, v.X('FE'): 0.2}
    
    eq_cpu = equilibrium(tdb, ['AL', 'CU', 'FE', 'VA'], ['LIQUID'], conditions, gpu=False, verbose=True)
    cpu_gm = float(eq_cpu.GM.values.flatten()[0])
    
    print(f"\nCPU GM: {cpu_gm:.8f} J/mol")

if __name__ == "__main__":
    run_comprehensive_extraction()