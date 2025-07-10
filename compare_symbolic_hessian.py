#!/usr/bin/env python3
"""Compare symbolic expressions for hessian between CPU and GPU approaches"""

import sys
sys.path.insert(0, '.')

from pycalphad import Database, Model
import pycalphad.variables as v

# Load database and create model
db = Database('NbTi.tdb')
model = Model(db, ['NB', 'TI', 'VA'], 'BCC_A2')

print("=== Model Information ===")
print(f"Model state variables: {model.state_variables}")
print(f"Model site fractions: {model.site_fractions}")
print(f"Number of site fractions: {len(model.site_fractions)}")

# Get the G expression
print("\n=== G Expression (first 1000 chars) ===")
G_str = str(model.G)
print(G_str[:1000])

# Check if sympy/symengine is available
try:
    from symengine import sympify, Symbol
    print("\n=== Using SymEngine for symbolic math ===")
    
    # Get the energy expression
    G_expr = model.G
    
    # Get the variables to differentiate with respect to
    # For hessian, we need all variables (state vars + site fractions)
    T = model.state_variables[0]  # Temperature
    Y_NB = model.site_fractions[0]  # BCC_A20NB
    Y_TI = model.site_fractions[1]  # BCC_A20TI
    
    print(f"\nDifferentiating with respect to: {[T, Y_NB, Y_TI]}")
    
    # Compute first derivatives (gradient)
    print("\n=== First Derivatives (Gradient) ===")
    dG_dT = G_expr.diff(T)
    dG_dY_NB = G_expr.diff(Y_NB)
    dG_dY_TI = G_expr.diff(Y_TI)
    
    print(f"dG/dT (first 500 chars): {str(dG_dT)[:500]}")
    print(f"\ndG/dY_NB (first 500 chars): {str(dG_dY_NB)[:500]}")
    print(f"\ndG/dY_TI (first 500 chars): {str(dG_dY_TI)[:500]}")
    
    # Compute second derivatives (hessian elements we care about)
    print("\n=== Second Derivatives (Hessian) ===")
    print("Computing d²G/dY_NB² and d²G/dY_NB∂Y_TI...")
    
    d2G_dY_NB2 = dG_dY_NB.diff(Y_NB)
    d2G_dY_NB_dY_TI = dG_dY_NB.diff(Y_TI)
    d2G_dY_TI2 = dG_dY_TI.diff(Y_TI)
    
    print(f"\nd²G/dY_NB² (first 800 chars): {str(d2G_dY_NB2)[:800]}")
    print(f"\nd²G/dY_NB∂Y_TI (first 800 chars): {str(d2G_dY_NB_dY_TI)[:800]}")
    print(f"\nd²G/dY_TI² (first 800 chars): {str(d2G_dY_TI2)[:800]}")
    
    # Check for the (Y_NB + Y_TI) factor
    print("\n=== Checking for Site Fraction Sum Factor ===")
    sum_factor = Y_NB + Y_TI
    print(f"Sum factor in G: {sum_factor in G_expr.args if hasattr(G_expr, 'args') else 'Cannot check'}")
    
    # Try to evaluate at specific values to compare magnitudes
    print("\n=== Numerical Evaluation at Test Point ===")
    test_vals = {T: 1000.0, Y_NB: 0.6, Y_TI: 0.4}
    
    try:
        G_val = float(G_expr.subs(test_vals))
        d2G_dY_NB2_val = float(d2G_dY_NB2.subs(test_vals))
        d2G_dY_NB_dY_TI_val = float(d2G_dY_NB_dY_TI.subs(test_vals))
        
        print(f"At T=1000K, Y_NB=0.6, Y_TI=0.4:")
        print(f"  G = {G_val:.6e}")
        print(f"  d²G/dY_NB² = {d2G_dY_NB2_val:.6e}")
        print(f"  d²G/dY_NB∂Y_TI = {d2G_dY_NB_dY_TI_val:.6e}")
    except Exception as e:
        print(f"Could not evaluate numerically: {e}")
        
except ImportError:
    print("\n=== SymEngine not available ===")
    print("Cannot perform symbolic differentiation analysis")
    
# Now check what the GPU code generation would do
print("\n=== GPU Code Generation Approach ===")
from pycalphad.gpu.gpu_codegen import _nb_formulahess_from_model

# This is what the GPU uses
gpu_hess_code = _nb_formulahess_from_model(model, 0, None, validate=False, verbose=True)
print("\nGPU generated hessian code (first 1500 chars):")
print(gpu_hess_code[:1500])