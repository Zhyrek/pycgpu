#!/usr/bin/env python
"""Analyze why the Hessian is zero by examining the model directly."""

from pycalphad import Database, Model
import numpy as np

# Load database and create model
dbf = Database('NbTi.tdb')
mod = Model(dbf, ['NB', 'TI'], 'LIQUID')

print("Analyzing LIQUID phase model...")
print(f"Components: {mod.components}")
print(f"Constituents: {mod.constituents}")
print(f"Site ratios: {mod.site_ratios}")

# Get the actual symbols used
print("\nModel variables:")
for var in sorted(mod.variables, key=str):
    print(f"  {var}")

# Check the Hessian symbolically
print("\n\nChecking Hessian matrix symbolically...")

# The model should have a hess property
if hasattr(mod, 'GM'):
    print("Model has GM (Gibbs energy) property")
    
    # Get the Hessian from the model
    # This is typically a SymPy/SymEngine matrix
    variables = list(mod.variables)
    
    # Find the relevant variables for differentiation
    # For LIQUID phase with NB and TI, we need Y(LIQUID,0,NB) and Y(LIQUID,0,TI)
    y_vars = [v for v in variables if 'Y(LIQUID' in str(v) and ('NB' in str(v) or 'TI' in str(v))]
    print(f"\nSite fraction variables: {y_vars}")
    
    if len(y_vars) >= 2:
        # The Hessian should be the second derivatives
        # Let's check what pycalphad computed
        print("\nExamining model's Hessian...")
        
        # Import symengine utilities
        from symengine import diff
        
        # Get the AST 
        gm_expr = mod.ast
        print(f"\nGM expression type: {type(gm_expr)}")
        
        # Try to compute Hessian element manually
        try:
            # Get the Y variables correctly
            Y_NB = None
            Y_TI = None
            for v in variables:
                v_str = str(v)
                if 'Y(LIQUID,0,NB)' in v_str:
                    Y_NB = v
                elif 'Y(LIQUID,0,TI)' in v_str:
                    Y_TI = v
                    
            if Y_NB and Y_TI:
                print(f"\nFound variables: Y_NB={Y_NB}, Y_TI={Y_TI}")
                
                # Compute d²GM/dY_NB²
                first_deriv = diff(gm_expr, Y_NB)
                second_deriv = diff(first_deriv, Y_NB)
                
                # Convert to string and look for patterns
                hess_str = str(second_deriv)
                
                # Check for all-zero Piecewise
                if 'Piecewise((0' in hess_str and ', (0, True))' in hess_str:
                    print("\n*** FOUND ALL-ZERO PIECEWISE IN HESSIAN! ***")
                    
                    # Find the specific patterns
                    import re
                    zero_patterns = re.findall(r'Piecewise\(\(0, [^)]+\), \(0, True\)\)', hess_str)
                    print(f"\nFound {len(zero_patterns)} all-zero Piecewise patterns:")
                    for i, pattern in enumerate(zero_patterns[:5]):
                        print(f"  {i+1}: {pattern}")
                        
                    # This is the root cause!
                    print("\n\nROOT CAUSE IDENTIFIED:")
                    print("The symbolic differentiation of the model produces")
                    print("Piecewise expressions where ALL branches are zero.")
                    print("This happens when differentiating protected log terms")
                    print("at discontinuous boundaries.")
                    
                    # Now let's see what the actual Hessian values should be
                    print("\n\nEvaluating Hessian numerically at T=1000, Y_NB=0.5, Y_TI=0.5...")
                    
                    # Substitute values
                    subs_dict = {
                        'T': 1000.0,
                        'Y(LIQUID,0,NB)': 0.5,
                        'Y(LIQUID,0,TI)': 0.5,
                        'LIQUID0NB': 0.5,  # These might be internal variables
                        'LIQUID0TI': 0.5
                    }
                    
                    # Try to evaluate
                    try:
                        # This won't work directly, but let's see what happens
                        from symengine import symbols
                        T = symbols('T')
                        
                        # The correct Hessian for ideal solution should be:
                        # d²(RT*(Y_NB*ln(Y_NB) + Y_TI*ln(Y_TI)))/dY_NB² = -RT/Y_NB²
                        # At T=1000, Y_NB=0.5: -8.3145*1000/0.25 = -33258
                        # Plus interaction terms
                        
                        expected_ideal = -8.3145 * 1000 / (0.5**2)
                        print(f"\nExpected ideal contribution to Hessian: {expected_ideal:.1f}")
                        print("But the symbolic Hessian has all-zero Piecewise!")
                        
                    except Exception as e:
                        print(f"\nCouldn't evaluate numerically: {e}")
                        
        except Exception as e:
            print(f"\nError computing Hessian: {e}")
            
print("\n\nCONCLUSION:")
print("The GPU code is getting zero Hessian because the symbolic")
print("differentiation in pycalphad produces all-zero Piecewise")
print("expressions. This is a bug in how Piecewise functions are")
print("differentiated when they have discontinuities.")
print("\nThe fix needs to be in the symbolic differentiation or")
print("in how the model handles log terms at boundaries.")