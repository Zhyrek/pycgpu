#!/usr/bin/env python3
"""Test GPU hessian with correct variable mapping"""

import numpy as np

# The issue is that the GPU generates hessian with model variables [T, Y_NB, Y_TI]
# but it's being called with workspace variables [N, P, T, Y_NB, Y_TI]

# Let's test what happens when we evaluate the GPU hessian with the wrong mapping

def gpu_hessian_element_33(x):
    """GPU's d²G/dY_NB² assuming x = [T, Y_NB, Y_TI]"""
    # This is from the GPU generated code (simplified)
    T = x[0]
    Y_NB = x[1] 
    Y_TI = x[2]
    
    # Just the main terms for element [1,1] which maps to [3,3] in workspace coords
    # From the generated code: out[3] which is d²G/dY_NB²
    result = -2.0 * (-8519.353 + 142.045475*T) / (Y_NB + Y_TI)**2
    result += 26090.6 * Y_TI / (Y_NB + Y_TI)
    # Additional terms...
    
    return result

def test_variable_mappings():
    """Test what happens with different variable mappings"""
    
    # Test conditions
    T = 1000.0
    Y_NB = 0.612245
    Y_TI = 0.387755
    
    # Model format: [T, Y_NB, Y_TI]
    model_dof = np.array([T, Y_NB, Y_TI])
    
    # Workspace format: [N, P, T, Y_NB, Y_TI]
    N = 1.0
    P = 101325.0
    workspace_dof = np.array([N, P, T, Y_NB, Y_TI])
    
    print("=== Variable Mapping Test ===")
    print(f"Model DOF: {model_dof}")
    print(f"Workspace DOF: {workspace_dof}")
    
    # If GPU function expects model format but gets workspace format:
    # It would read: T=N=1.0, Y_NB=P=101325, Y_TI=T=1000
    wrong_T = workspace_dof[0]  # 1.0 instead of 1000
    wrong_Y_NB = workspace_dof[1]  # 101325 instead of 0.612245
    wrong_Y_TI = workspace_dof[2]  # 1000 instead of 0.387755
    
    print(f"\nIf GPU expects model format but gets workspace format:")
    print(f"  T = {wrong_T} (should be {T})")
    print(f"  Y_NB = {wrong_Y_NB} (should be {Y_NB})")
    print(f"  Y_TI = {wrong_Y_TI} (should be {Y_TI})")
    print(f"  Y_NB + Y_TI = {wrong_Y_NB + wrong_Y_TI} (should be ~1.0)")
    
    # This would explain why the hessian is wrong!
    # The site fraction sum would be 101325 + 1000 = 102325 instead of 1.0
    # Terms with (Y_NB + Y_TI) in denominator would be ~102325x smaller
    # But wait, that would make the hessian smaller, not larger...
    
    # Actually, let me check the workspace format the GPU is using
    print("\n=== Checking GPU's actual usage ===")
    print("From debug output, GPU hessian is called with workspace DOF:")
    print("  hess_cols=5, workspace_dof values: [1.000000, 101325.000000, 1000.000000, 0.612245, 0.387755]")
    print("\nBut GPU code expects model format [T, Y_NB, Y_TI]!")
    print("So GPU reads: T=1.0, Y_NB=101325, Y_TI=1000")
    
    # Actually, I need to look more carefully at the generated code index mapping
    
test_variable_mappings()