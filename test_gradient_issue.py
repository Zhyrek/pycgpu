#!/usr/bin/env python
"""Test to identify the specific gradient mapping issue in ternary systems."""

from pycalphad import Database, equilibrium, variables as v
import numpy as np
import io
import sys
import re

def analyze_gradient_debug(output_text, system_name):
    """Analyze GPU gradient debug output."""
    print(f"\n{system_name} GRADIENT ANALYSIS:")
    print("-" * 40)
    
    # Look for formulagrad debug output
    grad_debug_pattern = r"\[GPU GRAD DEBUG\] Phase formulagrad results:.*?temp_grad: \[(.*?)\].*?mapped to grad\[2\]=(.*?), grad\[3\]=(.*?)(?:, grad\[4\]=(.*?))?.*?DOF values: T=(.*?), Y1=(.*?)(?:, Y2=(.*?))?"
    grad_matches = list(re.finditer(grad_debug_pattern, output_text, re.DOTALL))
    
    for i, match in enumerate(grad_matches[:2]):  # First 2 for brevity
        temp_grad = match.group(1)
        grad2 = match.group(2)
        grad3 = match.group(3) 
        grad4 = match.group(4) if match.group(4) else "N/A"
        T = match.group(5)
        Y1 = match.group(6)
        Y2 = match.group(7) if match.group(7) else "N/A"
        
        print(f"  Phase {i}:")
        print(f"    temp_grad: [{temp_grad}]")
        print(f"    mapped to: grad[2]={grad2}, grad[3]={grad3}, grad[4]={grad4}")
        print(f"    DOF values: T={T}, Y1={Y1}, Y2={Y2}")
        
        # Count number of gradient terms
        temp_grad_vals = temp_grad.split(', ')
        print(f"    Number of gradient terms: {len(temp_grad_vals)}")
        
    # Look for c_G calculation details
    cg_pattern = r"GPU DEBUG: Phase (\d+) c_G values.*?gradient values: \[(.*?)\]"
    cg_matches = list(re.finditer(cg_pattern, output_text, re.DOTALL))
    
    print(f"\n  c_G calculation gradients:")
    for match in cg_matches[:2]:  # First 2 phases
        phase_idx = match.group(1)
        gradients = match.group(2)
        grad_vals = gradients.split(', ')
        print(f"    Phase {phase_idx}: {len(grad_vals)} gradient values used in c_G")
        
    # Look for phase_dof values
    phase_dof_pattern = r"phase_dof = (\d+)"
    phase_dofs = list(set(re.findall(phase_dof_pattern, output_text)))
    print(f"  Phase DOF values: {phase_dofs}")
    
    # Look for num_statevars
    num_statevars_pattern = r"num_statevars=(\d+)"
    num_statevars = list(set(re.findall(num_statevars_pattern, output_text)))
    print(f"  num_statevars values: {num_statevars}")

# Test simplified binary system
print("BINARY SYSTEM TEST (Au-Bi):")
tdb_binary = Database('AuBi-07Wan.tdb')
eq_binary = equilibrium(tdb_binary, ['AU', 'BI', 'VA'], ['FCC_A1'], {
    v.T: 800, v.P: 101325, v.X('BI'): 0.3
}, gpu=True, verbose=False)

print("✓ Binary system works")

# Test simplified ternary system  
print("\nTERNARY SYSTEM TEST (Al-Cu-Fe):")
try:
    tdb_ternary = Database('Al-Cu-Fe.tdb')
    
    # Capture output to analyze gradient mapping
    gpu_output = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = gpu_output
    
    eq_ternary = equilibrium(tdb_ternary, ['AL', 'CU', 'FE', 'VA'], ['LIQUID'], {
        v.T: 1200, v.P: 101325, v.X('CU'): 0.3, v.X('FE'): 0.2  
    }, gpu=True, verbose=True)
    
    sys.stdout = old_stdout
    gpu_text = gpu_output.getvalue()
    
    print("✓ Ternary system completed")
    
    # Analyze the gradient mapping
    analyze_gradient_debug(gpu_text, "TERNARY")
    
except Exception as e:
    sys.stdout = old_stdout
    print(f"✗ Ternary system failed: {e}")

print("\n" + "="*60)
print("HYPOTHESIS VERIFICATION:")
print("-"*60)
print("Key Questions:")
print("1. Does binary formulagrad output 2 values [dG/dT, dG/dY1]?")
print("2. Does ternary formulagrad output 3 values [dG/dT, dG/dY1, dG/dY2]?") 
print("3. Are c_G calculations using the right number of gradient terms?")
print("4. Is the indexing grad[num_statevars + j] correct for both cases?")
print()
print("Expected findings:")
print("- Binary: phase_dof=1, uses grad[3] for c_G[0]")
print("- Ternary: phase_dof=2, uses grad[3], grad[4] for c_G[0], c_G[1]")
print("- If ternary formulagrad only outputs 2 values but c_G expects 3,")
print("  then grad[4] will be uninitialized/zero, causing wrong c_G!")