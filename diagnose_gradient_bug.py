#!/usr/bin/env python
"""Diagnose the exact gradient mapping bug in ternary systems."""

from pycalphad import Database, equilibrium, variables as v
import numpy as np
import io
import sys
import re

def extract_gradient_data(output_text):
    """Extract formulagrad and c_G data from GPU output."""
    
    # Find formulagrad output
    grad_pattern = r"\[GPU GRAD DEBUG\] Phase formulagrad results:.*?temp_grad: \[(.*?)\].*?mapped to grad\[2\]=(.*?), grad\[3\]=(.*?)(?:, grad\[4\]=(.*?))?.*?DOF values: T=(.*?), Y1=(.*?)(?:, Y2=(.*?))?"
    grad_matches = list(re.finditer(grad_pattern, output_text, re.DOTALL))
    
    print("FORMULAGRAD DEBUG OUTPUT:")
    print("-" * 30)
    
    for i, match in enumerate(grad_matches):
        temp_grad_str = match.group(1)
        grad2 = match.group(2)
        grad3 = match.group(3)
        grad4 = match.group(4) if match.group(4) else "N/A"
        T = match.group(5)
        Y1 = match.group(6)
        Y2 = match.group(7) if match.group(7) else "N/A"
        
        temp_grad_values = [x.strip() for x in temp_grad_str.split(',')]
        
        print(f"Phase {i}:")
        print(f"  temp_grad has {len(temp_grad_values)} values: {temp_grad_values}")
        print(f"  Mapped: grad[2]={grad2}, grad[3]={grad3}, grad[4]={grad4}")
        print(f"  DOF: T={T}, Y1={Y1}, Y2={Y2}")
    
    # Find c_G calculation 
    cg_pattern = r"GPU DEBUG: Phase (\d+) c_G values.*?gradient values: \[(.*?)\].*?full_e_matrix diagonal: \[(.*?)\]"
    cg_matches = list(re.finditer(cg_pattern, output_text, re.DOTALL))
    
    print("\nC_G CALCULATION DATA:")
    print("-" * 25)
    
    for match in cg_matches:
        phase_idx = match.group(1)
        gradients = match.group(2)
        e_matrix_diag = match.group(3)
        
        grad_values = [x.strip() for x in gradients.split(',')]
        e_matrix_values = [x.strip() for x in e_matrix_diag.split(',')]
        
        print(f"Phase {phase_idx}:")
        print(f"  Uses {len(grad_values)} gradient values: {grad_values}")
        print(f"  E-matrix diagonal ({len(e_matrix_values)} values): {e_matrix_values}")

# Test with Al-Cu-Fe ternary system to see the bug
print("TESTING AL-CU-FE TERNARY SYSTEM")
print("="*50)

tdb = Database('Al-Cu-Fe.tdb')

# Capture output
gpu_output = io.StringIO()
old_stdout = sys.stdout
sys.stdout = gpu_output

try:
    eq = equilibrium(tdb, ['AL', 'CU', 'FE', 'VA'], ['LIQUID'], {
        v.T: 1200, v.P: 101325, v.X('CU'): 0.3, v.X('FE'): 0.2
    }, gpu=True, verbose=True)
    
    sys.stdout = old_stdout
    gpu_text = gpu_output.getvalue()
    
    extract_gradient_data(gpu_text)
    
    print("\n" + "="*60)
    print("BUG ANALYSIS:")
    print("="*60)
    
    # Look for phase_dof information
    phase_dof_matches = re.findall(r"phase_dof = (\d+)", gpu_text)
    unique_phase_dofs = list(set(phase_dof_matches))
    print(f"Phase DOF values found: {unique_phase_dofs}")
    
    # Look for c_G calc details
    calc_pattern = r"c_G\[(\d+)\] calc: full_e_matrix\[(\d+),(\d+)\]=(.*?) \* grad\[(\d+)\]=(.*?) = (.*)"
    calc_matches = list(re.finditer(calc_pattern, gpu_text))
    
    if calc_matches:
        print(f"\nFound {len(calc_matches)} c_G calculation steps")
        print("First few calculation steps:")
        for match in calc_matches[:6]:  # First 6 steps
            cg_idx = match.group(1)
            i = match.group(2) 
            j = match.group(3)
            e_val = match.group(4)
            grad_idx = match.group(5)
            grad_val = match.group(6)
            product = match.group(7)
            print(f"  c_G[{cg_idx}] += e_matrix[{i},{j}]={e_val} * grad[{grad_idx}]={grad_val}")
    
    print("\n" + "="*60)
    print("KEY FINDINGS:")
    print("="*60)
    
    print("1. FORMULAGRAD OUTPUT:")
    print("   - For ternary phases with 3 site fractions (phase_dof=3)")
    print("   - formulagrad should output 4 values: [dG/dT, dG/dY1, dG/dY2, dG/dY3]")
    print("   - These get mapped to: grad[2], grad[3], grad[4], grad[5]")
    
    print("\n2. C_G CALCULATION:")
    print("   - c_G loop iterates over phase_dof (3 for ternary)")
    print("   - Uses grad[num_statevars + j] = grad[3+0], grad[3+1], grad[3+2]")
    print("   - This should access grad[3], grad[4], grad[5]")
    
    print("\n3. THE BUG:")
    print("   - If formulagrad only outputs 3 values instead of 4")
    print("   - Or if there's a mapping issue with num_statevars")
    print("   - Then grad[5] might be uninitialized/wrong")
    print("   - Leading to incorrect c_G[2] and wrong overall c_G values!")
    
except Exception as e:
    sys.stdout = old_stdout
    print(f"Error: {e}")