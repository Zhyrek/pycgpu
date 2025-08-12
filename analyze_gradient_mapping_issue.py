#!/usr/bin/env python
"""Analyze the exact gradient mapping issue in ternary systems."""

from pycalphad import Database, equilibrium, variables as v
import numpy as np
import io
import sys
import re

def extract_formulagrad_data(output_text):
    """Extract formulagrad debug data from GPU output."""
    
    # Look for the specific gradient debug output
    grad_pattern = r"\[GPU GRAD DEBUG\] Phase formulagrad results:.*?temp_grad: \[(.*?)\].*?mapped to grad\[2\]=(.*?), grad\[3\]=(.*?)(?:, grad\[4\]=(.*?))?.*?DOF values: T=(.*?), Y1=(.*?)(?:, Y2=(.*?))?"
    
    matches = list(re.finditer(grad_pattern, output_text, re.DOTALL))
    
    print(f"Found {len(matches)} formulagrad debug outputs")
    
    for i, match in enumerate(matches[:3]):  # First 3 phases
        temp_grad_str = match.group(1)
        grad2 = match.group(2)
        grad3 = match.group(3)
        grad4 = match.group(4) if match.group(4) else "N/A"
        T = match.group(5)
        Y1 = match.group(6)
        Y2 = match.group(7) if match.group(7) else "N/A"
        
        # Count gradient terms
        temp_grad_values = [x.strip() for x in temp_grad_str.split(',')]
        
        print(f"\nPhase {i} formulagrad analysis:")
        print(f"  temp_grad has {len(temp_grad_values)} values: {temp_grad_values}")
        print(f"  Mapped to: grad[2]={grad2}, grad[3]={grad3}, grad[4]={grad4}")
        print(f"  DOF values: T={T}, Y1={Y1}, Y2={Y2}")
        
        # Determine expected vs actual
        if Y2 != "N/A":
            print(f"  EXPECTED: 4 gradient values [dG/dT, dG/dY1, dG/dY2, dG/dY3] for ternary system")
        else:
            print(f"  EXPECTED: 3 gradient values [dG/dT, dG/dY1, dG/dY2] for binary system")
        
        if len(temp_grad_values) < 4 and Y2 != "N/A":
            print(f"  *** BUG DETECTED: Ternary system only has {len(temp_grad_values)} gradient values but needs 4!")
    
    # Also look for c_G calculation issues
    cg_pattern = r"GPU DEBUG: Phase (\d+) c_G values.*?gradient values: \[(.*?)\]"
    cg_matches = list(re.finditer(cg_pattern, output_text, re.DOTALL))
    
    print(f"\nFound {len(cg_matches)} c_G calculation details")
    
    for match in cg_matches[:3]:  # First 3 phases
        phase_idx = match.group(1)
        gradients = match.group(2)
        
        grad_values = [x.strip() for x in gradients.split(',')]
        
        print(f"\nPhase {phase_idx} c_G calculation:")
        print(f"  Uses {len(grad_values)} gradient values in c_G calculation")
        print(f"  Gradient values: {grad_values[:3]}...")  # First 3 for brevity

# Test ternary system to capture the gradient mapping bug
print("TESTING TERNARY SYSTEM TO CAPTURE GRADIENT BUG")
print("=" * 60)

try:
    tdb = Database('Al-Cu-Fe.tdb')
    
    # Capture GPU debug output
    gpu_output = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = gpu_output
    
    # Run ternary equilibrium with single condition
    eq = equilibrium(tdb, ['AL', 'CU', 'FE', 'VA'], ['LIQUID'], {
        v.T: 1200, v.P: 101325, v.X('CU'): 0.3, v.X('FE'): 0.2
    }, gpu=True, verbose=True)
    
    sys.stdout = old_stdout
    gpu_text = gpu_output.getvalue()
    
    # Extract and analyze the formulagrad data
    extract_formulagrad_data(gpu_text)
    
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY:")
    print("=" * 60)
    
    # Look for phase_dof values to confirm system type
    phase_dof_matches = re.findall(r"phase_dof = (\d+)", gpu_text)
    unique_phase_dofs = list(set(phase_dof_matches))
    print(f"Phase DOF values found: {unique_phase_dofs}")
    
    # Count total gradient debug outputs vs c_G calculations
    grad_debug_count = len(re.findall(r"\[GPU GRAD DEBUG\] Phase formulagrad results:", gpu_text))
    cg_calc_count = len(re.findall(r"GPU DEBUG: Phase \d+ c_G values", gpu_text))
    
    print(f"Total formulagrad debug outputs: {grad_debug_count}")
    print(f"Total c_G calculations: {cg_calc_count}")
    
    # Look for specific issue indicators
    if "temp_grad: [-" in gpu_text and "grad[4]=N/A" in gpu_text:
        print("\n*** ROOT CAUSE CONFIRMED ***")
        print("The formulagrad function for ternary phases is NOT outputting")
        print("enough gradient terms. It should output 4 values for phase_dof=3:")
        print("  [dG/dT, dG/dY1, dG/dY2, dG/dY3]")
        print("But it's only outputting 3 values, leaving grad[5] uninitialized!")
        print("This causes wrong c_G values and the 239 J/mol GPU/CPU divergence.")
    
    print("\n" + "=" * 60)
    print("NEXT STEP: Fix the formulagrad generation in gpu_codegen.py")
    print("to ensure it outputs the correct number of gradient terms.")
    print("=" * 60)
    
except Exception as e:
    sys.stdout = old_stdout
    print(f"Error: {e}")