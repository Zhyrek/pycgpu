#!/usr/bin/env python
"""Test gradient mapping differences between binary and ternary systems."""

from pycalphad import Database, equilibrium, variables as v
import numpy as np
import io
import sys
import re

def capture_gpu_output(tdb, comps, phases, conditions, system_name):
    """Capture GPU debug output for analysis."""
    print(f"\n{system_name} SYSTEM GPU DEBUG OUTPUT:")
    print("=" * 50)
    
    # Capture GPU output
    gpu_output = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = gpu_output
    
    try:
        eq_gpu = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=True)
        sys.stdout = old_stdout
        gpu_text = gpu_output.getvalue()
        
        # Look for formulagrad debug output
        grad_debug_pattern = r"\[GPU GRAD DEBUG\] Phase formulagrad results:(.*?)(?=\[|$)"
        grad_matches = list(re.finditer(grad_debug_pattern, gpu_text, re.DOTALL))
        
        print(f"Found {len(grad_matches)} formulagrad debug outputs")
        
        for i, match in enumerate(grad_matches[:2]):  # First 2 for brevity
            details = match.group(1)
            print(f"\nFormulagrad result {i}:")
            
            # Extract temp_grad values
            temp_grad_match = re.search(r"temp_grad: \[(.*?)\]", details)
            if temp_grad_match:
                temp_grad_str = temp_grad_match.group(1)
                print(f"  temp_grad: [{temp_grad_str}]")
                
            # Extract mapped grad values  
            mapped_match = re.search(r"mapped to grad\[2\]=(.*?), grad\[3\]=(.*?)(?:, grad\[4\]=(.*?))?(?:\s|$)", details)
            if mapped_match:
                grad2 = mapped_match.group(1)
                grad3 = mapped_match.group(2) 
                grad4 = mapped_match.group(3) if mapped_match.group(3) else "N/A"
                print(f"  mapped to: grad[2]={grad2}, grad[3]={grad3}, grad[4]={grad4}")
                
            # Extract DOF values
            dof_match = re.search(r"DOF values: T=(.*?), Y1=(.*?)(?:, Y2=(.*?))?", details)
            if dof_match:
                T = dof_match.group(1)
                Y1 = dof_match.group(2)
                Y2 = dof_match.group(3) if dof_match.group(3) else "N/A"
                print(f"  DOF values: T={T}, Y1={Y1}, Y2={Y2}")
                
        # Look for c_G calculation details
        cg_pattern = r"GPU DEBUG: Phase (\d+) c_G values.*?gradient values: \[(.*?)\].*?full_e_matrix diagonal: \[(.*?)\]"
        cg_matches = list(re.finditer(cg_pattern, gpu_text, re.DOTALL))
        
        print(f"\nFound {len(cg_matches)} c_G calculation details")
        
        for match in cg_matches[:2]:  # First 2 phases
            phase_idx = match.group(1)
            gradients = match.group(2)
            e_matrix_diag = match.group(3)
            
            print(f"\nPhase {phase_idx} c_G calculation:")
            print(f"  Gradients: [{gradients}]")
            print(f"  E-matrix diagonal: [{e_matrix_diag}]")
            
        # Look for phase_dof information
        phase_dof_pattern = r"phase_dof = (\d+)"
        phase_dof_matches = re.findall(phase_dof_pattern, gpu_text)
        if phase_dof_matches:
            unique_dofs = set(phase_dof_matches)
            print(f"\nPhase DOF values found: {unique_dofs}")
            
        return gpu_text
        
    except Exception as e:
        sys.stdout = old_stdout
        print(f"Error in {system_name}: {e}")
        return ""

# Test binary system (Au-Bi)
tdb_binary = Database('AuBi-07Wan.tdb')
comps_binary = ['AU', 'BI', 'VA']
phases_binary = ['LIQUID', 'FCC_A1']

conditions_binary = {
    v.T: 800,
    v.P: 101325,
    v.X('BI'): 0.3
}

binary_output = capture_gpu_output(tdb_binary, comps_binary, phases_binary, 
                                 conditions_binary, "BINARY")

# Test ternary system (Al-Cu-Fe) 
tdb_ternary = Database('Al-Cu-Fe.tdb')
comps_ternary = ['AL', 'CU', 'FE', 'VA']
phases_ternary = ['LIQUID', 'BCC_A2']

conditions_ternary = {
    v.T: 1200,
    v.P: 101325,
    v.X('CU'): 0.3,
    v.X('FE'): 0.2
}

ternary_output = capture_gpu_output(tdb_ternary, comps_ternary, phases_ternary,
                                  conditions_ternary, "TERNARY")

print("\n" + "=" * 80)
print("ANALYSIS SUMMARY")
print("=" * 80)

print("\nKEY QUESTIONS TO ANSWER:")
print("1. Does binary formulagrad output 2 values [dG/dT, dG/dY1]?")
print("2. Does ternary formulagrad output 3 values [dG/dT, dG/dY1, dG/dY2]?")
print("3. Are the gradient mappings correct in both cases?")
print("4. Are the c_G calculation loop bounds using the right phase_dof?")
print("5. Are the gradients being used from the right indices in c_G calc?")

if "temp_grad: [" in binary_output:
    print("\n✓ Binary system formulagrad debug output found")
else:
    print("\n✗ Binary system formulagrad debug output NOT found")
    
if "temp_grad: [" in ternary_output:
    print("✓ Ternary system formulagrad debug output found")
else:
    print("✗ Ternary system formulagrad debug output NOT found")