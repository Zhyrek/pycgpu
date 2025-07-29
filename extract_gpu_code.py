#!/usr/bin/env python
"""Extract and examine specific GPU generated functions."""

from pycalphad import Database
from pycalphad.core.workspace import Workspace
from pycalphad.gpu import gpu_equilibrium
import pycalphad.variables as v
import re

# Load database
db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']
phases = list(db.phases.keys())

# Create workspace
conditions = {
    v.T: 1273.15,
    v.P: 101325,
    v.X('AL'): 0.3,
    v.X('CU'): 0.3,
    v.N: 1
}

wks = Workspace(db, components, phases[:3], conditions)  # Just first 3 phases

print("Attempting to generate GPU code and capture the generated C functions...")
print("="*80)

# Try to run GPU equilibrium with verbose to capture the generated code
try:
    # Monkey patch to capture generated code
    original_generate = gpu_equilibrium._generate_c_code_for_phase_models
    generated_code = {}
    
    def capture_generate(wks_obj, include_hess=False, validate=True):
        result = original_generate(wks_obj, include_hess, validate)
        generated_code.update(result)
        return result
    
    gpu_equilibrium._generate_c_code_for_phase_models = capture_generate
    
    # Try to run equilibrium (will fail but we want the generated code)
    try:
        from pycalphad import equilibrium
        result = equilibrium(db, components, phases[:3], conditions, 
                           calc_opts={'pdens': 10}, gpu=True, verbose=True)
    except Exception as e:
        print(f"Expected failure: {type(e).__name__}")
    
    # Restore original
    gpu_equilibrium._generate_c_code_for_phase_models = original_generate
    
except Exception as e:
    print(f"Error: {e}")
    generated_code = {}

# Now analyze the captured code
if generated_code:
    print(f"\nCaptured {len(generated_code)} generated functions")
    
    # Look for specific patterns
    for func_name, func_code in list(generated_code.items())[:5]:
        print(f"\n{'='*60}")
        print(f"Function: {func_name}")
        print(f"{'='*60}")
        
        # Show the function signature and first few lines
        lines = func_code.split('\n')
        for i, line in enumerate(lines[:20]):
            print(f"{i+1:3d}: {line}")
        
        # Look for problematic patterns
        print("\nChecking for issues:")
        
        # Pattern 1: Adjacent numbers
        for i, line in enumerate(lines):
            match = re.search(r'(\d+\.?\d*)\s+(\d+\.?\d*)', line)
            if match:
                print(f"  Line {i+1}: Adjacent numbers: {match.group()}")
        
        # Pattern 2: Operators at line start (might indicate missing semicolon)
        for i, line in enumerate(lines):
            if line.strip().startswith(('+', '-', '*', '/')) and i > 0:
                print(f"  Line {i+1}: Operator at line start: {line.strip()[:50]}")
        
        # Pattern 3: Very long lines (might have concatenation issues)
        for i, line in enumerate(lines):
            if len(line) > 200:
                print(f"  Line {i+1}: Very long line ({len(line)} chars)")
else:
    print("Could not capture generated code")

# Alternative: Try to intercept at a lower level
print("\n" + "="*80)
print("Attempting direct code generation...")

try:
    from pycalphad.gpu.gpu_codegen import _generate_c_code_for_phase_models
    
    # Generate code directly
    phase_codes = _generate_c_code_for_phase_models(wks, include_hess=True, validate=False)
    
    print(f"Generated {len(phase_codes)} code blocks")
    
    # Find a specific function to examine
    for key, code in phase_codes.items():
        if '_obj' in key and 'AL13FE4' in key:
            print(f"\nExample function: {key}")
            print("="*60)
            lines = code.split('\n')
            for i, line in enumerate(lines[:30]):
                print(f"{i+1:3d}: {line}")
            
            # Check for the specific error patterns
            print("\nPotential issues in this function:")
            for i, line in enumerate(lines):
                # Look for number followed by identifier without operator
                match = re.search(r'(\d+\.?\d*)\s*([a-zA-Z_]\w*)', line)
                if match and not re.search(r'(\d+\.?\d*)\s*\*\s*([a-zA-Z_]\w*)', line):
                    print(f"  Line {i+1}: Number before identifier: '{match.group()}'")
            break
            
except Exception as e:
    print(f"Direct generation failed: {e}")