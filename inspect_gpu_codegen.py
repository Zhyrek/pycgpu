#!/usr/bin/env python
"""Inspect the generated C code for each phase to find systematic errors."""

from pycalphad import Database
from pycalphad.core.workspace import Workspace
from pycalphad.gpu.gpu_codegen import _generate_phase_energy_functions
import pycalphad.variables as v
import numpy as np

# Load database
db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']
phases = list(db.phases.keys())

print("="*80)
print("GPU Code Generation Inspection for Al-Cu-Fe Phases")
print("="*80)

# Create a minimal workspace for code generation
conditions = {
    v.T: 1273.15,
    v.P: 101325,
    v.X('AL'): 0.3,
    v.X('CU'): 0.3,
    v.N: 1
}

# Create workspace
wks = Workspace(db, components, phases, conditions, parameters=None)

# Generate code for each phase and inspect
phase_codes = {}
errors_found = []

for i, phase_name in enumerate(phases[:5]):  # Start with first 5 phases
    print(f"\n{'='*60}")
    print(f"Phase {i}: {phase_name}")
    print(f"{'='*60}")
    
    try:
        # Generate the energy functions for this phase
        model = wks.models[phase_name]
        
        # Get the variable mapping
        print(f"\nSublattices: {model.phase_obj.sublattices}")
        print(f"Site fractions: {model.site_fractions}")
        
        # Generate the code
        code_dict = _generate_phase_energy_functions(
            models={phase_name: model},
            max_dof_per_phase=len(model.site_fractions),
            verbose=True
        )
        
        # Extract the generated functions
        for func_name, func_code in code_dict.items():
            if phase_name in func_name:
                print(f"\n--- {func_name} ---")
                # Show first 500 chars of each function
                if len(func_code) > 500:
                    print(func_code[:500] + "...")
                else:
                    print(func_code)
                
                # Check for common issues
                issues = []
                
                # Check for adjacent numbers without operators
                import re
                adjacent_nums = re.findall(r'(\d+\.?\d*)\s+(\d+\.?\d*)', func_code)
                if adjacent_nums:
                    issues.append(f"Adjacent numbers without operators: {adjacent_nums[:3]}")
                
                # Check for double operators
                double_ops = re.findall(r'[\+\-\*/]{2,}', func_code)
                if double_ops:
                    issues.append(f"Double operators: {double_ops[:3]}")
                
                # Check for unbalanced parentheses
                open_parens = func_code.count('(')
                close_parens = func_code.count(')')
                if open_parens != close_parens:
                    issues.append(f"Unbalanced parentheses: {open_parens} open, {close_parens} close")
                
                # Check for invalid variable names
                invalid_vars = re.findall(r'\b\d+[a-zA-Z_]\w*\b', func_code)
                if invalid_vars:
                    issues.append(f"Invalid variable names starting with digit: {invalid_vars[:3]}")
                
                if issues:
                    print(f"\n⚠️  POTENTIAL ISSUES:")
                    for issue in issues:
                        print(f"   - {issue}")
                    errors_found.append((phase_name, func_name, issues))
                
        phase_codes[phase_name] = code_dict
        
    except Exception as e:
        print(f"\n❌ ERROR generating code for {phase_name}: {type(e).__name__}: {str(e)[:200]}")
        errors_found.append((phase_name, "generation", str(e)))

print("\n" + "="*80)
print("SUMMARY OF ISSUES FOUND:")
print("="*80)

if errors_found:
    for phase, func, issues in errors_found:
        print(f"\n{phase} - {func}:")
        if isinstance(issues, str):
            print(f"  {issues}")
        else:
            for issue in issues:
                print(f"  - {issue}")
else:
    print("No obvious issues found in the generated code.")

# Also check if we can find patterns in function names
print("\n" + "="*80)
print("GENERATED FUNCTION PATTERNS:")
print("="*80)

all_func_names = []
for phase_name, code_dict in phase_codes.items():
    for func_name in code_dict.keys():
        all_func_names.append(func_name)

print(f"\nTotal functions generated: {len(all_func_names)}")
print("\nFunction name patterns:")
for pattern in ['_obj', '_formulaobj', '_formulagrad', '_formulahess', '_internal_cons', '_mass_obj', '_formulamole']:
    matching = [f for f in all_func_names if pattern in f]
    print(f"  {pattern}: {len(matching)} functions")