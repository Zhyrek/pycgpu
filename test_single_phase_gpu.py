#!/usr/bin/env python
"""Test single phase GPU compilation to isolate the issue."""

from pycalphad import Database
from pycalphad.core.workspace import Workspace
from pycalphad.gpu.gpu_equilibrium import _test_compile_gpu_kernel
import pycalphad.variables as v
import os

# Load database
db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']

# Create workspace
conditions = {
    v.T: 1273.15,
    v.P: 101325,
    v.X('AL'): 0.3,
    v.X('CU'): 0.3,
    v.N: 1
}

print("Testing individual phase compilation...")
print("="*80)

# Test each phase individually
problematic_phases = []

for phase in list(db.phases.keys())[:10]:
    print(f"\nTesting {phase}:")
    
    try:
        # Create workspace with single phase
        wks = Workspace(db, components, [phase], conditions, verbose=False)
        
        # Try to compile just the kernel
        from pycalphad.gpu.gpu_equilibrium import _compile_gpu_equilibrium_kernel
        from pycalphad.gpu.gpu_codegen import _generate_c_code_for_phase_models
        
        # Generate code
        result = _generate_c_code_for_phase_models(wks, include_hess=True, validate=False)
        all_device_functions, init_calls, unique_models, phase_map = result
        
        print(f"  Generated code length: {len(all_device_functions)} chars")
        
        # Check for extremely long functions
        max_line_length = max(len(line) for line in all_device_functions.split('\n'))
        print(f"  Max line length: {max_line_length} chars")
        
        if max_line_length > 5000:
            print(f"  ⚠️  WARNING: Very long lines detected!")
            problematic_phases.append((phase, max_line_length))
            
            # Save the generated code for inspection
            filename = f"gpu_code_{phase}_long.c"
            with open(filename, 'w') as f:
                f.write(f"// Generated code for {phase}\n")
                f.write(f"// Max line length: {max_line_length}\n\n")
                f.write(all_device_functions[:10000])  # First 10k chars
                f.write("\n\n... truncated ...\n")
            print(f"  Saved to: {filename}")
        
        # Count number of expressions/functions
        func_count = all_device_functions.count('__device__')
        print(f"  Number of device functions: {func_count}")
        
    except Exception as e:
        print(f"  ✗ ERROR: {type(e).__name__}: {str(e)[:100]}")
        problematic_phases.append((phase, str(e)))

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if problematic_phases:
    print("\nProblematic phases:")
    for phase, issue in problematic_phases:
        print(f"  {phase}: {issue}")
else:
    print("\nAll phases generated code successfully.")

# Now test a simple equilibrium calculation with LIQUID only
print("\n" + "="*80)
print("Testing actual GPU compilation with LIQUID phase...")

try:
    from pycalphad import equilibrium
    result = equilibrium(db, components, ['LIQUID'], conditions, 
                        calc_opts={'pdens': 10}, 
                        gpu=True, 
                        verbose=True)
    print("✓ SUCCESS: LIQUID phase compiled and ran on GPU")
except Exception as e:
    print(f"✗ FAILED: {type(e).__name__}: {str(e)[:200]}")
    
    # Check if it's a compilation error
    if 'nvcc' in str(e):
        print("\nThis is an nvcc compilation error. The generated code is likely too complex.")