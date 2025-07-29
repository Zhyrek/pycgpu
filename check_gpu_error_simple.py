#!/usr/bin/env python
"""Try to run GPU with a simple phase to see the exact error."""

from pycalphad import Database, equilibrium
import pycalphad.variables as v
import warnings
warnings.filterwarnings('ignore')

db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']

# Test with just LIQUID phase which we know worked before
conditions = {
    v.T: 1273.15,
    v.P: 101325,
    v.X('AL'): 0.3,
    v.X('CU'): 0.3,
    v.N: 1
}

print("Testing GPU with just LIQUID phase...")
try:
    result = equilibrium(db, components, ['LIQUID'], conditions,
                        calc_opts={'pdens': 10}, gpu=True, verbose=True)
    print("SUCCESS!")
except Exception as e:
    print(f"FAILED: {type(e).__name__}")
    # The error message contains the actual compilation errors
    error_str = str(e)
    
    # Look for the actual error lines
    lines = error_str.split('\\n')
    print("\nExtracting actual compilation errors:")
    print("="*60)
    
    error_lines = []
    for i, line in enumerate(lines):
        if 'error' in line and 'error detected' not in line:
            error_lines.append(line)
            # Also get context
            if i > 0:
                error_lines.append(lines[i-1])
            if i < len(lines) - 1:
                error_lines.append(lines[i+1])
    
    # Show first few errors
    print("First compilation errors found:")
    for line in error_lines[:10]:
        print(line)