#!/usr/bin/env python3
"""
Search for the specific Hessian values mentioned:
hess[3,3] = 5.543000e+03 and hess[4,4] = 4.988700e+04
"""

import subprocess
import re

# Run a detailed trace to look for these specific values
script = '''
from pycalphad import Database, equilibrium, variables as v
import numpy as np
import os

# Enable detailed debugging
os.environ["CPU_DEBUG"] = "1"
os.environ["GPU_DEBUG"] = "1"

db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']

# Test different conditions
test_conditions = [
    {v.N: 1, v.P: 101325, v.T: 1000, v.X('TI'): 0.3},
    {v.N: 1, v.P: 101325, v.T: 300, v.X('TI'): 0.3},
    {v.N: 1, v.P: 101325, v.T: 300, v.X('TI'): 0.226294},  # Specific Y(TI) value
]

for i, conditions in enumerate(test_conditions):
    print(f"\\n\\n{'='*60}")
    print(f"Test {i+1}: T={conditions[v.T]}K, X(TI)={conditions.get(v.X('TI'), 'N/A')}")
    print('='*60)
    
    # Run CPU
    print("\\nCPU Calculation:")
    try:
        result = equilibrium(db, comps, phases, conditions, verbose=True, gpu=False, 
                           to="GM", calc_opts={"pdens": 50})
    except Exception as e:
        print(f"CPU Error: {e}")
    
    # Run GPU  
    print("\\nGPU Calculation:")
    try:
        result = equilibrium(db, comps, phases, conditions, verbose=False, gpu=True,
                           to="GM", calc_opts={"pdens": 50})
    except Exception as e:
        print(f"GPU Error: {e}")
'''

# Run the script and capture output
print("Running equilibrium calculations to find specific Hessian values...")
print("Looking for: hess[3,3] = 5.543000e+03 and hess[4,4] = 4.988700e+04")
print("="*80)

result = subprocess.run(['python', '-c', script], capture_output=True, text=True)

# Search for the specific values in the output
output = result.stdout + result.stderr

# Look for patterns that might contain our values
patterns = [
    r'5\.543.*e\+03|5543',
    r'4\.9887.*e\+04|49887',
    r'hess\[3\]\[3\].*=.*[\d.e+-]+',
    r'hess\[4\]\[4\].*=.*[\d.e+-]+',
    r'Row 3:.*[\d.e+-]+',
    r'Row 4:.*[\d.e+-]+',
]

print("\nSearching output for specific values...")
for pattern in patterns:
    matches = re.findall(pattern, output, re.IGNORECASE)
    if matches:
        print(f"\nPattern '{pattern}' matches:")
        for match in matches[:5]:  # Show first 5 matches
            print(f"  {match}")

# Also check if these values appear anywhere in the output
if '5543' in output or '5.543' in output:
    print("\n*** Found reference to 5543 in output! ***")
    # Find context
    for i, line in enumerate(output.split('\n')):
        if '5543' in line or '5.543' in line:
            start = max(0, i-2)
            end = min(len(output.split('\n')), i+3)
            print("Context:")
            for j in range(start, end):
                print(f"  {output.split('\n')[j]}")

if '49887' in output or '4.9887' in output:
    print("\n*** Found reference to 49887 in output! ***")
    # Find context
    for i, line in enumerate(output.split('\n')):
        if '49887' in line or '4.9887' in line:
            start = max(0, i-2)
            end = min(len(output.split('\n')), i+3)
            print("Context:")
            for j in range(start, end):
                print(f"  {output.split('\n')[j]}")

# Save full output for inspection
with open('hessian_search_output.txt', 'w') as f:
    f.write(output)
print(f"\nFull output saved to hessian_search_output.txt ({len(output)} chars)")