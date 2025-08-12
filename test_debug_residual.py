#!/usr/bin/env python

import subprocess
import tempfile
import os

# Create GPU test script with verbose output
gpu_script = """
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')
from pycalphad import Database, equilibrium
dbf = Database('Al-Cu-Fe.tdb')
result = equilibrium(dbf, ['AL','CU','FE','VA'], ['LIQUID'], 
                    {'T': 973.15, 'P': 101325, 'X_AL': 0.5, 'X_CU': 0.2}, 
                    verbose=True, calc_opts={'pdens': 50}, gpu=True)
"""

with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write(gpu_script)
    script_path = f.name

try:
    result = subprocess.run(['python', script_path], 
                          capture_output=True, text=True, timeout=120)
    output = result.stdout + result.stderr
    
    # Look for the residual calculation debug output
    lines = output.split('\n')
    for i, line in enumerate(lines):
        if 'GPU RESIDUAL CALC' in line:
            print(line)
        elif 'GPU CONSTRAINT DEBUG' in line:
            print(line)
            # Print next few lines for context
            for j in range(1, 10):
                if i+j < len(lines):
                    next_line = lines[i+j]
                    if 'System mole fractions' in next_line or 'Target value' in next_line:
                        print(next_line)
        elif 'GPU EQUILIBRIUM MATRIX' in line and 'Iteration 0' in line:
            print(line)
            # Print the matrix rows
            for j in range(1, 10):
                if i+j < len(lines):
                    next_line = lines[i+j]
                    if 'Row 3:' in next_line:
                        print(next_line)
                        break
                        
except Exception as e:
    print(f"Error: {e}")
finally:
    os.unlink(script_path)