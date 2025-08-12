#!/usr/bin/env python

import warnings
warnings.filterwarnings('ignore')
import subprocess
import tempfile

# GPU test script to capture debug output
gpu_script = """
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')
from pycalphad import Database, equilibrium
dbf = Database('Al-Cu-Fe.tdb')
result = equilibrium(dbf, ['AL','CU','FE','VA'], ['LIQUID'], {'T': 973.15, 'P': 101325, 'X_AL': 0.5, 'X_CU': 0.2}, verbose=True, calc_opts={'pdens': 50}, gpu=True)
"""

with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write(gpu_script)
    script_path = f.name

try:
    result = subprocess.run(['python', script_path], 
                          capture_output=True, text=True, timeout=120)
    output = result.stdout + result.stderr
    
    # Look for constraint debug output
    lines = output.split('\n')
    for i, line in enumerate(lines):
        if 'GPU CONSTRAINT DEBUG' in line:
            print(line)
            for j in range(1, 10):
                if i+j < len(lines):
                    next_line = lines[i+j]
                    if 'System mole fractions' in next_line or 'ANALYSIS' in next_line or 'Coefficients' in next_line or 'Target value' in next_line:
                        print(next_line)
                    if 'Row' in next_line and 'residual' in next_line:
                        break
            print()
            
except Exception as e:
    print(f"Error: {e}")
finally:
    import os
    os.unlink(script_path)