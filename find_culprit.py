#!/usr/bin/env python

import subprocess
import tempfile
import os

# Create GPU script to capture the detailed constraint debug output
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
    
    print("=== SEARCHING FOR CONSTRAINT DEBUG OUTPUT ===")
    
    # Look for our detailed analysis
    lines = output.split('\n')
    found_analysis = False
    for i, line in enumerate(lines):
        if 'DETAILED ANALYSIS for AL constraint' in line:
            found_analysis = True
            print("FOUND THE CULPRIT DEBUG OUTPUT:")
            print(line)
            # Print next 10 lines to get all the analysis
            for j in range(1, 11):
                if i+j < len(lines):
                    print(lines[i+j])
            break
    
    if not found_analysis:
        print("DEBUG OUTPUT NOT FOUND. Looking for any constraint debug...")
        for line in lines:
            if 'GPU CONSTRAINT DEBUG' in line or 'mole_fractions' in line:
                print(line)
        
        if result.returncode != 0:
            print("\nERROR OUTPUT:")
            print(result.stderr[:1000])
            
except Exception as e:
    print(f"Error running test: {e}")
finally:
    os.unlink(script_path)