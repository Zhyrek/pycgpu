#!/usr/bin/env python3
"""
Run equilibrium and extract just the Hessian values for comparison.
"""

from pycalphad import Database, equilibrium, variables as v
import numpy as np
import subprocess
import re

db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']
eq_conditions = {v.N: 1, v.P: 101325, v.T: 1000, v.X('TI'): 0.3}

# Run GPU calculation and capture output
print("Running GPU calculation...")
result = subprocess.run(['python', '-c', '''
from pycalphad import Database, equilibrium, variables as v
import os
os.environ["GPU_DEBUG"] = "1"
db = Database("NbTi.tdb")
comps = ["NB", "TI", "VA"]
phases = ["BCC_A2", "LIQUID"]
eq_conditions = {v.N: 1, v.P: 101325, v.T: 1000, v.X("TI"): 0.3}
gpu_result = equilibrium(db, comps, phases, eq_conditions, verbose=False, gpu=True, to="GM", calc_opts={"pdens": 50})
'''], capture_output=True, text=True)

# Extract GPU Hessian values
gpu_output = result.stderr
hessian_matches = re.findall(r'Site fraction Hessian block:\s*\n\s*\[0\]\s*([\d.e+-]+).*\n\s*\[1\]\s*([\d.e+-]+)', gpu_output)

if hessian_matches:
    print("\n=== GPU Hessian Values ===")
    for i, match in enumerate(hessian_matches[:2]):  # First 2 matches
        h33 = float(match[0].split()[0])
        print(f"Phase {i}: H[3,3] = {h33:.6e}")

# Run CPU calculation  
print("\n\nRunning CPU calculation...")
result = subprocess.run(['python', '-c', '''
from pycalphad import Database, equilibrium, variables as v
db = Database("NbTi.tdb")
comps = ["NB", "TI", "VA"]
phases = ["BCC_A2", "LIQUID"]
eq_conditions = {v.N: 1, v.P: 101325, v.T: 1000, v.X("TI"): 0.3}
cpu_result = equilibrium(db, comps, phases, eq_conditions, verbose=True, gpu=False, to="GM", calc_opts={"pdens": 50})
'''], capture_output=True, text=True)

# Extract CPU Hessian values
cpu_output = result.stdout + result.stderr
cpu_hessian_matches = re.findall(r'Row 3:\s*([\d.e+-]+)\s*([\d.e+-]+)', cpu_output)

if cpu_hessian_matches:
    print("\n=== CPU Hessian Values ===")
    for i, match in enumerate(cpu_hessian_matches[:2]):  # First 2 matches
        h33 = float(match[0])
        print(f"Phase {i}: H[3,3] = {h33:.6e}")

print("\n=== Direct Comparison ===")
print("The GPU and CPU should produce identical Hessian values with identical inputs.")