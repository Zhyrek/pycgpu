#!/usr/bin/env python
"""Extract CPU and GPU free energy values from debug output."""

import os
import sys
import re
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Single test case
conditions = {v.X('TI'): 0.05, v.T: 800, v.P: 101325}

print("Testing X(TI)=0.05, T=800K")
print("=" * 50)

# CPU calculation - capture all output
print("Running CPU calculation...")
import subprocess
import tempfile

with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write(f"""
import sys
sys.path.insert(0, '{os.getcwd()}')
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {{v.X('TI'): 0.05, v.T: 800, v.P: 101325}}

result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
cpu_gm = result_cpu.GM.values.flatten()[0]
print(f"CPU_GM: {{cpu_gm:.6f}} J/mol")

result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
gpu_gm = result_gpu.GM.values.flatten()[0]  
print(f"GPU_GM: {{gpu_gm:.6f}} J/mol")

error = abs(cpu_gm - gpu_gm)
print(f"ABSOLUTE_ERROR: {{error:.6f}} J/mol")
""")
    temp_file = f.name

try:
    result = subprocess.run(['python', temp_file], capture_output=True, text=True, timeout=120)
    
    # Extract values from output
    cpu_match = re.search(r'CPU_GM: ([-\d.]+) J/mol', result.stdout)
    gpu_match = re.search(r'GPU_GM: ([-\d.]+) J/mol', result.stdout)
    error_match = re.search(r'ABSOLUTE_ERROR: ([\d.]+) J/mol', result.stdout)
    
    if cpu_match and gpu_match and error_match:
        cpu_gm = float(cpu_match.group(1))
        gpu_gm = float(gpu_match.group(1))
        error = float(error_match.group(1))
        
        print(f"CPU Free Energy: {cpu_gm:.3f} J/mol")
        print(f"GPU Free Energy: {gpu_gm:.3f} J/mol")
        print(f"Absolute Error:  {error:.3f} J/mol")
        print(f"Relative Error:  {(error/abs(cpu_gm)*100):.4f}%")
    else:
        print("Could not extract values from output:")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        
finally:
    os.unlink(temp_file)