#!/usr/bin/env python3
"""
Trace the exact Hessian calculations for BCC_A2 phase in both CPU and GPU.
Looking for hess[3,3] = 5.543000e+03 and hess[4,4] = 4.988700e+04
"""

from pycalphad import Database, equilibrium, variables as v
import subprocess
import re
import numpy as np

# First, let's run a CPU equilibrium calculation with detailed output
script = '''
import numpy as np
from pycalphad import Database, Model, equilibrium, variables as v
from pycalphad.core.utils import unpack_components, unpack_phases
from pycalphad.core.phase_rec import PhaseRecord
import pycalphad.core.eqsolver
import types

# Patch the CPU solver to print Hessian values
original_compute_hessian = None

def patched_compute_hessian(self, workspace, out):
    """Patched version that prints Hessian values"""
    # Call original function
    result = original_compute_hessian(self, workspace, out)
    
    # Print Hessian values
    if hasattr(self, '_phase_dof') and self._phase_dof > 0:
        hess = out.reshape((self._phase_dof, self._phase_dof))
        print(f"\\n[CPU HESSIAN] Phase {self.phase_name} Hessian:")
        for i in range(hess.shape[0]):
            print(f"  Row {i}: " + " ".join(f"{hess[i,j]:12.6e}" for j in range(hess.shape[1])))
        
        # Check for target values
        if hess.shape[0] > 3 and hess.shape[1] > 3:
            if abs(hess[3,3] - 5.543000e+03) < 1:
                print(f"  *** FOUND hess[3,3] = 5.543000e+03! ***")
        if hess.shape[0] > 4 and hess.shape[1] > 4:
            if abs(hess[4,4] - 4.988700e+04) < 1:
                print(f"  *** FOUND hess[4,4] = 4.988700e+04! ***")
                
    return result

# Apply the patch
if hasattr(PhaseRecord, 'formulahess'):
    original_compute_hessian = PhaseRecord.formulahess
    PhaseRecord.formulahess = patched_compute_hessian

db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']

# Test different conditions
test_conditions = [
    {v.N: 1, v.P: 101325, v.T: 1000, v.X('TI'): 0.3},
    {v.N: 1, v.P: 101325, v.T: 300, v.X('TI'): 0.3},
    {v.N: 1, v.P: 101325, v.T: 300, v.X('TI'): 0.226294},
]

for i, conditions in enumerate(test_conditions):
    print(f"\\n\\n{'='*60}")
    print(f"Test {i+1}: T={conditions[v.T]}K, X(TI)={conditions.get(v.X('TI'), 'N/A')}")
    print('='*60)
    
    try:
        # Run with verbose output to see Hessian calculations
        result = equilibrium(db, comps, phases, conditions, verbose=True, 
                           calc_opts={"pdens": 50})
        print(f"\\nEquilibrium phases: {result.Phase.values}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
'''

print("Running CPU equilibrium calculation with Hessian tracing...")
print("Looking for: hess[3,3] = 5.543000e+03 and hess[4,4] = 4.988700e+04")
print("="*80)

result = subprocess.run(['python', '-c', script], capture_output=True, text=True)

output = result.stdout + result.stderr

# Search for the specific values
if '5.543' in output or '5543' in output:
    print("\n*** Found reference to 5.543/5543 in output! ***")
    
if '4.9887' in output or '49887' in output:
    print("\n*** Found reference to 4.9887/49887 in output! ***")

# Extract and display any Hessian output
hessian_lines = []
in_hessian = False
for line in output.split('\n'):
    if '[CPU HESSIAN]' in line:
        in_hessian = True
        hessian_lines.append(line)
    elif in_hessian and ('Row' in line or '***' in line):
        hessian_lines.append(line)
    elif in_hessian and not line.strip().startswith(' '):
        in_hessian = False

if hessian_lines:
    print("\nFound Hessian output:")
    for line in hessian_lines:
        print(line)
else:
    print("\nNo Hessian output found in CPU calculation.")

# Save full output
with open('cpu_hessian_trace.txt', 'w') as f:
    f.write(output)
print(f"\nFull output saved to cpu_hessian_trace.txt")

# Now check GPU Hessian
print("\n" + "="*80)
print("Checking GPU Hessian calculation...")

gpu_script = '''
import os
os.environ["GPU_DEBUG"] = "1"

from pycalphad import Database, equilibrium, variables as v

db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']  
phases = ['BCC_A2', 'LIQUID']

conditions = {v.N: 1, v.P: 101325, v.T: 300, v.X('TI'): 0.226294}

try:
    result = equilibrium(db, comps, phases, conditions, gpu=True, calc_opts={"pdens": 50})
    print(f"GPU equilibrium phases: {result.Phase.values}")
except Exception as e:
    print(f"GPU Error: {e}")
    import traceback
    traceback.print_exc()
'''

result = subprocess.run(['python', '-c', gpu_script], capture_output=True, text=True)
gpu_output = result.stdout + result.stderr

# Check for Hessian values in GPU output
gpu_hessian_lines = []
for line in gpu_output.split('\n'):
    if 'hessian' in line.lower() or 'hess' in line.lower():
        gpu_hessian_lines.append(line)

if gpu_hessian_lines:
    print("\nFound GPU Hessian references:")
    for line in gpu_hessian_lines[:10]:  # First 10 lines
        print(line)

with open('gpu_hessian_trace.txt', 'w') as f:
    f.write(gpu_output)
print(f"\nGPU output saved to gpu_hessian_trace.txt")

print("\nDone.")