#!/usr/bin/env python
"""Trace the exact inputs to Hessian functions for CPU and GPU."""

import os
import sys
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import numpy as np

# First, let's add more detailed debug output to the CPU and GPU code
# Add debug prints to capture the DOF values passed to formulahess

# Modify CPU code to print DOF before formulahess
cpu_debug_patch = """
            if (state.iteration < 3) {
                printf("[CPU FORMULAHESS INPUT] Phase %d iteration %d, DOF: ", idx, state.iteration);
                for (int i = 0; i < 5; i++) {
                    printf("%.15e ", compsets[idx].dof[i]);
                }
                printf("\\n");
            }
"""

# Read minimizer.pyx and add debug before formulahess call
with open('pycalphad/core/minimizer.pyx', 'r') as f:
    minimizer_content = f.read()

# Find the formulahess call and add debug before it
if 'phase_records[idx].formulahess' in minimizer_content and '[CPU FORMULAHESS INPUT]' not in minimizer_content:
    # Add debug before the formulahess call
    formulahess_pos = minimizer_content.find('phase_records[idx].formulahess(')
    if formulahess_pos > 0:
        # Find the start of the line
        line_start = minimizer_content.rfind('\n', 0, formulahess_pos) + 1
        indent = ' ' * (formulahess_pos - line_start)
        
        # Insert debug code
        new_content = (minimizer_content[:line_start] + 
                      cpu_debug_patch.replace('    ', indent) +
                      minimizer_content[line_start:])
        
        with open('pycalphad/core/minimizer.pyx', 'w') as f:
            f.write(new_content)
        
        print("✓ Added CPU formulahess input debug")

# Now add GPU debug - modify minimizer.h
with open('pycalphad/gpu/minimizer.h', 'r') as f:
    gpu_content = f.read()

# Find GPU formulahess call and add debug
if 'pr->formulahess(csst->hess, compset->dof)' in gpu_content and 'GPU FORMULAHESS INPUT' not in gpu_content:
    gpu_debug = """// DEBUG: Print DOF values passed to formulahess
                if (iteration < 3) {
                    printf("[GPU FORMULAHESS INPUT] Phase %d iteration %d, DOF: ", idx, iteration);
                    for (int i = 0; i < 5; i++) {
                        printf("%.15e ", compset->dof[i]);
                    }
                    printf("\\n");
                }
                """
    
    # Find the formulahess call
    formulahess_pos = gpu_content.find('pr->formulahess(csst->hess, compset->dof);')
    if formulahess_pos > 0:
        # Insert before the call
        gpu_content = gpu_content[:formulahess_pos] + gpu_debug + gpu_content[formulahess_pos:]
        
        with open('pycalphad/gpu/minimizer.h', 'w') as f:
            f.write(gpu_content)
        
        print("✓ Added GPU formulahess input debug")

# Create test script
test_script = """#!/usr/bin/env python
import os
import sys
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import numpy as np

# Enable debug
os.environ['PYCALPHAD_DEBUG_CATEGORIES'] = 'HESSIAN'

# Create test case
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

print("=== Tracing Hessian Function Inputs ===")
print("Testing with T=1000K, X(TI)=0.01")

# Test conditions
conditions = {v.X('TI'): 0.01, v.T: 1000, v.P: 101325}

# Run CPU
print("\\n--- CPU Calculation ---")
try:
    cpu_result = equilibrium(dbf, comps, phases, conditions, gpu=False)
    print("✓ CPU completed")
except Exception as e:
    print(f"✗ CPU failed: {e}")

# Run GPU  
print("\\n--- GPU Calculation ---")
try:
    gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True)
    print("✓ GPU completed")
except Exception as e:
    print(f"✗ GPU failed: {e}")
"""

with open('trace_hessian_inputs_test.py', 'w') as f:
    f.write(test_script)

print("\n✓ Created trace_hessian_inputs_test.py")
print("\nNow run: python trace_hessian_inputs_test.py 2>&1 | grep 'FORMULAHESS INPUT'")
print("This will show the exact DOF values passed to each Hessian calculation.")