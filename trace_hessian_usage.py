#!/usr/bin/env python
"""Trace which Hessian values are actually used in equilibrium calculations."""

import os
import sys
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')

# Modify gpu code to print when Hessian is used in compute_phase_matrix
with open('pycalphad/gpu/minimizer.h', 'r') as f:
    content = f.read()

# Find compute_phase_matrix and add debug when Hessian is accessed
if 'compute_phase_matrix(' in content and '[GPU HESSIAN USED]' not in content:
    # Find where Hessian is used in matrix construction
    insert_pos = content.find('// Copy the Hessian block')
    if insert_pos > 0:
        debug_code = """
        // DEBUG: Print which Hessian values are being used
        if (thread_id == 0 && iteration < 3) {
            printf("[GPU HESSIAN USED] Iteration %d, Phase %d: Using H[%d,%d]=%e, H[%d,%d]=%e\\n", 
                   iteration, idx,
                   spec->num_statevars, spec->num_statevars, csst->hess[(spec->num_statevars) * csst->hess_cols + (spec->num_statevars)],
                   spec->num_statevars+1, spec->num_statevars+1, csst->hess[(spec->num_statevars+1) * csst->hess_cols + (spec->num_statevars+1)]);
        }
        """
        content = content[:insert_pos] + debug_code + content[insert_pos:]
        
        with open('pycalphad/gpu/minimizer.h', 'w') as f:
            f.write(content)
        print("✓ Added GPU Hessian usage debug")

# Create test script
test_script = """#!/usr/bin/env python
import os
import sys
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Enable debug
os.environ['PYCALPHAD_DEBUG_CATEGORIES'] = 'HESSIAN'

# Create test case
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

conditions = {v.X('TI'): 0.01, v.T: 1000, v.P: 101325}

print("=== Tracing Hessian Usage ===")

# Run GPU only
try:
    gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True)
    print("✓ GPU completed")
except Exception as e:
    print(f"✗ GPU failed: {e}")
"""

with open('trace_hessian_usage_test.py', 'w') as f:
    f.write(test_script)

print("✓ Created trace_hessian_usage_test.py")
print("\nRun: python trace_hessian_usage_test.py 2>&1 | grep -E '(FORMULAHESS INPUT|HESSIAN USED|iteration)'")
print("This will show when Hessian is calculated vs when it's used.")