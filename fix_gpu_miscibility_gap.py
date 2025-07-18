#!/usr/bin/env python
"""Fix GPU handling of miscibility gaps."""

import os
import sys
sys.path.insert(0, os.getcwd())

# The issue is that when we have two instances of the same phase (miscibility gap),
# the GPU solver incorrectly merges their site fractions during iteration.
# We need to prevent this.

# For now, let's add a check to detect when we have multiple instances of the same phase
# and handle them more carefully.

with open('pycalphad/gpu/minimizer.h', 'r') as f:
    content = f.read()

# Find where we check for phase consolidation
consolidation_check = content.find('for (int comp_idx = 0; comp_idx < spec->num_components; ++comp_idx) {')
if consolidation_check > 0:
    # Find the should_consolidate = true line before this
    should_cons_pos = content.rfind('bool should_consolidate = true;', 0, consolidation_check)
    if should_cons_pos > 0:
        # Add a check for same phase model
        new_code = '''bool should_consolidate = true;
            
            // Check if these are the same phase model (miscibility gap case)
            bool same_phase_model = (compset1->phase_record == compset2->phase_record);
            if (same_phase_model && state->iteration < 2) {
                // For miscibility gaps, be more conservative about consolidation in early iterations
                // This prevents premature merging of phases that should remain distinct
                COMPSET_CONSOLIDATE_DISTANCE = 1e-6;  // Tighter tolerance
            }
            '''
        content = content[:should_cons_pos] + new_code + content[should_cons_pos + len('bool should_consolidate = true;'):]
        print("✓ Added miscibility gap handling")

with open('pycalphad/gpu/minimizer.h', 'w') as f:
    f.write(content)

# Also add a simpler test case to verify
test_script = """
import os
import sys
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

conditions = {v.X('TI'): 0.01, v.T: 1000, v.P: 101325}

print("Testing GPU with miscibility gap fix...")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True)
gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()[0]

result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False)
cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()[0]

print(f"GPU X(TI) = {gpu_x_ti:.8f}")
print(f"CPU X(TI) = {cpu_x_ti:.8f}")
print(f"Difference: {abs(gpu_x_ti - cpu_x_ti):.2e}")

if abs(gpu_x_ti - cpu_x_ti) < 1e-6:
    print("✓ SUCCESS: GPU and CPU match!")
else:
    print(f"✗ FAIL: Still differ by {100*abs(gpu_x_ti - cpu_x_ti)/cpu_x_ti:.2f}%")
"""

with open('test_miscibility_fix.py', 'w') as f:
    f.write(test_script)

print("\nRun: rm -f generated_equilibrium_kernel.cu && python test_miscibility_fix.py")