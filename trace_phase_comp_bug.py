#!/usr/bin/env python
"""Trace the exact moment phase compositions change to 0.989890."""

import os
import sys
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')

# Add debug to the remove_and_consolidate_phases function
with open('pycalphad/gpu/minimizer.h', 'r') as f:
    content = f.read()

# Find where formulamoles is calculated in remove_and_consolidate_phases
if 'PHASE_COMP_BUG' not in content:
    pos = content.find('compset->phase_record->formulamole_obj(formulamoles, compset->dof);')
    if pos > 0:
        # Find the context
        start = content.rfind('__device__ void remove_and_consolidate_phases', 0, pos)
        if start > 0:
            debug_code = '''
        // PHASE_COMP_BUG DEBUG
        if (thread_id == 0 && state->iteration < 3 && idx == 0) {
            printf("[PHASE_COMP_BUG] Before formulamole_obj call:\\n");
            printf("  Phase %d DOF: [%.15e, %.15e, %.15e, %.15e, %.15e]\\n", 
                   idx, compset->dof[0], compset->dof[1], compset->dof[2], compset->dof[3], compset->dof[4]);
        }
        '''
            content = content[:pos] + debug_code + '\n        ' + content[pos:]
            
            # Add after the formulamole call
            after_pos = content.find('state->phase_compositions[idx * MAX_COMPONENTS + comp_idx] = formulamoles[comp_idx];', pos)
            if after_pos > 0:
                after_line_end = content.find('\n', after_pos)
                debug_code2 = '''
        // PHASE_COMP_BUG DEBUG continued
        if (thread_id == 0 && state->iteration < 3 && idx == 0 && comp_idx < 2) {
            printf("  formulamoles[%d] = %.15e -> phase_compositions[%d] = %.15e\\n", 
                   comp_idx, formulamoles[comp_idx], 
                   idx * MAX_COMPONENTS + comp_idx,
                   state->phase_compositions[idx * MAX_COMPONENTS + comp_idx]);
        }
                '''
                content = content[:after_line_end] + '\n' + debug_code2 + content[after_line_end:]
    
    with open('pycalphad/gpu/minimizer.h', 'w') as f:
        f.write(content)
    print("✓ Added PHASE_COMP_BUG debug")

# Create test
test_script = """
import os
import sys
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

print("=== Tracing Phase Composition Bug ===\\n")

conditions = {v.X('TI'): 0.01, v.T: 1000, v.P: 101325}

try:
    gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True)
    print(f"\\nGPU completed: X(TI) = {gpu_result.X.sel(component='TI').values.flatten()[0]:.6f}")
except Exception as e:
    print(f"GPU failed: {e}")
"""

with open('trace_phase_comp_bug_test.py', 'w') as f:
    f.write(test_script)

print("✓ Created trace_phase_comp_bug_test.py")
print("\nRun: rm -f generated_equilibrium_kernel.cu && python trace_phase_comp_bug_test.py 2>&1 | grep -A 5 'PHASE_COMP_BUG'")