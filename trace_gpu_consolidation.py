#!/usr/bin/env python
"""Trace exactly what happens during GPU consolidation."""

import os
import sys
sys.path.insert(0, os.getcwd())

# Add detailed debug to consolidation
with open('pycalphad/gpu/minimizer.h', 'r') as f:
    content = f.read()

# Add debug after consolidation to see what phase remains
debug_pos = content.find('state->phase_amt[idx1] = fmax(total_amt, 1e-8);')
if debug_pos > 0:
    # Find the end of the line
    line_end = content.find('\n', debug_pos)
    debug_code = '''
                
                // DEBUG: What happens after consolidation
                if (thread_id == 0 && state->iteration < 2) {
                    printf("[CONSOLIDATION] Consolidated phases %d and %d:\\n", idx1, idx2);
                    printf("  Phase %d: amount=%.15e, X=[%.15e, %.15e]\\n", 
                           idx1, state->phase_amt[idx1],
                           state->phase_compositions[idx1 * MAX_COMPONENTS + 0],
                           state->phase_compositions[idx1 * MAX_COMPONENTS + 1]);
                    printf("  Phase %d: amount=%.15e (removed)\\n", idx2, state->phase_amt[idx2]);
                    
                    // Also show site fractions
                    CompositionSet* cs1 = &state->compsets[idx1];
                    printf("  Phase %d site fractions: Y=[%.15e, %.15e]\\n",
                           idx1, cs1->dof[3], cs1->dof[4]);
                }'''
    
    content = content[:line_end] + debug_code + content[line_end:]
    
    with open('pycalphad/gpu/minimizer.h', 'w') as f:
        f.write(content)
    print("✓ Added consolidation debug")

# Create test script
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

print("Testing GPU consolidation...")
result = equilibrium(dbf, comps, phases, conditions, gpu=True)
print(f"\\nGPU X(TI) = {result.X.sel(component='TI').values.flatten()[0]:.8f}")
"""

with open('test_consolidation.py', 'w') as f:
    f.write(test_script)

print("\nRun: rm -f generated_equilibrium_kernel.cu && python test_consolidation.py 2>&1 | grep -E '(CONSOLIDATION|iteration)' -A5")