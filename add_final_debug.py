#!/usr/bin/env python
"""Add debug to see final equilibrium values."""

import os
import sys
sys.path.insert(0, os.getcwd())

with open('pycalphad/gpu/eqsolver.h', 'r') as f:
    content = f.read()

# Find where results are stored
store_pos = content.find('result->converged = converged;')
if store_pos > 0:
    debug_code = '''
    // DEBUG: Final values
    if (thread_id == 0) {
        printf("\\n[GPU FINAL VALUES]\\n");
        printf("  Converged: %s\\n", converged ? "true" : "false");
        printf("  Number of stable phases: %d\\n", current_sys_state.num_free_stable_compsets);
        for (int i = 0; i < current_sys_state.num_free_stable_compsets; ++i) {
            int idx = current_sys_state.free_stable_compset_indices[i];
            printf("  Phase %d:\\n", idx);
            printf("    Amount: %.15e\\n", current_sys_state.phase_amt[idx]);
            printf("    X(NB): %.15e\\n", current_sys_state.phase_compositions[idx * MAX_COMPONENTS + 0]);
            printf("    X(TI): %.15e\\n", current_sys_state.phase_compositions[idx * MAX_COMPONENTS + 1]);
            CompositionSet* cs = &current_sys_state.compsets[idx];
            printf("    Y(NB): %.15e\\n", cs->dof[current_spec.num_statevars + 0]);
            printf("    Y(TI): %.15e\\n", cs->dof[current_spec.num_statevars + 1]);
        }
        printf("  System mole fractions: X(NB)=%.15e, X(TI)=%.15e\\n",
               current_sys_state.mole_fractions[0], current_sys_state.mole_fractions[1]);
    }
    '''
    content = content[:store_pos] + debug_code + '\n    ' + content[store_pos:]
    print("✓ Added final values debug")

with open('pycalphad/gpu/eqsolver.h', 'w') as f:
    f.write(content)

print("\nRun: rm -f generated_equilibrium_kernel.cu && python test_consolidation_fix.py 2>&1 | grep -A20 'GPU FINAL VALUES'")