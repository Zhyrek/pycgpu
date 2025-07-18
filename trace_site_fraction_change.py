#!/usr/bin/env python
"""Trace when site fractions change to 0.989890."""

import os
import sys
sys.path.insert(0, os.getcwd())

# Add debug before and after key operations
with open('pycalphad/gpu/minimizer.h', 'r') as f:
    content = f.read()

# Add debug before consolidation check
check_pos = content.find('// Check for phases that need to be consolidated')
if check_pos > 0:
    debug_before = '''
    // DEBUG: Site fractions before consolidation loop
    if (thread_id == 0 && state->iteration < 2) {
        for (int i = 0; i < state->num_free_stable_compsets; ++i) {
            int idx = state->free_stable_compset_indices[i];
            CompositionSet* cs = &state->compsets[idx];
            printf("[BEFORE CONSOLIDATION] Phase %d Y=[%.15e, %.15e]\\n",
                   idx, cs->dof[3], cs->dof[4]);
        }
    }
    '''
    content = content[:check_pos] + debug_before + '\n    ' + content[check_pos:]

# Add debug inside the consolidation condition
inside_pos = content.find('if (should_consolidate) {')
if inside_pos > 0:
    # Find the opening brace
    brace_pos = content.find('{', inside_pos)
    debug_inside = '''
                
                // DEBUG: Site fractions at consolidation moment
                if (thread_id == 0 && state->iteration < 2) {
                    CompositionSet* cs1 = &state->compsets[idx1];
                    CompositionSet* cs2 = &state->compsets[idx2];
                    printf("[AT CONSOLIDATION] Phase %d Y=[%.15e, %.15e]\\n",
                           idx1, cs1->dof[3], cs1->dof[4]);
                    printf("[AT CONSOLIDATION] Phase %d Y=[%.15e, %.15e]\\n",
                           idx2, cs2->dof[3], cs2->dof[4]);
                }'''
    content = content[:brace_pos+1] + debug_inside + content[brace_pos+1:]

# Add debug after phases_changed check
after_pos = content.find('if (phases_changed) {')
if after_pos > 0:
    # Find the end of the function
    func_end = content.find('return phases_changed;', after_pos)
    if func_end > 0:
        debug_after = '''
    
    // DEBUG: Site fractions after consolidation
    if (thread_id == 0 && state->iteration < 2 && phases_changed) {
        printf("[AFTER CONSOLIDATION] phases_changed=true\\n");
        for (int i = 0; i < state->num_free_stable_compsets; ++i) {
            int idx = state->free_stable_compset_indices[i];
            CompositionSet* cs = &state->compsets[idx];
            printf("[AFTER CONSOLIDATION] Phase %d Y=[%.15e, %.15e]\\n",
                   idx, cs->dof[3], cs->dof[4]);
        }
    }
    '''
        content = content[:func_end] + debug_after + '\n    ' + content[func_end:]

with open('pycalphad/gpu/minimizer.h', 'w') as f:
    f.write(content)

print("✓ Added site fraction change debug")
print("\nRun: rm -f generated_equilibrium_kernel.cu && python test_consolidation.py 2>&1 | grep -E '(BEFORE|AT|AFTER) CONSOLIDATION' -A1")