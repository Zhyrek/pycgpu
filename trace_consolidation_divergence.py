#!/usr/bin/env python
"""Trace the phase consolidation process to find divergence source."""

import os
import sys
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import numpy as np

# Add more detailed debug to phase consolidation
# Modify CPU code
with open('pycalphad/core/minimizer.pyx', 'r') as f:
    cpu_content = f.read()

# Add debug after consolidation
if '[CPU CONSOLIDATION]' not in cpu_content:
    # Find the consolidation section
    insert_pos = cpu_content.find('# Consolidate phases')
    if insert_pos > 0:
        # Find the end of the consolidation block to add summary
        end_pos = cpu_content.find('# Adjust equilibrium matrix', insert_pos)
        if end_pos > 0:
            debug_code = '''
            # DEBUG: Print consolidation result
            print(f"[CPU CONSOLIDATION] After consolidation at iteration {state.iteration}:")
            print(f"  Number of phases: {len([cs for cs in compsets[:state.num_stable_phases] if cs.NP > 0])}")
            for i in range(state.num_stable_phases):
                if compsets[i].NP > 0:
                    print(f"  Phase {i}: NP={compsets[i].NP:.6f}, X(TI)={compsets[i].X[1]:.6f}")
            '''
            cpu_content = cpu_content[:end_pos] + debug_code + '\n' + cpu_content[end_pos:]
            
            with open('pycalphad/core/minimizer.pyx', 'w') as f:
                f.write(cpu_content)
            print("✓ Added CPU consolidation debug")

# Modify GPU code to add similar debug
with open('pycalphad/gpu/minimizer.h', 'r') as f:
    gpu_content = f.read()

if '[GPU CONSOLIDATION]' not in gpu_content:
    # Find after consolidation
    insert_pos = gpu_content.find('Should consolidate: YES')
    if insert_pos > 0:
        end_pos = gpu_content.find('\n', insert_pos)
        debug_code = '''
        printf("[GPU CONSOLIDATION] Consolidating phases %d and %d at iteration %d\\n", i, j, iteration);
        printf("  Phase %d: NP=%.6f, X(TI)=%.6f\\n", i, compsets[i].NP, phase_compositions[i * spec->num_components + 1]);
        printf("  Phase %d: NP=%.6f, X(TI)=%.6f\\n", j, compsets[j].NP, phase_compositions[j * spec->num_components + 1]);
        '''
        gpu_content = gpu_content[:end_pos] + '\n' + debug_code + gpu_content[end_pos:]
        
        # Also add after consolidation is done
        done_pos = gpu_content.find('compsets[i].NP = total_NP;')
        if done_pos > 0:
            end_pos = gpu_content.find('\n', done_pos)
            debug_code2 = '''
            printf("[GPU CONSOLIDATION] Result: Phase %d has NP=%.6f, X(TI)=%.6f\\n", 
                   i, compsets[i].NP, phase_compositions[i * spec->num_components + 1]);
            '''
            gpu_content = gpu_content[:end_pos] + '\n' + debug_code2 + gpu_content[end_pos:]
        
        with open('pycalphad/gpu/minimizer.h', 'w') as f:
            f.write(gpu_content)
        print("✓ Added GPU consolidation debug")

# Create test script
test_script = """#!/usr/bin/env python
import os
import sys
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Create test case
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

print("=== Tracing Phase Consolidation ===")
print("Testing with T=1000K, X(TI)=0.01\\n")

conditions = {v.X('TI'): 0.01, v.T: 1000, v.P: 101325}

# Run CPU
print("--- CPU Calculation ---")
try:
    cpu_result = equilibrium(dbf, comps, phases, conditions, gpu=False)
    print(f"✓ CPU completed: X(TI) = {cpu_result.X.sel(component='TI').values.flatten()[0]:.6f}")
except Exception as e:
    print(f"✗ CPU failed: {e}")

# Run GPU  
print("\\n--- GPU Calculation ---")
try:
    gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True)
    print(f"✓ GPU completed: X(TI) = {gpu_result.X.sel(component='TI').values.flatten()[0]:.6f}")
except Exception as e:
    print(f"✗ GPU failed: {e}")
"""

with open('trace_consolidation_test.py', 'w') as f:
    f.write(test_script)

print("\n✓ Created trace_consolidation_test.py")
print("\nRun: python trace_consolidation_test.py 2>&1 | grep -E '(CONSOLIDATION|iteration [0-2])'")
print("This will show exactly what happens during phase consolidation.")