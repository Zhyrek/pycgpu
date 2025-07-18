#!/usr/bin/env python
"""Debug the formulamole calculation to see why it's wrong."""

import os
import sys
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')

# Add debug to GPU formulamole calculation
with open('pycalphad/gpu/minimizer.h', 'r') as f:
    content = f.read()

# Find where formulamole is called and add debug
if 'DEBUG formulamole calculation' not in content:
    # Find the formulamole call
    pos = content.find('compset->phase_record->formulamole_obj(formulamoles, compset->dof);')
    if pos > 0:
        debug_code = '''
        // DEBUG formulamole calculation
        if (thread_id == 0 && state->iteration < 3 && idx == 0) {
            printf("[GPU FORMULAMOLE DEBUG] Iteration %d, Phase %d\\n", state->iteration, idx);
            printf("  Input DOF: [", );
            for (int k = 0; k < 5; k++) {
                printf("%.15e ", compset->dof[k]);
            }
            printf("]\\n");
            printf("  Calling formulamole_obj...\\n");
        }
        '''
        content = content[:pos] + debug_code + '\n        ' + content[pos:]
        
        # Add after the call too
        after_pos = content.find('formulamoles[comp_idx];', pos)
        if after_pos > 0:
            after_pos = content.find('\n', after_pos)
            debug_code2 = '''
        // DEBUG: Print formulamole results
        if (thread_id == 0 && state->iteration < 3 && idx == 0) {
            printf("  Formulamole result: [%.15e, %.15e]\\n", formulamoles[0], formulamoles[1]);
            printf("  This will be stored as phase_compositions\\n");
        }
            '''
            content = content[:after_pos] + debug_code2 + content[after_pos:]
    
    with open('pycalphad/gpu/minimizer.h', 'w') as f:
        f.write(content)
    print("✓ Added formulamole debug")

# Also check what the generated formulamole function looks like
print("\nChecking generated formulamole function...")

# Create a test to examine the function
test_script = """
import os
import sys
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')

from pycalphad import Database
from pycalphad.core.utils import filter_phases
from pycalphad.gpu.gpu_codegen import _generate_phase_records
import numpy as np

# Create test case
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Generate phase records to see the formulamole function
phase_records, models = _generate_phase_records(dbf, comps, phases, {}, (('NB', 'TI'), 'VA'), None)

# Check BCC_A2 formulamole
for pr, model in zip(phase_records, models):
    if pr['phase_name'] == 'BCC_A2':
        print(f"\\nBCC_A2 formulamole function:")
        print(f"Number of elements: {model.components}")
        print(f"Site ratios: {model._site_ratios}")
        
        # Test the function with the DOF values we see
        dof = np.array([1.0, 101325.0, 1000.0, 0.991525, 0.008475])
        
        # The mole fractions should be calculated as:
        # For a single sublattice with NB and TI:
        # moles(NB) = Y(NB) * site_ratio
        # moles(TI) = Y(TI) * site_ratio
        # X(NB) = moles(NB) / (moles(NB) + moles(TI))
        # X(TI) = moles(TI) / (moles(NB) + moles(TI))
        
        y_nb = dof[3]
        y_ti = dof[4]
        print(f"\\nInput site fractions: Y(NB)={y_nb:.6f}, Y(TI)={y_ti:.6f}")
        print(f"Sum of site fractions: {y_nb + y_ti:.6f}")
        
        # For single sublattice, X should equal Y
        print(f"\\nExpected X(NB) = {y_nb:.6f}")
        print(f"Expected X(TI) = {y_ti:.6f}")
        
        # But we're seeing X(NB)=0.989890, X(TI)=0.010110
        # Let's check if this is a normalization issue
        sum_expected = 0.991525 + 0.008475  # = 1.0
        sum_observed = 0.989890 + 0.010110  # = 1.0
        
        print(f"\\nObserved X(NB) = 0.989890, X(TI) = 0.010110")
        print(f"Sum of observed: {sum_observed:.6f}")
        
        # The pattern suggests the issue might be in the DOF indexing
        # or in how the dependent site fraction is handled
        break
"""

with open('debug_formulamole_test.py', 'w') as f:
    f.write(test_script)

print("✓ Created debug_formulamole_test.py")
print("\nRun the following commands:")
print("1. python debug_formulamole_test.py  # To check the formulamole logic")
print("2. rm -f generated_equilibrium_kernel.cu && python test_cpu_gpu_comparison.py 2>&1 | grep 'FORMULAMOLE DEBUG' -A 5")