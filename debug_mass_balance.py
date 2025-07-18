#!/usr/bin/env python
"""Debug mass balance constraint enforcement."""

import os
import sys
sys.path.insert(0, os.getcwd())

# Add debug to check mass balance after consolidation
with open('pycalphad/gpu/minimizer.h', 'r') as f:
    content = f.read()

# Find the prescribed composition constraint calculation
prescribed_pos = content.find('rhs_ptr[constraint_idx] = spec->prescribed_mole_fractions[prescribed_comp_idx] - state_X[prescribed_comp_idx];')
if prescribed_pos > 0:
    # Add debug before this line
    debug_code = '''
            // DEBUG: Mass balance constraint
            if (thread_id == 0 && state->iteration < 5) {
                printf("[MASS BALANCE] Constraint %d: target X[%d] = %.15e, current = %.15e, error = %.15e\\n",
                       constraint_idx, prescribed_comp_idx, 
                       spec->prescribed_mole_fractions[prescribed_comp_idx],
                       state_X[prescribed_comp_idx],
                       spec->prescribed_mole_fractions[prescribed_comp_idx] - state_X[prescribed_comp_idx]);
            }
            '''
    content = content[:prescribed_pos] + debug_code + '\n            ' + content[prescribed_pos:]
    print("✓ Added mass balance debug")

with open('pycalphad/gpu/minimizer.h', 'w') as f:
    f.write(content)

# Test script
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

print("Testing mass balance debug...")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True)
gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()[0]
print(f"\\nFinal GPU X(TI) = {gpu_x_ti:.8f}")
print(f"Target X(TI) = 0.01000000")
print(f"Error = {gpu_x_ti - 0.01:.2e}")
"""

with open('test_mass_balance.py', 'w') as f:
    f.write(test_script)

print("\nRun: rm -f generated_equilibrium_kernel.cu && python test_mass_balance.py 2>&1 | grep 'MASS BALANCE' | head -20")