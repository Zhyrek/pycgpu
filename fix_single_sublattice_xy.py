#!/usr/bin/env python
"""Fix X = Y constraint for single-sublattice phases."""

import os
import sys
sys.path.insert(0, os.getcwd())

with open('pycalphad/gpu/minimizer.h', 'r') as f:
    content = f.read()

# After the site fraction update in the solver, we need to ensure X = Y for single-sublattice phases
# Find where site fractions are updated
update_pos = content.find('compset->dof[spec->num_statevars + i] = new_y_for_phase[i];')
if update_pos > 0:
    # Find the end of the update loop
    loop_end = content.find('}', update_pos)
    if loop_end > 0:
        # Find the next line after the loop
        next_line = content.find('\n', loop_end) + 1
        
        fix_code = '''
        // CRITICAL FIX: For single-sublattice phases, ensure X = Y
        // This is required because formulamole_obj returns X values, not Y values
        if (compset->phase_record != nullptr && compset->phase_record->phase_dof == spec->num_components - 1) {
            // Single sublattice phase - recompute phase compositions
            double formulamoles[MAX_COMPONENTS];
            for (int i = 0; i < MAX_COMPONENTS; ++i) formulamoles[i] = 0.0;
            
            if (compset->phase_record->formulamole_obj != nullptr) {
                compset->phase_record->formulamole_obj(formulamoles, compset->dof);
            }
            
            // Update phase compositions to match site fractions
            for (int comp_idx = 0; comp_idx < spec->num_components; ++comp_idx) {
                state->phase_compositions[idx * MAX_COMPONENTS + comp_idx] = formulamoles[comp_idx];
            }
        }
        '''
        
        content = content[:next_line] + fix_code + '\n' + content[next_line:]
        print("✓ Added X = Y fix for single-sublattice phases")
else:
    print("✗ Could not find site fraction update location")

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

print("Testing GPU with X = Y fix...")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True)
gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()[0]

result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False)
cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()[0]

print(f"\\nGPU X(TI) = {gpu_x_ti:.8f}")
print(f"CPU X(TI) = {cpu_x_ti:.8f}")
print(f"Difference: {abs(gpu_x_ti - cpu_x_ti):.2e}")

if abs(gpu_x_ti - cpu_x_ti) < 1e-6:
    print("\\n✓ SUCCESS: GPU and CPU match!")
else:
    print(f"\\n✗ Still differ by {100*abs(gpu_x_ti - cpu_x_ti)/cpu_x_ti:.1f}%")
"""

with open('test_xy_fix.py', 'w') as f:
    f.write(test_script)

print("\nRun: rm -f generated_equilibrium_kernel.cu && python test_xy_fix.py")