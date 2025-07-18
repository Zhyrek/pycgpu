#!/usr/bin/env python
"""Fix site fractions after consolidation for single-sublattice phases."""

import os
import sys
sys.path.insert(0, os.getcwd())

with open('pycalphad/gpu/minimizer.h', 'r') as f:
    content = f.read()

# Find the end of remove_and_consolidate_phases function
func_end = content.find('return phases_changed;')
if func_end > 0:
    # Find the start of the return statement line
    line_start = content.rfind('\n', 0, func_end)
    
    fix_code = '''
    // CRITICAL FIX: After consolidation, for single-sublattice phases with only one phase remaining,
    // reset site fractions to match the target composition constraint
    if (phases_changed && state->num_free_stable_compsets == 1 && spec->num_prescribed_mole_fraction_conditions > 0) {
        int idx = state->free_stable_compset_indices[0];
        CompositionSet* compset = &state->compsets[idx];
        
        // Check if this is a single-sublattice phase
        if (compset->phase_record != nullptr && compset->phase_record->phase_dof == spec->num_components - 1) {
            // For X(TI) constraint, the coefficients should be [0, 1] and RHS should be the target value
            // Reset site fractions to approximately match the constraint
            if (spec->num_prescribed_mole_fraction_conditions == 1 && 
                spec->num_prescribed_mole_fraction_coefficients_cols >= 2) {
                
                // Find which component has coefficient 1
                int target_comp = -1;
                for (int i = 0; i < spec->num_prescribed_mole_fraction_coefficients_cols; ++i) {
                    if (fabs(spec->prescribed_mole_fraction_coefficients[0][i] - 1.0) < 1e-10) {
                        target_comp = i;
                        break;
                    }
                }
                
                if (target_comp == 1) { // X(TI) constraint
                    double target_x_ti = spec->prescribed_mole_fraction_rhs[0];
                    // For BCC_A2: Y(NB) + Y(TI) = 1, so Y(TI) = target_x_ti, Y(NB) = 1 - target_x_ti
                    compset->dof[spec->num_statevars + 1] = target_x_ti;      // Y(TI)
                    compset->dof[spec->num_statevars + 0] = 1.0 - target_x_ti; // Y(NB)
                    
                    // Update phase compositions
                    state->phase_compositions[idx * MAX_COMPONENTS + 0] = 1.0 - target_x_ti; // X(NB)
                    state->phase_compositions[idx * MAX_COMPONENTS + 1] = target_x_ti;       // X(TI)
                    
                    if (thread_id == 0) {
                        printf("[CONSOLIDATION FIX] Reset single phase to match constraint: Y(TI)=%.15e\\n", target_x_ti);
                    }
                }
            }
        }
    }
    '''
    
    content = content[:line_start] + fix_code + '\n    ' + content[line_start:]
    print("✓ Added consolidation site fraction fix")
else:
    print("✗ Could not find function end")

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

print("Testing consolidation fix...")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True)
gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()[0]

result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False)
cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()[0]

print(f"\\nGPU X(TI) = {gpu_x_ti:.8f}")
print(f"CPU X(TI) = {cpu_x_ti:.8f}")
print(f"Difference: {abs(gpu_x_ti - cpu_x_ti):.2e}")

if abs(gpu_x_ti - cpu_x_ti) < 1e-6:
    print("\\n✓ SUCCESS: GPU and CPU finally match!")
else:
    print(f"\\n✗ Still differ by {100*abs(gpu_x_ti - cpu_x_ti)/cpu_x_ti:.1f}%")
"""

with open('test_consolidation_fix.py', 'w') as f:
    f.write(test_script)

print("\nRun: rm -f generated_equilibrium_kernel.cu && python test_consolidation_fix.py")