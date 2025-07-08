import numpy as np
from pycalphad import Database, equilibrium, variables as v

# Setup
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']
conds = {v.T: 1800, v.P: 101325, v.X('TI'): 0.3}

print("=== Testing GPU Equilibrium with Hessian Debug ===")

# First, let's add debug output to the GPU minimizer
# We need to modify the GPU code to print Hessian values

# Read the current minimizer.h
with open('pycalphad/gpu/minimizer.h', 'r') as f:
    minimizer_code = f.read()

# Check if we already have Hessian debug output
if 'GPU DEBUG: Hessian calculated' not in minimizer_code:
    print("Adding Hessian debug output to GPU minimizer...")
    
    # Find the location where Hessian is calculated
    hess_calc_pos = minimizer_code.find('pr->formulahess(csst->hess, model_dof_for_calcs);')
    
    if hess_calc_pos != -1:
        # Find the end of the line
        line_end = minimizer_code.find('\n', hess_calc_pos)
        
        # Insert debug output after the Hessian calculation
        debug_code = '''
        // DEBUG: Print Hessian values
        if (condition_idx == 0 && csst->phase_idx < 2) {
            printf("GPU DEBUG: Hessian calculated for phase %d\\n", csst->phase_idx);
            printf("  Hessian size: %dx%d\\n", phase_dof + NUM_STATEVARS, phase_dof + NUM_STATEVARS);
            if (phase_dof > 0) {
                printf("  Site fraction Hessian block:\\n");
                for (int i = 0; i < phase_dof; i++) {
                    printf("    [%d]", i);
                    for (int j = 0; j < phase_dof; j++) {
                        int idx = (NUM_STATEVARS + i) * (phase_dof + NUM_STATEVARS) + (NUM_STATEVARS + j);
                        printf(" %e", csst->hess[idx]);
                    }
                    printf("\\n");
                }
            }
        }
'''
        
        new_minimizer = minimizer_code[:line_end] + debug_code + minimizer_code[line_end:]
        
        # Write the modified file
        with open('pycalphad/gpu/minimizer.h', 'w') as f:
            f.write(new_minimizer)
        
        print("Added Hessian debug output to minimizer.h")
    else:
        print("Could not find Hessian calculation in minimizer.h")
else:
    print("Hessian debug output already present in minimizer.h")

# Now run the equilibrium calculation
print("\nRunning GPU equilibrium with debug output...")
eq_gpu = equilibrium(dbf, comps, phases, conds, calc_opts={'pdens': 10}, gpu=True)
print(f"\nGPU GM: {float(eq_gpu.GM.values[0]):.6f}")

# Run CPU for comparison
print("\nRunning CPU equilibrium...")
eq_cpu = equilibrium(dbf, comps, phases, conds, calc_opts={'pdens': 10})
print(f"CPU GM: {float(eq_cpu.GM.values[0]):.6f}")

print(f"\nGM Difference: {abs(eq_cpu.GM.values[0] - eq_gpu.GM.values[0]):.6e} J/mol")
if abs(eq_cpu.GM.values[0] - eq_gpu.GM.values[0]) > 0.001:
    print("ERROR: Difference exceeds 0.001 J/mol threshold!")