#!/usr/bin/env python
"""Compare the actual matrix construction between GPU and CPU."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("COMPARING GPU vs CPU MATRIX CONSTRUCTION")
print("=" * 60)

# We need to run the GPU code with debug output to see what matrix it constructs
# when it gets stuck at X(TI)=0.903147

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("From our analysis, we know:")
print("1. CPU constructs matrix with constraint row: [0, 0, 0.903147]")
print("2. CPU constructs RHS: [0, 0, -0.003147] (negative change needed)")
print("3. This gives solution: [0, 0, -0.00348448]")
print("4. Which updates phase amount: 1.0 + (-0.00348448) = 0.9965155")
print("5. Final X(TI): 0.9965155 * 0.903147 = 0.900000")

print(f"\n" + "="*60)
print("HYPOTHESIS: GPU CONSTRUCTS WRONG RHS")

print("\nThe GPU might be constructing RHS as:")
print("  RHS = target - residual = 0.9 - 0.003147 = 0.896853")
print("  This gives wrong solution leading to wrong final composition")

# Test what happens with the wrong RHS (what GPU might be doing)
wrong_matrix = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0], 
    [0.0, 0.0, 0.903147]
])

wrong_rhs = np.array([0.0, 0.0, 0.896853])  # GPU's likely construction

wrong_solution = np.linalg.solve(wrong_matrix, wrong_rhs)
wrong_new_phase_amount = 1.0 + wrong_solution[2]
wrong_final_X_TI = wrong_new_phase_amount * 0.903147

print(f"\nWith wrong RHS construction:")
print(f"  Solution: {wrong_solution}")
print(f"  New phase amount: {wrong_new_phase_amount:.10f}")
print(f"  Final X(TI): {wrong_final_X_TI:.10f}")
print(f"  Error: This is way off target!")

print(f"\n" + "="*60)
print("THE ACTUAL ISSUE")

print("\nThe GPU is likely constructing the RHS incorrectly.")
print("It probably does: RHS = target_composition")
print("But it should do: RHS = desired_change_in_composition")

print("\nCorrect approach:")
print("  Current overall X(TI): 1.0 * 0.903147 = 0.903147")
print("  Target overall X(TI): 0.9") 
print("  Required change: 0.9 - 0.903147 = -0.003147")
print("  RHS should be: [0, 0, -0.003147]")

print("\nWrong approach (what GPU likely does):")
print("  RHS = [0, 0, target] = [0, 0, 0.9]")
print("  Or RHS = [0, 0, target - current_residual]")

print(f"\n" + "="*60)
print("CONCLUSION")

print("\nThe root cause is likely in the GPU's RHS construction.")
print("The GPU and CPU linear solvers work identically.")
print("But the GPU constructs the wrong RHS for the mass balance constraint.")

print("\nTo fix this, we need to check how the GPU constructs equilibrium_rhs")
print("in the mass balance constraint section of the minimizer code.")

print("\nSpecifically look for where the GPU sets:")
print("  equilibrium_rhs[constraint_row] = ???")
print("  And compare with the CPU's approach in minimizer.pyx")

# Let's run a quick test to confirm GPU gets stuck at 0.903147
print(f"\n" + "="*60)
print("CONFIRMING GPU BEHAVIOR")

try:
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()
    gpu_np = result_gpu.NP.values.flatten()
    
    gpu_overall = sum(np * x for np, x in zip(gpu_np, gpu_x_ti) if np > 1e-12)
    print(f"GPU result: X(TI) = {gpu_overall:.10f}")
    
    if abs(gpu_overall - 0.903147) < 0.0001:
        print("✓ Confirmed: GPU gets stuck at ~0.903147")
        print("✓ This matches our hypothesis about wrong RHS construction")
    else:
        print(f"✗ GPU gives different result: {gpu_overall:.10f}")
        
except Exception as e:
    print(f"GPU test failed: {e}")

print(f"\n" + "="*60)
print("NEXT STEPS")
print("1. Examine GPU minimizer RHS construction for mass balance")
print("2. Fix RHS to use desired_change instead of target_value")
print("3. Test that fix resolves the convergence issue")