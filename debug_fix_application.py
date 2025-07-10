#!/usr/bin/env python3
"""Debug why fix_hessian_spurious_terms is not working"""

from pycalphad import Database
from pycalphad.core.workspace import Workspace
from pycalphad.model import Model
from pycalphad.gpu.gpu_codegen import notebook_source_from_expr
import pycalphad.variables as v

# Load database and create workspace
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']

# Create models
models = {}
for phase in phases:
    models[phase] = Model(db, comps, phase)

# Create workspace
conditions = {v.T: 300, v.P: 101325}
wks = Workspace(db, comps, phases, conditions, models=models, phase_record_factory=None)

print("=== Debugging fix_hessian_spurious_terms ===\n")

# The issue is that out[18] corresponds to hess[3,3] in the flattened array
# For a 5x5 Hessian (T, P, N, Y_NB, Y_TI), the indices are:
# Row 0: 0, 1, 2, 3, 4
# Row 1: 5, 6, 7, 8, 9
# Row 2: 10, 11, 12, 13, 14
# Row 3: 15, 16, 17, 18, 19
# Row 4: 20, 21, 22, 23, 24

# So out[18] is row 3, col 3 (0-indexed), which is the Y_NB diagonal
# And out[24] is row 4, col 4 (0-indexed), which is the Y_TI diagonal

print("Index mapping for 5x5 Hessian:")
print("out[18] = hess[3,3] = d²G/dY_NB² (should have no pow(x[4], (-1)))")
print("out[24] = hess[4,4] = d²G/dY_TI² (should have no pow(x[3], (-1)))")
print()

# In the model DOF space, the indices are different:
# Model has [T, Y_NB, Y_TI] so it's 3x3
# Row 0: 0, 1, 2 (T derivatives)
# Row 1: 3, 4, 5 (Y_NB derivatives)
# Row 2: 6, 7, 8 (Y_TI derivatives)

print("In model DOF space (3x3):")
print("Index 4 = hess[1,1] = d²G/dY_NB²")
print("Index 8 = hess[2,2] = d²G/dY_TI²")
print()

print("The fix_hessian_spurious_terms function is called with i_sym_idx and j_sym_idx")
print("which are indices in the ordered_symbols_for_diff list.")
print()

# Check what ordered_symbols_for_diff contains
model = models['BCC_A2']
print("Model variables:", model.variables)
print("These map to indices: T=0, Y_NB=1, Y_TI=2 in the model space")
print()

print("So when i_sym_idx=1, j_sym_idx=1, we're processing d²G/dY_NB²")
print("The fix should remove pow(x[4], (-1)) terms (x[4] is Y_TI)")
print()

print("But in the workspace DOF, x[3]=Y_NB and x[4]=Y_TI")
print("So the fix is looking for the right pattern.")
print()

print("The issue might be that the pattern doesn't match exactly what's in the code.")