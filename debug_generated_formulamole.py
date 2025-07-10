#!/usr/bin/env python3
"""Generate and examine the formulamole_grad function"""
from pycalphad import Database, Model, Workspace

# Load database and create model
db = Database('NbTi.tdb')
mod = Model(db, ['NB', 'TI', 'VA'], 'BCC_A2')

# Create workspace to get proper variable ordering
wks = Workspace(db, ['NB', 'TI', 'VA'], ['BCC_A2'], {})

# Import the code generation functions
from pycalphad.gpu.gpu_codegen import _nb_formulamole_grad_from_model

# Generate the formulamole_grad function
print("=== Generating formulamole_grad function ===")
code = _nb_formulamole_grad_from_model(mod, 0, wks, verbose=True)

print("\n=== Generated C code ===")
print(code)

# Let's also manually trace through what should happen
print("\n=== Manual calculation ===")
print("For BCC_A2 phase:")
print("- moles(NB) = Y(BCC_A2,0,NB)")
print("- moles(TI) = Y(BCC_A2,0,TI)")
print("\nDerivatives:")
print("- d(moles_NB)/d(Y_NB) = 1, d(moles_NB)/d(Y_TI) = 0")
print("- d(moles_TI)/d(Y_NB) = 0, d(moles_TI)/d(Y_TI) = 1")
print("\nIn the flat output array:")
print("- Indices 0-4: derivatives of moles_NB w.r.t. [N, P, T, Y_NB, Y_TI] = [0, 0, 0, 1, 0]")
print("- Indices 5-9: derivatives of moles_TI w.r.t. [N, P, T, Y_NB, Y_TI] = [0, 0, 0, 0, 1]")