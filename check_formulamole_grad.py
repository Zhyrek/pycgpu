#!/usr/bin/env python3
"""Check the generated formulamole_grad functions"""

import sys
sys.path.insert(0, '.')

from pycalphad import Database, Model, variables as v
from pycalphad.core.workspace import Workspace
from pycalphad.gpu.gpu_codegen import get_energy_hess_gradient_functions

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']

# Create workspace to get state variables  
conditions = {v.N: 1, v.P: 101325, v.T: 1000}
wks = Workspace(db, comps, phases, conditions)

# Get energy functions for BCC_A2
phase = 'BCC_A2'
model = Model(db, comps, phase)

# Get the functions
funcs = get_energy_hess_gradient_functions(
    wks.phase_record_factory, db, comps, phase, model, 
    {}, {},  # No custom parameters
    include_hess=False,
    include_internal_cons=False,
    include_formulamole=True
)

print("\n=== FORMULAMOLE_GRAD FUNCTIONS ===")
for comp in ['NB', 'TI']:
    if f'formulamole_grad_{comp}' in funcs:
        print(f"\n{comp}:")
        print(funcs[f'formulamole_grad_{comp}'])