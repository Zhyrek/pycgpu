#!/usr/bin/env python3
"""Check what state variables are actually used in workspace"""

import sys
sys.path.insert(0, '.')

from pycalphad import Database, Model, variables as v
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.core.utils import get_state_variables

# Load database and create model
db = Database('NbTi.tdb')
model = Model(db, ['NB', 'TI', 'VA'], 'BCC_A2')

# Typical equilibrium conditions
conditions = {v.T: 1000, v.P: 101325, v.N: 1}

print("=== State Variables Analysis ===")
print(f"Model state variables: {model.state_variables}")
print(f"Conditions: {conditions}")

# Get state variables like workspace does
state_vars = get_state_variables(models={'BCC_A2': model}, conds=conditions)
print(f"\nget_state_variables returns: {state_vars}")
print(f"Sorted state variables: {sorted(state_vars, key=str)}")

# Create phase record factory
prf = PhaseRecordFactory(db, ['NB', 'TI', 'VA'], conditions, {'BCC_A2': model})
print(f"\nPhaseRecordFactory state_variables: {prf.state_variables}")

# So the GPU should be using [N, P, T] ordering, not [T]
print("\n=== GPU Variable Ordering ===")
print("GPU should use workspace state variables [N, P, T] + site fractions")
print("This gives indices: N=0, P=1, T=2, Y_NB=3, Y_TI=4")

# Now check what the GPU code generation actually does
from pycalphad.gpu.gpu_codegen import notebook_get_all_syms_for_model

# Create minimal workspace  
class MinimalWorkspace:
    def __init__(self, prf):
        self.components = ['NB', 'TI', 'VA']
        self.phase_record_factory = prf
        
wks = MinimalWorkspace(prf)

gpu_syms = notebook_get_all_syms_for_model(model, wks)
print(f"\nGPU notebook_get_all_syms_for_model returns: {gpu_syms}")
print(f"GPU uses {len(gpu_syms)} variables total")