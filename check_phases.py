#!/usr/bin/env python
"""Check which phases are present in the test case."""

from pycalphad import Database, calculate, equilibrium, variables as v
from pycalphad.core.workspace import Workspace
import numpy as np

# Load database and set up system
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = list(db.phases.keys())

print("Phases in database:", phases)

# Set up conditions for test
T = 1300.0
conditions = {v.T: T, v.P: 101325, v.X('TI'): 0.5}

# Run calculate to create workspace
calc_result = calculate(db, comps, phases, T=T, P=101325, N=1, output='GM')

# Create workspace
wks = Workspace(db, comps, phases, conditions, calc_result, 
                parameters={}, phase_record_factory=None)

print("\nPhases in workspace:", wks.phases)
print("Number of unique phases:", len(wks.phases))

# Check models
print("\nModels in workspace:")
for phase_name, model in wks.models.items():
    print(f"  {phase_name}: {model}")
    
# Check if phase models generate Hessians
from pycalphad.model import Model
print("\nChecking Hessian availability:")
for phase_name in wks.phases:
    model = Model(db, comps, phase_name)
    print(f"  {phase_name}: G = {model.G}")
    try:
        # Test if Hessian can be computed
        from symengine import symbols, diff
        vars = model.variables
        if len(vars) > 0:
            # Try to compute second derivative
            g_expr = model.G
            first_deriv = diff(g_expr, vars[0])
            second_deriv = diff(first_deriv, vars[0])
            print(f"    Can compute Hessian: Yes")
        else:
            print(f"    Can compute Hessian: No variables")
    except Exception as e:
        print(f"    Can compute Hessian: No ({e})")