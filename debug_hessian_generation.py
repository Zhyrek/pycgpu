#!/usr/bin/env python3
"""Debug Hessian generation to see if it's a list or single expression"""

from pycalphad import Database
from pycalphad.model import Model
from pycalphad.core.workspace import Workspace
import pycalphad.variables as v

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']

# Create model
model = Model(db, comps, 'BCC_A2')

print("Model.G type:", type(model.G))
print("Is it a list?", isinstance(model.G, (list, tuple)))

# Check what notebook_source_from_expr receives
print("\nIn _nb_formulahess_from_model, model.G is passed as expr_or_list_in")
print("This will be a single expression, not a list")
print("\nSo the fix is being applied in the SINGLE expression case")
print("But the single expression case does NOT call fix_hessian_spurious_terms!")

print("\nThe fix needs to be added to the single expression Hessian case")