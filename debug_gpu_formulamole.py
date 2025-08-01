#!/usr/bin/env python3
"""Debug GPU formulamole generation for FCC_A1 phase"""

import numpy as np
from pycalphad import Database, Model
from pycalphad.core.workspace import Workspace
from pycalphad.gpu.gpu_codegen import _nb_formulamole_obj_from_model, _nb_formulamole_grad_from_model
import pycalphad.variables as v

# Initialize database and system
db = Database('AuBi-07Wan.tdb')
phases = ['FCC_A1']
comps = ['AU', 'BI', 'VA']

# Create workspace
wks = Workspace(db, comps, phases, {v.T: 600, v.P: 101325, v.X('BI'): 0.3})

# Get the FCC_A1 model
model_fcc = Model(db, comps, 'FCC_A1')

print("FCC_A1 Phase Model:")
print(f"  Sublattices: {model_fcc.site_fractions}")
print(f"  Site ratios: {model_fcc.site_ratios}")
print(f"  Components: {model_fcc.components}")
print(f"  Nonvacant elements: {model_fcc.nonvacant_elements}")

# Generate formulamole_obj code
print("\n" + "="*80)
print("Generated formulamole_obj code:")
print("="*80)
formulamole_obj_code = _nb_formulamole_obj_from_model(model_fcc, 0, wks, validate=False, verbose=True)
print(formulamole_obj_code)

# Generate formulamole_grad code
print("\n" + "="*80)
print("Generated formulamole_grad code:")
print("="*80)
formulamole_grad_code = _nb_formulamole_grad_from_model(model_fcc, 0, wks, validate=False, verbose=True)
print(formulamole_grad_code)

# Now let's manually check the moles calculation
print("\n" + "="*80)
print("Manual moles calculation:")
print("="*80)

# For FCC_A1 with site fractions Y(FCC_A1,0,AU)=0.7, Y(FCC_A1,0,BI)=0.3, Y(FCC_A1,1,VA)=1.0
print("\nCase 1: Y(FCC_A1,0,AU)=0.7, Y(FCC_A1,0,BI)=0.3, Y(FCC_A1,1,VA)=1.0")
print(f"  moles(AU) = {model_fcc.moles('AU', per_formula_unit=True)}")
print(f"  moles(BI) = {model_fcc.moles('BI', per_formula_unit=True)}")
print(f"  moles(VA) should be 0.0")

# Calculate actual values
y_au = 0.7
y_bi = 0.3
y_va = 1.0

# Using the model's expressions
subs = {
    v.SiteFraction('FCC_A1', 0, v.Species('AU')): y_au,
    v.SiteFraction('FCC_A1', 0, v.Species('BI')): y_bi,
    v.SiteFraction('FCC_A1', 1, v.Species('VA')): y_va,
}

moles_au = model_fcc.moles('AU', per_formula_unit=True).xreplace(subs)
moles_bi = model_fcc.moles('BI', per_formula_unit=True).xreplace(subs)

print(f"\nSubstituted values:")
print(f"  moles(AU) = {moles_au}")
print(f"  moles(BI) = {moles_bi}")
print(f"  Total moles atoms = {moles_au + moles_bi}")