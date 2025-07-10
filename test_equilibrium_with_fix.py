#!/usr/bin/env python3
"""Test equilibrium calculation to see if fix helps"""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.workspace import Workspace
from pycalphad.core.starting_point import starting_point

# Set up test case
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']
conds = {v.T: 1800, v.P: 101325, v.X('TI'): 0.3}

# CPU equilibrium calculation
cpu_result = equilibrium(dbf, comps, phases, conds, verbose=False)
cpu_gm = float(cpu_result.GM.values)
print(f"CPU equilibrium GM: {cpu_gm:.1f} J/mol")

# GPU equilibrium calculation (requires GPU support)
try:
    gpu_result = equilibrium(dbf, comps, phases, conds, verbose=False, gpu=True)
    gpu_gm = float(gpu_result.GM.values)
    print(f"GPU equilibrium GM: {gpu_gm:.1f} J/mol")
    
    error = abs(gpu_gm - cpu_gm)
    rel_error = error / abs(cpu_gm) * 100
    print(f"\nAbsolute error: {error:.1f} J/mol")
    print(f"Relative error: {rel_error:.2f}%")
    
    if rel_error < 1.0:
        print("\nSUCCESS: GPU and CPU results match within 1%")
    else:
        print(f"\nERROR: GPU result differs by {rel_error:.1f}%")
        
except Exception as e:
    print(f"GPU calculation failed: {e}")
    print("\nTrying manual verification...")
    
    # Manually check the Hessian values
    wks = Workspace(database=dbf, components=comps, phases=phases, conditions=conds)
    
    # Get starting point
    points = starting_point(dbf, comps, phases, conds, wks)
    
    print(f"\nStarting point Y_NB: {points.Y.sel(vertex=0, component='NB').values[0]:.4f}")
    print(f"Starting point Y_TI: {points.Y.sel(vertex=0, component='TI').values[0]:.4f}")