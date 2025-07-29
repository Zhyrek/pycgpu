#!/usr/bin/env python
"""Trace how phases are initialized in GPU vs CPU."""

import numpy as np
from pycalphad import Database, calculate, equilibrium, variables as v
import os

# Clear GPU cache
if 'CUDA_CACHE_DISABLE' in os.environ:
    import cupy as cp
    cp.clear_memo()

# Load database
db = Database('NbTi.tdb')
components = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'HCP_A3']

# Test conditions
conditions = {
    v.T: 500,
    v.P: 101325,
    v.X('TI'): 0.1,
    v.N: 1
}

print("Checking phase initialization...")
print("="*80)

# First check what the calculate step produces
print("\nRunning calculate step...")
calc_result = calculate(db, components, phases, T=conditions[v.T], P=conditions[v.P], 
                       output='GM', model=None, points={'BCC_A2': 50, 'HCP_A3': 50})

# Find BCC_A2 points
bcc_mask = calc_result.Phase == 'BCC_A2'
bcc_points = np.sum(bcc_mask)
print(f"Calculate generated {bcc_points} points for BCC_A2")

if bcc_points > 0:
    bcc_y = calc_result.Y.sel(vertex='Y1').values[bcc_mask]
    print(f"BCC_A2 Y(TI) range: {bcc_y.min():.4f} to {bcc_y.max():.4f}")
    
    # Check if we have points in both Ti-poor and Ti-rich regions
    ti_poor = np.sum(bcc_y < 0.5)
    ti_rich = np.sum(bcc_y > 0.5)
    print(f"Ti-poor points (Y<0.5): {ti_poor}")
    print(f"Ti-rich points (Y>0.5): {ti_rich}")
    
    if ti_poor > 0 and ti_rich > 0:
        print("✓ Calculate has points in both regions of miscibility gap")
    else:
        print("✗ Calculate missing points in one region of miscibility gap")
        print("  This would prevent GPU from finding both phases!")

# Check starting points passed to equilibrium solver
print("\n" + "="*80)
print("The issue is likely:")
print("1. Calculate step not generating points in both regions")
print("2. GPU solver converging both phases to same composition")
print("3. Phase consolidation removing one phase")
print("="*80)