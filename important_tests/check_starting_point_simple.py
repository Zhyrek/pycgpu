#!/usr/bin/env python
"""Check what starting point passes to GPU vs CPU."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, calculate, variables as v
from pycalphad.core.starting_point import starting_point
from pycalphad.core.workspace import Workspace
import warnings
warnings.filterwarnings("ignore")

def main():
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2', 'L12', 'ALCU_THETA', 'AL13FE4', 'AL5FE2']
    
    conditions = {
        v.X('AL'): 0.60,
        v.X('CU'): 0.10,
        v.T: 600,
        v.P: 101325
    }
    
    print("=" * 80)
    print("CHECKING STARTING POINT GENERATION")
    print("=" * 80)
    
    # Create workspace
    wks = Workspace(database=dbf, components=comps, phases=phases, conditions=conditions)
    state_variables = [v.P, v.T]
    
    # Calculate grid with default sampling
    print("\n1. Calculating grid...")
    grid = calculate(dbf, comps, phases, output='GM', **{
        'T': 600, 'P': 101325, 'X(AL)': 0.60, 'X(CU)': 0.10})
    
    print(f"   Grid shape: {grid.GM.shape}")
    print(f"   Total points: {grid.GM.size}")
    
    # Check energies by phase
    print("\n2. Minimum energy by phase in grid:")
    for phase in phases:
        phase_mask = grid.Phase.values == phase
        if phase_mask.any():
            phase_gm = grid.GM.values[phase_mask]
            print(f"   {phase:12s}: min GM = {np.min(phase_gm):8.1f} J/mol, count = {phase_mask.sum()}")
    
    # Get starting point
    print("\n3. Getting starting point from grid...")
    sp = starting_point(conditions, state_variables, wks.phase_record_factory, grid)
    
    print(f"   Starting point NP shape: {sp.NP.shape}")
    print(f"   Starting point Phase shape: {sp.Phase.shape}")
    
    # Check active phases
    np_vals = sp.NP.values.flatten()
    phase_vals = sp.Phase.values.flatten()
    
    print("\n4. Active phases in starting point:")
    active_count = 0
    for i, (phase, amount) in enumerate(zip(phase_vals, np_vals)):
        if not np.isnan(amount) and amount > 1e-6:
            print(f"   Phase {i}: {phase:12s} = {amount:.6f}")
            active_count += 1
    
    print(f"\nTotal active phases: {active_count}")
    
    print("\n" + "=" * 80)
    print("KEY FINDING:")
    print("=" * 80)
    if active_count == 3:
        print("Starting point has 3 phases! This is what gets passed to both CPU and GPU.")
        print("The difference is:")
        print("- CPU solver drops one phase during iteration")
        print("- GPU solver keeps all 3 phases")
        print("\nThis explains the matrix size difference (6x6 vs 7x7)")
    elif active_count == 2:
        print("Starting point has 2 phases.")
        print("Need to check why GPU ends up with 3 phases.")
    
    # Now test with neighbor point
    print("\n" + "=" * 80)
    print("TESTING WITH NEIGHBOR POINT")
    print("=" * 80)
    
    print("\n5. Testing neighbor point X(AL)=0.59...")
    grid2 = calculate(dbf, comps, phases, output='GM', **{
        'T': 600, 'P': 101325, 'X(AL)': 0.59, 'X(CU)': 0.10})
    
    print(f"   Grid shape: {grid2.GM.shape}")
    print(f"   Total points: {grid2.GM.size}")
    
    # Get starting point
    print("\n6. Getting starting point from neighbor grid...")
    conditions2 = {
        v.X('AL'): 0.59,
        v.X('CU'): 0.10,
        v.T: 600,
        v.P: 101325
    }
    sp2 = starting_point(conditions2, state_variables, wks.phase_record_factory, grid2)
    
    np_vals2 = sp2.NP.values.flatten()
    phase_vals2 = sp2.Phase.values.flatten()
    
    print("\n7. Active phases at X(AL)=0.59:")
    active_count2 = 0
    for i, (phase, amount) in enumerate(zip(phase_vals2, np_vals2)):
        if not np.isnan(amount) and amount > 1e-6:
            print(f"   Phase {i}: {phase:12s} = {amount:.6f}")
            active_count2 += 1
    
    print(f"\nTotal active phases at X(AL)=0.59: {active_count2}")

if __name__ == "__main__":
    main()