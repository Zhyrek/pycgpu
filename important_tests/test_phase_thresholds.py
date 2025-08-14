#!/usr/bin/env python
"""Test what phase amounts are being passed to CPU vs GPU."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def main():
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2', 'L12', 'ALCU_THETA', 'AL13FE4', 'AL5FE2']
    
    print("=" * 80)
    print("PHASE THRESHOLD ANALYSIS")
    print("=" * 80)
    
    conditions = {
        v.X('AL'): 0.60,
        v.X('CU'): 0.10,
        v.T: 600,
        v.P: 101325
    }
    
    print("\nConstants:")
    print("- CPU MIN_PHASE_FRACTION: 1e-6")
    print("- GPU uses MIN_PHASE_FRACTION/100.0 = 1e-8 for initial phase filtering")
    
    print("\nThis means:")
    print("- CPU drops phases with amount < 1e-6")
    print("- GPU keeps phases with amount > 1e-8")
    print("- GPU threshold is 100x lower (more permissive)")
    
    print("\n" + "=" * 80)
    print("TESTING PHASE AMOUNTS")
    print("=" * 80)
    
    # Let's see what phases and amounts are in the starting point
    from pycalphad import calculate
    from pycalphad.core.starting_point import starting_point
    from pycalphad.core.workspace import Workspace
    
    # Create workspace
    wks = Workspace(database=dbf, components=comps, phases=phases, conditions=conditions)
    state_variables = [v.P, v.T]
    
    # Calculate with default sampling to get starting grid
    print("\nCalculating starting grid...")
    grid = calculate(dbf, comps, phases, output='GM', **{
        'T': 600, 'P': 101325, 'X(AL)': 0.60, 'X(CU)': 0.10})
    
    # Get starting point
    sp = starting_point(conditions, state_variables, wks.phase_record_factory, grid)
    
    # Extract phase amounts
    np_vals = sp.NP.values.flatten()
    phase_vals = sp.Phase.values.flatten()
    
    print("\nStarting point phase amounts:")
    print("-" * 50)
    
    active_phases = []
    for i, (phase, amount) in enumerate(zip(phase_vals, np_vals)):
        if not np.isnan(amount):
            active_phases.append((i, phase, amount))
            
    # Sort by amount
    active_phases.sort(key=lambda x: x[2], reverse=True)
    
    for idx, phase, amount in active_phases:
        cpu_keep = "YES" if amount >= 1e-6 else "NO"
        gpu_keep = "YES" if amount > 1e-8 else "NO"
        
        print(f"  Phase {idx}: {phase:12s} = {amount:.10f}")
        print(f"           CPU keeps (>= 1e-6): {cpu_keep}")
        print(f"           GPU keeps (> 1e-8):  {gpu_keep}")
        
        if cpu_keep != gpu_keep:
            print(f"           ⚠️ DIFFERENCE: GPU keeps but CPU drops!")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    cpu_count = sum(1 for _, _, amt in active_phases if amt >= 1e-6)
    gpu_count = sum(1 for _, _, amt in active_phases if amt > 1e-8)
    
    print(f"CPU would start with {cpu_count} phases")
    print(f"GPU would start with {gpu_count} phases")
    
    if gpu_count > cpu_count:
        print(f"\n⚠️ GPU includes {gpu_count - cpu_count} extra phase(s) due to lower threshold!")
        print("This explains the matrix size difference (7x7 vs 6x6)")
        print("\nThe GPU threshold should be changed from MIN_PHASE_FRACTION/100.0 to MIN_PHASE_FRACTION")

if __name__ == "__main__":
    main()