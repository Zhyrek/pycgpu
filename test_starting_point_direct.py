#!/usr/bin/env python
"""Direct test of starting_point function to see if it produces different results."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, calculate, variables as v
from pycalphad.core.starting_point import starting_point
from pycalphad.core.utils import unpack_conditions
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
import warnings
warnings.filterwarnings("ignore")

def test_starting_point_directly():
    """Test starting_point function directly with same inputs."""
    
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    phases = ['AU2BI_C15', 'BCC_A2', 'FCC_A1', 'HCP_A3', 'LIQUID', 'RHOMBOHEDRAL_A7']
    
    conditions = {
        v.X('BI'): 0.1,
        v.T: 400,
        v.P: 101325,
        v.N: 1.0
    }
    
    print("="*70)
    print("DIRECT STARTING_POINT TEST")
    print("="*70)
    
    # Prepare inputs exactly as CPU/GPU do
    unitless_conds = unpack_conditions(conditions)
    state_variables = sorted([c for c in conditions.keys() if c in [v.T, v.P, v.N]], key=str)
    nonvacant_components = [x for x in sorted(comps) if x != 'VA']
    
    # Create phase records
    models = {phase_name: dbf.phases[phase_name].model_hints.get('ordered_phase', phase_name) 
              for phase_name in phases}
    phase_records = PhaseRecordFactory(dbf, comps, state_variables, models)
    
    # Calculate grid
    print("\nCalculating grid...")
    grid_opts = {'pdens': 60}
    str_conds = {str(k): v for k, v in conditions.items()}
    statevar_strings = [str(x) for x in state_variables]
    grid_opts.update({key: value for key, value in str_conds.items() if key in statevar_strings})
    
    grid = calculate(dbf, comps, phases, output='GM', 
                    model=models, fake_points=True,
                    phase_records=phase_records, 
                    parameters={}, to_xarray=False, 
                    conditions=conditions, **grid_opts)
    
    print(f"Grid shape: {grid.GM.shape}")
    
    # Call starting_point with verbose
    print("\n--- Calling starting_point with verbose=True ---")
    result = starting_point(unitless_conds, state_variables, phase_records, grid, verbose=True)
    
    print("\n--- Starting Point Results ---")
    print(f"GM: {result.GM.values.flatten()[0]:.6f}")
    print(f"MU: {result.MU.values.flatten()}")
    
    # Extract active phases
    phases_arr = result.Phase.values.flatten()
    np_arr = result.NP.values.flatten()
    
    print("\nActive phases:")
    for phase, amount in zip(phases_arr[:10], np_arr[:10]):
        if phase != '' and amount > 1e-8:
            print(f"  {phase}: {amount:.6f}")
    
    # Now test with different phase ordering
    print("\n" + "="*70)
    print("TEST WITH DIFFERENT PHASE ORDER")
    print("="*70)
    
    # Reorder phases
    phases_reordered = ['FCC_A1', 'HCP_A3', 'AU2BI_C15', 'BCC_A2', 'LIQUID', 'RHOMBOHEDRAL_A7']
    
    models_reordered = {phase_name: dbf.phases[phase_name].model_hints.get('ordered_phase', phase_name) 
                        for phase_name in phases_reordered}
    phase_records_reordered = PhaseRecordFactory(dbf, comps, state_variables, models_reordered)
    
    grid_reordered = calculate(dbf, comps, phases_reordered, output='GM', 
                              model=models_reordered, fake_points=True,
                              phase_records=phase_records_reordered, 
                              parameters={}, to_xarray=False, 
                              conditions=conditions, **grid_opts)
    
    print(f"\nReordered grid shape: {grid_reordered.GM.shape}")
    
    print("\n--- Calling starting_point with reordered phases ---")
    result_reordered = starting_point(unitless_conds, state_variables, phase_records_reordered, 
                                     grid_reordered, verbose=True)
    
    print("\n--- Reordered Starting Point Results ---")
    print(f"GM: {result_reordered.GM.values.flatten()[0]:.6f}")
    
    # Extract active phases
    phases_arr_reordered = result_reordered.Phase.values.flatten()
    np_arr_reordered = result_reordered.NP.values.flatten()
    
    print("\nActive phases:")
    for phase, amount in zip(phases_arr_reordered[:10], np_arr_reordered[:10]):
        if phase != '' and amount > 1e-8:
            print(f"  {phase}: {amount:.6f}")

if __name__ == "__main__":
    test_starting_point_directly()