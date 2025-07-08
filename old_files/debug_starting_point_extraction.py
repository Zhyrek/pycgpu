#!/usr/bin/env python
"""
Debug the starting point data extraction in GPU code.
"""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.starting_point import starting_point
from pycalphad.core.workspace import Workspace
import warnings
warnings.filterwarnings('ignore')

def debug_starting_point_extraction():
    """Debug how starting_point data is extracted"""
    print("="*60)
    print("STARTING POINT EXTRACTION DEBUG")
    print("="*60)
    
    dbf = Database("NbTi.tdb")
    
    # Use the same single condition that shows the problem
    conditions = {v.X("TI"): 1e-10, v.T: 600, v.P: 101325}
    
    print(f"Condition: T=600K, X_TI=1e-10 (pure NB system)")
    
    # Mimic the GPU preparation process
    comp_list = ['NB', 'TI', 'VA']
    phase_list = ['LIQUID', 'BCC_A2']
    
    print(f"\n1. CREATING WORKSPACE OBJECT:")
    wks_obj = Workspace(database=dbf, components=comp_list, phases=phase_list,
                       conditions=conditions, verbose=False)
    
    # Get state variables and conditions like GPU code does
    state_variables = [v.T, v.P]
    all_conditions = conditions
    
    print(f"  Components: {comp_list}")
    print(f"  Phases: {phase_list}")
    print(f"  State variables: {state_variables}")
    print(f"  All conditions: {all_conditions}")
    
    # Call starting_point like the GPU code does
    print(f"\n2. CALLING STARTING_POINT:")
    grid = None  # No grid for single point
    properties = starting_point(all_conditions, state_variables, wks_obj.phase_record_factory, grid)
    
    print(f"  Starting point properties object: {type(properties)}")
    print(f"  Available attributes: {[attr for attr in dir(properties) if not attr.startswith('_')]}")
    
    # Check shapes of key properties
    print(f"\n3. STARTING POINT PROPERTY SHAPES:")
    for prop_name in ['Phase', 'NP', 'MU', 'X', 'Y']:
        if hasattr(properties, prop_name):
            prop_val = getattr(properties, prop_name)
            print(f"  {prop_name}: shape={prop_val.shape if hasattr(prop_val, 'shape') else 'no shape'}, type={type(prop_val)}")
            if hasattr(prop_val, 'shape') and len(prop_val.shape) <= 2:
                print(f"    Value: {prop_val}")
            elif hasattr(prop_val, 'shape'):
                print(f"    Value (first 3 elements): {prop_val.flat[:3]}")
    
    # Simulate what GPU code does with unravel_index
    print(f"\n4. GPU EXTRACTION SIMULATION:")
    
    # This is where the bug likely is - check gm_array shape
    dummy_gm_array = np.zeros((1, 1, 1, 1))  # Single condition should give this shape
    num_conditions_total = 1
    
    for cond_idx in range(num_conditions_total):
        print(f"  Condition {cond_idx}:")
        multi_idx = np.unravel_index(cond_idx, dummy_gm_array.shape)
        print(f"    multi_idx from unravel_index: {multi_idx}")
        
        # Try extracting like GPU code does
        try:
            phase_values = properties.Phase[multi_idx] if len(multi_idx) > 0 else properties.Phase
            np_values = properties.NP[multi_idx] if len(multi_idx) > 0 else properties.NP
            mu_values = properties.MU[multi_idx] if len(multi_idx) > 0 else properties.MU
            
            print(f"    Extracted Phase: {phase_values}")
            print(f"    Extracted NP: {np_values}")
            print(f"    Extracted MU: {mu_values}")
            
        except Exception as e:
            print(f"    Extraction failed: {e}")
            
        # Try direct access (what should work for single condition)
        print(f"  Direct access (correct method):")
        print(f"    Direct Phase: {properties.Phase}")
        print(f"    Direct NP: {properties.NP}")
        print(f"    Direct MU: {properties.MU}")
        
    # Count active phases correctly
    print(f"\n5. ACTIVE PHASE ANALYSIS:")
    phase_values = properties.Phase
    np_values = properties.NP
    
    print(f"  Raw phase names: {phase_values}")
    print(f"  Raw phase amounts: {np_values}")
    
    # Find truly active phases
    active_mask = np_values > 1e-8
    active_phases = phase_values[active_mask]
    active_amounts = np_values[active_mask]
    
    print(f"  Active mask: {active_mask}")
    print(f"  Active phases: {active_phases}")
    print(f"  Active amounts: {active_amounts}")
    print(f"  Number of active phases: {len(active_phases)}")
    
    # Compare with CPU equilibrium
    print(f"\n6. COMPARE WITH CPU EQUILIBRIUM:")
    cpu_result = equilibrium(dbf, comp_list, phase_list, conditions, gpu=False, verbose=False)
    cpu_phases = cpu_result.Phase.values[0,0,0,0,:]
    cpu_np = cpu_result.NP.values[0,0,0,0,:]
    cpu_active_mask = cpu_np > 1e-8
    
    print(f"  CPU final phases: {cpu_phases}")
    print(f"  CPU final amounts: {cpu_np}")
    print(f"  CPU active phases: {cpu_phases[cpu_active_mask]}")
    
    if len(active_phases) == len(cpu_phases[cpu_active_mask]):
        print(f"  ✅ Starting point has correct number of active phases")
    else:
        print(f"  ❌ Starting point has {len(active_phases)} phases, CPU final has {len(cpu_phases[cpu_active_mask])}")

if __name__ == "__main__":
    debug_starting_point_extraction()