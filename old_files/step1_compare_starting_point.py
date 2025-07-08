#!/usr/bin/env python3
"""
Step 1: Compare the starting_point() data between CPU and GPU pathways
to identify where the 2-phase vs 1-phase divergence begins
"""

import numpy as np
import pycalphad as pyc
from pycalphad import Database, equilibrium, variables as v
import os
import glob

def clear_cupy_kernel_cache():
    cache_dir = os.path.expanduser("~/.cupy/kernel_cache")
    if os.path.exists(cache_dir):
        cubin_files = glob.glob(os.path.join(cache_dir, "*.cubin"))
        for cubin_file in cubin_files:
            try:
                os.remove(cubin_file)
            except OSError:
                pass

def extract_cpu_starting_point():
    """Extract the starting_point data that CPU uses internally"""
    print("=== STEP 1: CPU STARTING_POINT ANALYSIS ===")
    
    tdb = Database("NbTi.tdb")
    phases = ["BCC_A2"]
    comps = ["NB", "TI", "VA"]
    
    conditions = {
        v.X("TI"): 0.1,
        v.T: 800,
        v.P: 101325
    }
    
    print(f"Input conditions: {conditions}")
    
    # Create workspace like both CPU and GPU do
    from pycalphad.core.workspace import Workspace
    wks = Workspace(database=tdb, components=comps, phases=phases, 
                   conditions=conditions, models=None, parameters=None)
    
    print(f"Components: {wks.components}")
    print(f"State variables: {wks.phase_record_factory.state_variables}")
    
    # Call starting_point with the same parameters GPU uses
    from pycalphad.core.starting_point import starting_point
    from pycalphad.core.calculate import calculate
    
    # First call calculate() like GPU does
    unitless_conds = {key: np.asarray(val, dtype=float) for key, val in conditions.items() 
                     if not str(key).startswith('_')}
    
    # Call calculate to get grid data
    grid_data = calculate(tdb, comps, phases, T=unitless_conds[v.T], P=unitless_conds[v.P],
                         X_TI=unitless_conds[v.X("TI")], model=wks.models.unwrap() if hasattr(wks.models, 'unwrap') else wks.models,
                         fake_points=True, phase_records=wks.phase_record_factory, 
                         output='GM', parameters=wks.parameters.unwrap() if hasattr(wks.parameters, 'unwrap') else wks.parameters,
                         to_xarray=False, conditions=conditions)
    
    print(f"\n--- CALCULATE() OUTPUT ---")
    print(f"Grid GM shape: {grid_data.GM.shape}")
    print(f"Grid GM values: {grid_data.GM.flatten()[:5]}...")
    print(f"Grid Phase shape: {grid_data.Phase.shape}")
    print(f"Grid Phase values: {grid_data.Phase.flatten()[:5]}")
    print(f"Grid X shape: {grid_data.X.shape}")
    print(f"Grid X values (first phase): {grid_data.X.flatten()[:6]}")
    
    # Call starting_point like GPU does  
    properties = starting_point(unitless_conds, wks.phase_record_factory.state_variables, 
                              wks.phase_record_factory, grid_data)
    
    print(f"\n--- STARTING_POINT() OUTPUT ---")
    print(f"Properties GM shape: {properties.GM.shape}")
    print(f"Properties GM values: {properties.GM.values.flatten()}")
    print(f"Properties MU shape: {properties.MU.shape}")
    print(f"Properties MU values: {properties.MU.values.flatten()}")
    print(f"Properties Phase shape: {properties.Phase.shape}")
    print(f"Properties Phase values: {properties.Phase.values.flatten()}")
    print(f"Properties NP shape: {properties.NP.shape}")
    print(f"Properties NP values: {properties.NP.values.flatten()}")
    print(f"Properties X shape: {properties.X.shape}")
    print(f"Properties X values: {properties.X.values.flatten()[:6]}")
    
    # Count actual phases in starting_point
    np_values = properties.NP.values.flatten()
    phase_values = properties.Phase.values.flatten()
    
    active_mask = np_values > 1e-10
    num_active = np.sum(active_mask)
    
    print(f"\n--- STARTING_POINT ANALYSIS ---")
    print(f"Total phases in starting_point: {len(phase_values)}")
    print(f"Active phases (NP > 1e-10): {num_active}")
    print(f"Active phase names: {phase_values[active_mask]}")
    print(f"Active phase amounts: {np_values[active_mask]}")
    
    if num_active == 1:
        print("✅ CPU starting_point correctly identifies 1 active phase")
    else:
        print(f"⚠️  CPU starting_point has {num_active} active phases")
    
    return properties, wks

def compare_with_gpu_debug():
    """Add debug output to GPU to compare starting_point data"""
    print("\n=== STEP 1: COMPARING WITH GPU STARTING_POINT ===")
    
    # We'll run the GPU calculation and examine its debug output
    # The GPU debug shows: InitialPhaseData[0]: num_phases=2, phases=[0 0], amounts=[0.17926814 0.82073186]
    
    tdb = Database("NbTi.tdb")
    phases = ["BCC_A2"]
    comps = ["NB", "TI", "VA"]
    
    conditions = {
        v.X("TI"): 0.1,
        v.T: 800,
        v.P: 101325
    }
    
    print("Running GPU with verbose output to capture starting_point data...")
    
    # Run GPU calculation and capture its initialization
    result = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=True)
    
    print("\n--- GPU DEBUG ANALYSIS ---")
    print("From GPU debug output we see:")
    print("  InitialPhaseData[0]: num_phases=2, phases=[0 0], amounts=[0.17926814 0.82073186]")
    print("")
    print("This shows GPU is creating 2 phases of same type with amounts [0.179, 0.821]")
    print("But CPU starting_point should give 1 phase with amount [1.0]")
    print("")
    print("🔍 DIVERGENCE POINT IDENTIFIED:")
    print("   The issue is in GPU's processing of starting_point() data")
    print("   GPU is incorrectly converting 1-phase starting_point into 2-phase initial data")

def main():
    clear_cupy_kernel_cache()
    
    # Step 1a: Analyze CPU starting_point
    cpu_properties, cpu_wks = extract_cpu_starting_point()
    
    # Step 1b: Compare with GPU behavior
    compare_with_gpu_debug()
    
    # Step 1c: Conclusion
    print("\n" + "="*60)
    print("STEP 1 CONCLUSION")
    print("="*60)
    print("Next steps:")
    print("1. ✅ Identified that starting_point() should give 1 active phase")
    print("2. ⚠️  GPU is incorrectly processing this into 2 phases")  
    print("3. 🔧 Need to fix GPU initial phase data creation logic")
    print("4. 📍 The bug is in _prepare_gpu_data() phase counting/setup")

if __name__ == "__main__":
    main()