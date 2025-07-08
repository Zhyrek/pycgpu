#!/usr/bin/env python3
"""
Debug script to specifically examine the starting_point composition data
and verify it contains the correct values that should be transferred to GPU
"""

import numpy as np
import pycalphad as pyc
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.workspace import Workspace
from pycalphad.core.starting_point import starting_point
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

def debug_starting_point_compositions():
    print("=== DEBUGGING STARTING_POINT COMPOSITIONS ===")
    clear_cupy_kernel_cache()
    
    tdb = Database("NbTi.tdb")
    phases = ["BCC_A2"]
    comps = ["NB", "TI", "VA"]
    
    conditions = {
        v.X("TI"): 0.1,
        v.T: 800,
        v.P: 101325
    }
    
    print(f"Input conditions: {conditions}")
    print(f"Expected: NB=0.9, TI=0.1, VA=0.0")
    
    # Create workspace
    wks = Workspace(database=tdb, components=comps, phases=phases, 
                   conditions=conditions, models=None, parameters=None)
    
    # Get unitless conditions
    unitless_conds = {key: np.asarray(val, dtype=float) for key, val in conditions.items() 
                     if not str(key).startswith('_')}
    
    print(f"\nComponents: {comps}")
    print(f"Unitless conditions: {unitless_conds}")
    
    # First call calculate() to generate grid data like GPU does
    from pycalphad.core.calculate import calculate
    
    grid_data = calculate(tdb, comps, phases, T=unitless_conds[v.T], P=unitless_conds[v.P],
                         model=wks.models.unwrap() if hasattr(wks.models, 'unwrap') else wks.models,
                         fake_points=True, phase_records=wks.phase_record_factory, 
                         output='GM', parameters=wks.parameters.unwrap() if hasattr(wks.parameters, 'unwrap') else wks.parameters,
                         to_xarray=False, conditions=conditions)
    
    print(f"\nGrid data GM shape: {grid_data.GM.shape}")
    print(f"Grid data X shape: {grid_data.X.shape}")
    print(f"Grid data X values: {grid_data.X}")
    
    # Call starting_point with grid data
    properties = starting_point(unitless_conds, wks.phase_record_factory.state_variables, 
                              wks.phase_record_factory, grid_data)
    
    print(f"\n--- STARTING_POINT DETAILED ANALYSIS ---")
    print(f"GM: {properties.GM.values}")
    print(f"MU: {properties.MU.values}")
    print(f"Phase: {properties.Phase.values}")
    print(f"NP: {properties.NP.values}")
    print(f"X shape: {properties.X.shape}")
    print(f"X values: {properties.X.values}")
    
    # Extract compositions for each phase
    x_values = properties.X.values
    np_values = properties.NP.values.flatten()
    phase_values = properties.Phase.values.flatten()
    
    print(f"\n--- COMPOSITION ANALYSIS ---")
    print(f"X array shape: {x_values.shape}")
    print(f"NP array: {np_values}")
    print(f"Phase array: {phase_values}")
    
    # Find active phases
    active_mask = np_values > 1e-10
    num_active = np.sum(active_mask)
    
    print(f"\nActive phases: {num_active}")
    print(f"Active phase indices: {np.where(active_mask)[0]}")
    print(f"Active phase names: {phase_values[active_mask]}")
    print(f"Active phase amounts: {np_values[active_mask]}")
    
    # Extract compositions for active phases
    for i, is_active in enumerate(active_mask):
        if is_active:
            # Extract composition for this phase
            if x_values.ndim == 6:  # (1,1,1,1,phases,components)
                comp = x_values[0, 0, 0, 0, i, :]
            elif x_values.ndim == 3:  # (phases, components, ?)
                comp = x_values[i, :, 0] if x_values.shape[2] > 0 else x_values[i, :]
            elif x_values.ndim == 2:  # (phases, components)
                comp = x_values[i, :]
            else:
                comp = x_values.flatten()[i*len(comps):(i+1)*len(comps)]
            
            print(f"\nPhase {i} ({phase_values[i]}) composition:")
            for j, component in enumerate(comps):
                if j < len(comp):
                    print(f"  {component}: {comp[j]:.6f}")
                    
    # Also test CPU equilibrium to compare
    print(f"\n--- CPU EQUILIBRIUM FOR COMPARISON ---")
    cpu_result = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=False)
    
    cpu_x = cpu_result.X.values
    cpu_np = cpu_result.NP.values.flatten()
    cpu_phases = cpu_result.Phase.values.flatten()
    
    cpu_active_mask = ~np.isnan(cpu_np) & (cpu_np > 1e-10)
    
    print(f"CPU final result:")
    for i, is_active in enumerate(cpu_active_mask):
        if is_active and i < cpu_x.shape[-2]:
            comp = cpu_x.reshape(-1, cpu_x.shape[-1])[i][:len(comps)]
            print(f"Phase {i} ({cpu_phases[i]}) final composition:")
            for j, component in enumerate(comps):
                if j < len(comp):
                    print(f"  {component}: {comp[j]:.6f}")

if __name__ == "__main__":
    debug_starting_point_compositions()