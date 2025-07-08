#!/usr/bin/env python3
"""
Debug the ~942 J/mol GM difference between CPU and GPU calculations
by comparing each step of the calculation process
"""

import numpy as np
import pycalphad as pyc
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.workspace import Workspace
from pycalphad.core.starting_point import starting_point
from pycalphad.core.calculate import calculate
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

def debug_energy_calculations():
    """Compare the energy calculation details between CPU and GPU"""
    print("=== DEBUGGING GM DIFFERENCE: CPU vs GPU ENERGY CALCULATIONS ===")
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
    print(f"Expected composition: NB=0.9, TI=0.1, VA=0.0")
    print(f"Expected: Single BCC_A2 phase equilibrium")
    
    # ==== CPU DETAILED ANALYSIS ====
    print("\n" + "="*60)
    print("CPU DETAILED ENERGY ANALYSIS")
    print("="*60)
    
    # Create workspace to access internal CPU calculations
    wks = Workspace(database=tdb, components=comps, phases=phases, 
                   conditions=conditions, models=None, parameters=None)
    
    # Get CPU results with detailed info
    cpu_result = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=False)
    
    cpu_gm = cpu_result.GM.values.flatten()[0]
    cpu_mu = cpu_result.MU.values.flatten()
    cpu_np = cpu_result.NP.values.flatten()
    cpu_x = cpu_result.X.values
    cpu_phases = cpu_result.Phase.values.flatten()
    
    print(f"CPU Final Results:")
    print(f"  GM: {cpu_gm:.6f} J/mol")
    print(f"  MU: {cpu_mu[:2]}")  # First 2 chemical potentials
    print(f"  NP: {cpu_np[~np.isnan(cpu_np)]}")  # Non-NaN phase amounts
    print(f"  Phases: {cpu_phases[~np.isnan(cpu_np)]}")
    
    # Extract CPU composition for first active phase
    cpu_active_mask = ~np.isnan(cpu_np) & (cpu_np > 1e-10)
    if np.any(cpu_active_mask):
        first_active_idx = np.where(cpu_active_mask)[0][0]
        if first_active_idx < cpu_x.shape[-2]:
            cpu_comp = cpu_x.reshape(-1, cpu_x.shape[-1])[first_active_idx][:2]
            print(f"  Composition: NB={cpu_comp[0]:.6f}, TI={cpu_comp[1]:.6f}")
    
    # ==== GPU DETAILED ANALYSIS ====
    print("\n" + "="*60)
    print("GPU DETAILED ENERGY ANALYSIS")
    print("="*60)
    
    # Get GPU results with detailed info
    gpu_result = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=False)
    
    gpu_gm = gpu_result.GM.values.flatten()[0]
    gpu_mu = gpu_result.MU.values.flatten()
    gpu_np = gpu_result.NP.values.flatten()
    gpu_x = gpu_result.X.values
    gpu_phases = gpu_result.Phase.values.flatten()
    
    print(f"GPU Final Results:")
    print(f"  GM: {gpu_gm:.6f} J/mol")
    print(f"  MU: {gpu_mu[:2]}")  # First 2 chemical potentials
    print(f"  NP: {gpu_np[gpu_np > 1e-10]}")  # Active phase amounts
    print(f"  Phases: {gpu_phases[gpu_np > 1e-10]}")
    
    # Extract GPU composition for first active phase
    gpu_active_mask = gpu_np > 1e-10
    if np.any(gpu_active_mask):
        first_active_idx = np.where(gpu_active_mask)[0][0]
        if first_active_idx < gpu_x.shape[-2]:
            gpu_comp = gpu_x.reshape(-1, gpu_x.shape[-1])[first_active_idx][:2]
            print(f"  Composition: NB={gpu_comp[0]:.6f}, TI={gpu_comp[1]:.6f}")
    
    # ==== COMPARISON AND ANALYSIS ====
    print("\n" + "="*60)
    print("DETAILED COMPARISON ANALYSIS")
    print("="*60)
    
    gm_diff = abs(cpu_gm - gpu_gm)
    mu_diff = np.abs(cpu_mu[:2] - gpu_mu[:2])
    
    print(f"Differences:")
    print(f"  GM difference: {gm_diff:.6f} J/mol")
    print(f"  MU differences: NB={mu_diff[0]:.6f}, TI={mu_diff[1]:.6f} J/mol")
    
    # Analyze potential causes
    print(f"\nPotential causes of GM difference:")
    print(f"1. Different energy calculation methods")
    print(f"2. Different chemical potential calculations") 
    print(f"3. Different composition values affecting energy")
    print(f"4. Numerical precision differences")
    print(f"5. Different equilibrium convergence criteria")
    
    # Check if composition differences could explain GM difference
    if np.any(cpu_active_mask) and np.any(gpu_active_mask):
        comp_diff = np.abs(cpu_comp - gpu_comp)
        print(f"\nComposition differences:")
        print(f"  NB: {comp_diff[0]:.6f}")
        print(f"  TI: {comp_diff[1]:.6f}")
        
        if np.max(comp_diff) > 1e-6:
            print(f"  ⚠️  Significant composition differences detected!")
            print(f"  This could contribute to the GM difference.")
        else:
            print(f"  ✓ Compositions are very close.")
    
    # Check chemical potential differences
    if np.max(mu_diff) > 100:  # More than 100 J/mol difference
        print(f"\n⚠️  Large chemical potential differences detected!")
        print(f"  This suggests the GPU calculation is not reaching the same equilibrium state as CPU.")
    else:
        print(f"\n✓ Chemical potentials are reasonably close.")
    
    return cpu_gm, gpu_gm, gm_diff

def analyze_starting_point_vs_final():
    """Compare starting_point data with final equilibrium results"""
    print("\n" + "="*60) 
    print("STARTING_POINT vs FINAL RESULTS ANALYSIS")
    print("="*60)
    
    tdb = Database("NbTi.tdb")
    phases = ["BCC_A2"]
    comps = ["NB", "TI", "VA"]
    
    conditions = {
        v.X("TI"): 0.1,
        v.T: 800,
        v.P: 101325
    }
    
    # Create workspace to access starting_point
    wks = Workspace(database=tdb, components=comps, phases=phases, 
                   conditions=conditions, models=None, parameters=None)
    
    # Get unitless conditions
    unitless_conds = {key: np.asarray(val, dtype=float) for key, val in conditions.items() 
                     if not str(key).startswith('_')}
    
    # Call calculate first to get grid data
    grid_data = calculate(tdb, comps, phases, T=unitless_conds[v.T], P=unitless_conds[v.P],
                         model=wks.models.unwrap() if hasattr(wks.models, 'unwrap') else wks.models,
                         fake_points=True, phase_records=wks.phase_record_factory, 
                         output='GM', parameters=wks.parameters.unwrap() if hasattr(wks.parameters, 'unwrap') else wks.parameters,
                         to_xarray=False, conditions=conditions)
    
    # Call starting_point  
    try:
        properties = starting_point(unitless_conds, wks.phase_record_factory.state_variables, 
                                  wks.phase_record_factory, grid_data)
        
        starting_gm = properties.GM.values.flatten()[0]
        starting_mu = properties.MU.values.flatten()
        
        print(f"Starting_point results:")
        print(f"  GM: {starting_gm:.6f} J/mol")
        print(f"  MU: {starting_mu[:2]}")
        
    except Exception as e:
        print(f"Error calling starting_point: {e}")
        return
    
    # Compare with final CPU results
    cpu_result = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=False)
    final_cpu_gm = cpu_result.GM.values.flatten()[0]
    final_cpu_mu = cpu_result.MU.values.flatten()
    
    print(f"\nFinal CPU equilibrium results:")
    print(f"  GM: {final_cpu_gm:.6f} J/mol")
    print(f"  MU: {final_cpu_mu[:2]}")
    
    print(f"\nStarting_point vs Final CPU:")
    print(f"  GM difference: {abs(starting_gm - final_cpu_gm):.6f} J/mol")
    print(f"  MU differences: {np.abs(starting_mu[:2] - final_cpu_mu[:2])}")
    
    if abs(starting_gm - final_cpu_gm) < 1e-6:
        print(f"  ✓ Starting_point and final CPU results are essentially identical")
        print(f"  This means the GPU should use starting_point values directly")
    else:
        print(f"  ⚠️  Starting_point and final CPU results differ")
        print(f"  CPU performs additional equilibrium refinement")

if __name__ == "__main__":
    # Run the detailed analysis
    cpu_gm, gpu_gm, gm_diff = debug_energy_calculations()
    
    # Analyze starting_point vs final results
    analyze_starting_point_vs_final()
    
    print("\n" + "="*60)
    print("NEXT STEPS FOR GPU FIXES")
    print("="*60)
    print("Based on the analysis above:")
    print("1. If starting_point ≈ final CPU: GPU should use starting_point GM directly")
    print("2. If chemical potentials differ significantly: Fix GPU chemical potential calculation")
    print("3. If compositions differ: Improve GPU composition accuracy")
    print(f"4. Current GM difference: {gm_diff:.1f} J/mol needs to be reduced to <1 J/mol")