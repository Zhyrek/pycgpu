#!/usr/bin/env python3
"""
Debug the initial conditions and phase data being passed to GPU vs CPU
to identify why GPU is finding different equilibrium results
"""

import numpy as np
import pycalphad as pyc
from pycalphad import Database, equilibrium, variables as v
import time
import os
import glob

def clear_cupy_kernel_cache():
    """Clear CuPy kernel cache to ensure fresh compilation"""
    cache_dir = os.path.expanduser("~/.cupy/kernel_cache")
    if os.path.exists(cache_dir):
        cubin_files = glob.glob(os.path.join(cache_dir, "*.cubin"))
        if cubin_files:
            print(f"[CACHE] Clearing {len(cubin_files)} .cubin files...")
            for cubin_file in cubin_files:
                try:
                    os.remove(cubin_file)
                except OSError:
                    pass

def debug_starting_point_data():
    """Compare the starting_point data used by CPU vs GPU"""
    print("=== DEBUGGING STARTING POINT DATA ===")
    
    clear_cupy_kernel_cache()
    
    tdb = Database("NbTi.tdb")
    phases = ["BCC_A2"]
    comps = ["NB", "TI", "VA"]
    
    conditions = {
        v.X("TI"): 0.1,
        v.T: 800,
        v.P: 101325
    }
    
    print(f"Conditions: {conditions}")
    
    # First run CPU to see what starting_point gives us
    print("\n" + "="*50)
    print("CPU STARTING POINT ANALYSIS")
    print("="*50)
    
    # We need to call the internal workspace functions to see starting_point data
    from pycalphad.core.workspace import Workspace
    
    # Create workspace like the CPU does
    wks = Workspace(database=tdb, components=comps, phases=phases, 
                   conditions=conditions, models=None, parameters=None)
    
    # Call starting_point like CPU does
    from pycalphad.core.starting_point import starting_point
    
    # Get the same unitless conditions that both CPU and GPU use
    unitless_conds = {key: np.asarray(val, dtype=float) for key, val in conditions.items() 
                     if not str(key).startswith('_')}
    state_variables = wks.phase_record_factory.state_variables
    
    print(f"State variables: {state_variables}")
    print(f"Unitless conditions: {unitless_conds}")
    
    # Call starting_point
    properties = starting_point(unitless_conds, state_variables, wks.phase_record_factory, None)
    
    print(f"\nStarting point properties:")
    print(f"GM shape: {properties.GM.shape}")
    print(f"GM values: {properties.GM.values}")
    print(f"MU shape: {properties.MU.shape}")  
    print(f"MU values: {properties.MU.values}")
    print(f"Phase shape: {properties.Phase.shape}")
    print(f"Phase values: {properties.Phase.values}")
    print(f"NP shape: {properties.NP.shape}")
    print(f"NP values: {properties.NP.values}")
    print(f"X shape: {properties.X.shape}")
    print(f"X values: {properties.X.values}")
    
    # Now run the GPU calculation with extra debugging
    print("\n" + "="*50)
    print("GPU DETAILED DEBUGGING")
    print("="*50)
    
    # Modify GPU code temporarily to extract initial phase data
    # For now, let's run GPU and examine its debug output
    result_gpu = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=True)
    
    print(f"\nFinal GPU result:")
    print(f"GM: {result_gpu.GM.values}")
    print(f"MU: {result_gpu.MU.values}")
    print(f"NP: {result_gpu.NP.values}")
    print(f"Phase: {result_gpu.Phase.values}")
    print(f"X: {result_gpu.X.values}")

def analyze_phase_amounts_discrepancy():
    """Focus specifically on why GPU is getting phase amounts wrong"""
    print("\n" + "="*60)
    print("PHASE AMOUNTS DISCREPANCY ANALYSIS")
    print("="*60)
    
    tdb = Database("NbTi.tdb")
    phases = ["BCC_A2"]
    comps = ["NB", "TI", "VA"] 
    
    conditions = {
        v.X("TI"): 0.1,
        v.T: 800,
        v.P: 101325
    }
    
    # Run CPU calculation
    print("CPU Calculation:")
    cpu_result = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=False)
    
    # Extract and analyze CPU results
    cpu_gm = cpu_result.GM.values.flatten()[0]
    cpu_np = cpu_result.NP.values.flatten()
    cpu_phase = cpu_result.Phase.values.flatten()
    cpu_x = cpu_result.X.values
    
    print(f"  GM: {cpu_gm:.6f}")
    print(f"  Phase amounts (NP): {cpu_np}")
    print(f"  Phase names: {cpu_phase}")
    print(f"  Compositions (X): {cpu_x.shape} -> {cpu_x.flatten()[:6]}")
    
    # Count active phases
    active_cpu_phases = np.sum(~np.isnan(cpu_np) & (cpu_np > 1e-10))
    print(f"  Active phases: {active_cpu_phases}")
    
    # Run GPU calculation  
    print("\nGPU Calculation:")
    gpu_result = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=False)
    
    # Extract and analyze GPU results
    gpu_gm = gpu_result.GM.values.flatten()[0]
    gpu_np = gpu_result.NP.values.flatten()
    gpu_phase = gpu_result.Phase.values.flatten()
    gpu_x = gpu_result.X.values
    
    print(f"  GM: {gpu_gm:.6f}")
    print(f"  Phase amounts (NP): {gpu_np[:5]}...")  # First 5 values
    print(f"  Phase names: {gpu_phase[:5]}...")      # First 5 values
    print(f"  Compositions (X): {gpu_x.shape} -> {gpu_x.flatten()[:6]}")
    
    # Count active GPU phases
    active_gpu_phases = np.sum(gpu_np > 1e-10)
    print(f"  Active phases: {active_gpu_phases}")
    
    # Compare
    print(f"\nComparison:")
    print(f"  GM difference: {abs(cpu_gm - gpu_gm):.6f} J/mol")
    print(f"  Active phases: CPU={active_cpu_phases}, GPU={active_gpu_phases}")
    
    if active_cpu_phases != active_gpu_phases:
        print("  ❌ DIFFERENT NUMBER OF ACTIVE PHASES!")
        print("     This explains the GM difference - GPU found a different equilibrium state")
        
        # Check if GPU found multiple phases vs CPU single phase
        if active_cpu_phases == 1 and active_gpu_phases > 1:
            print("     GPU found multi-phase equilibrium where CPU found single phase")
            print("     This could indicate:")
            print("     1. Different convergence criteria")
            print("     2. Different initial conditions")  
            print("     3. Different phase selection algorithm")
            print("     4. Numerical precision affecting phase stability")

if __name__ == "__main__":
    debug_starting_point_data()
    analyze_phase_amounts_discrepancy()