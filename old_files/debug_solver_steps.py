#!/usr/bin/env python
"""
Debug script to compare CPU vs GPU solver step-by-step.
"""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.workspace import Workspace
from pycalphad.core.starting_point import starting_point
from pycalphad import calculate
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

def debug_cpu_solver_steps():
    """Extract step-by-step data from CPU solver"""
    print("="*60)
    print("CPU SOLVER STEP-BY-STEP DEBUG")
    print("="*60)
    
    # Use one of the problematic conditions from test_script.py
    dbf = Database("NbTi.tdb")
    
    # Use a single problematic condition: very dilute TI at 600K
    conditions = {v.X("TI"): 1e-10, v.T: 600, v.P: 101325}
    
    print(f"Test condition: T=600K, X_TI=1e-10, X_NB≈1.0")
    print(f"This should be pure BCC_A2 phase at equilibrium")
    
    # Run CPU equilibrium with maximum verbosity
    print(f"\nRunning CPU equilibrium calculation...")
    cpu_result = equilibrium(dbf, ['NB', 'TI', 'VA'], ['LIQUID', 'BCC_A2'], 
                           conditions, gpu=False, verbose=True)
    
    print(f"\nCPU FINAL RESULT:")
    print(f"  GM: {cpu_result.GM.values}")
    print(f"  MU: {cpu_result.MU.values}")
    print(f"  NP: {cpu_result.NP.values}")
    print(f"  Phase: {cpu_result.Phase.values}")
    print(f"  Active phases: {cpu_result.Phase.values[cpu_result.NP.values > 1e-6]}")
    
    return cpu_result

def debug_gpu_solver_preparation():
    """Debug GPU solver preparation phase"""
    print("\n" + "="*60)
    print("GPU SOLVER PREPARATION DEBUG")
    print("="*60)
    
    dbf = Database("NbTi.tdb")
    conditions = {v.X("TI"): 1e-10, v.T: 600, v.P: 101325}
    
    print(f"\nExamining GPU starting_point data for same condition...")
    
    # Replicate GPU data preparation process
    wks = Workspace(database=dbf, components=['NB', 'TI', 'VA'], 
                    phases=['LIQUID', 'BCC_A2'], conditions=conditions)
    
    unitless_conds = OrderedDict((key, wks.conditions[key]) for key in wks.conditions.keys())
    state_variables = wks.phase_record_factory.state_variables
    
    # Use the simpler approach - just examine what the GPU would get from the workspace
    print(f"Workspace conditions: {wks.conditions}")
    print(f"State variables: {state_variables}")
    
    # Get the equilibrium result that the GPU preparation would see
    # by calling the workspace equilibrium calculation
    try:
        # Use the workspace's eq property which gives the starting point
        properties = wks.eq
        print(f"Got properties from workspace.eq")
    except Exception as e:
        print(f"Failed to get workspace.eq: {e}")
        print(f"Trying alternative approach...")
        
        # Alternative: call equilibrium with single condition and examine starting point
        return None
    
    print(f"\nGPU STARTING POINT DATA:")
    print(f"  GM: {np.array(properties.GM)}")
    print(f"  MU: {np.array(properties.MU)}")
    print(f"  NP: {np.array(properties.NP)}")
    print(f"  Phase: {np.array(properties.Phase)}")
    
    # Check for issues in starting point
    np_data = np.array(properties.NP).flatten()
    phase_data = np.array(properties.Phase).flatten()
    active_mask = np_data > 1e-10
    num_active = np.sum(active_mask)
    
    print(f"\nSTARTING POINT ANALYSIS:")
    print(f"  Total phases available: {len(np_data)}")
    print(f"  Active phases: {num_active}")
    print(f"  Active phase names: {phase_data[active_mask]}")
    print(f"  Active phase amounts: {np_data[active_mask]}")
    
    if num_active == 0:
        print(f"  ❌ PROBLEM: No active phases in starting point!")
    elif num_active > 1:
        # Check if they're the same phase type
        unique_active_phases = set(phase_data[active_mask]) - {''}
        if len(unique_active_phases) == 1:
            print(f"  ⚠️  Multiple instances of same phase: {unique_active_phases}")
        else:
            print(f"  ✅ Multiple different phases: {unique_active_phases}")
    else:
        print(f"  ✅ Single active phase: {phase_data[active_mask][0]}")
    
    return properties

def run_gpu_with_detailed_debug():
    """Run GPU with maximum debug output"""
    print("\n" + "="*60)
    print("GPU SOLVER EXECUTION DEBUG")
    print("="*60)
    
    dbf = Database("NbTi.tdb")
    conditions = {v.X("TI"): 1e-10, v.T: 600, v.P: 101325}
    
    print(f"\nRunning GPU equilibrium with maximum verbosity...")
    
    try:
        gpu_result = equilibrium(dbf, ['NB', 'TI', 'VA'], ['LIQUID', 'BCC_A2'], 
                               conditions, gpu=True, verbose=True)
        
        print(f"\nGPU FINAL RESULT:")
        print(f"  GM: {gpu_result.GM.values}")
        print(f"  MU: {gpu_result.MU.values}")
        print(f"  NP: {gpu_result.NP.values}")
        print(f"  Phase: {gpu_result.Phase.values}")
        
        return gpu_result
        
    except Exception as e:
        print(f"GPU calculation failed: {e}")
        return None

def compare_cpu_gpu_starting_points(cpu_result, gpu_properties):
    """Compare the starting conditions"""
    print("\n" + "="*60)
    print("CPU vs GPU STARTING POINT COMPARISON")
    print("="*60)
    
    # CPU doesn't expose starting point directly, but we can infer it
    print(f"CPU final result (after convergence):")
    print(f"  GM: {cpu_result.GM.values}")
    print(f"  MU: {cpu_result.MU.values}")
    
    print(f"\nGPU starting point (before solver):")
    print(f"  GM: {np.array(gpu_properties.GM)}")
    print(f"  MU: {np.array(gpu_properties.MU)}")
    
    cpu_gm = float(cpu_result.GM.values)
    gpu_gm = float(np.array(gpu_properties.GM))
    
    print(f"\nGM comparison:")
    print(f"  CPU final: {cpu_gm:.6f}")
    print(f"  GPU start: {gpu_gm:.6f}")
    print(f"  Difference: {abs(cpu_gm - gpu_gm):.6f}")
    
    if abs(cpu_gm - gpu_gm) < 0.001:
        print(f"  ⚠️  WARNING: GPU and CPU values are nearly identical!")
        print(f"  This suggests GPU solver is not running at all.")
    else:
        print(f"  ✅ Values differ significantly - this is expected.")

def main():
    """Main debugging function"""
    print("STEP-BY-STEP CPU vs GPU SOLVER DEBUG")
    print("Investigating why GPU solver fails on dilute compositions")
    
    # Step 1: Debug CPU solver
    cpu_result = debug_cpu_solver_steps()
    
    # Step 2: Debug GPU preparation
    gpu_properties = debug_gpu_solver_preparation()
    
    # Step 3: Compare starting points
    compare_cpu_gpu_starting_points(cpu_result, gpu_properties)
    
    # Step 4: Run GPU with detailed debug
    gpu_result = run_gpu_with_detailed_debug()
    
    # Step 5: Final comparison
    if gpu_result is not None:
        print("\n" + "="*60)
        print("FINAL CPU vs GPU COMPARISON")
        print("="*60)
        
        cpu_gm = float(cpu_result.GM.values)
        gpu_gm = float(gpu_result.GM.values)
        
        print(f"Final GM values:")
        print(f"  CPU: {cpu_gm:.6f}")
        print(f"  GPU: {gpu_gm:.6f}")
        print(f"  Difference: {abs(cpu_gm - gpu_gm):.6f}")
        
        if abs(cpu_gm - gpu_gm) < 0.001:
            print(f"  🚨 CONFIRMED: GPU is returning starting point data!")
            print(f"  The GPU equilibrium solver is not running.")
        else:
            print(f"  ✅ GPU solver appears to be working.")

if __name__ == "__main__":
    main()