#!/usr/bin/env python
"""Check if GPU and CPU use identical grids."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

# Monkey patch to capture grid info
import pycalphad.core.workspace as ws_module
import pycalphad.gpu.gpu_equilibrium as gpu_module

captured_grids = {'cpu': None, 'gpu': None}

# Patch CPU calculate call
original_ws_recompute = ws_module.Workspace.recompute

def patched_ws_recompute(self):
    # Call original first
    original_ws_recompute(self)
    
    # Now capture the grid that was passed to starting_point
    # We'll patch calculate to capture it
    from pycalphad import calculate
    original_calculate = calculate
    
    def capturing_calculate(*args, **kwargs):
        result = original_calculate(*args, **kwargs)
        if captured_grids['cpu'] is None:
            captured_grids['cpu'] = {
                'GM': result.GM.copy() if hasattr(result.GM, 'copy') else result.GM,
                'Phase': result.Phase.copy() if hasattr(result.Phase, 'copy') else result.Phase,
                'shape': result.GM.shape
            }
        return result
    
    # Temporarily replace calculate
    import pycalphad
    pycalphad.calculate = capturing_calculate
    ws_module.calculate = capturing_calculate

ws_module.Workspace.recompute = patched_ws_recompute

# Patch GPU calculate call  
original_gpu_equilibrium = gpu_module.gpu_equilibrium

def patched_gpu_equilibrium(wks_obj, verbose=False, validate_code=False):
    # Patch calculate in GPU path
    from pycalphad import calculate
    original_calculate = calculate
    
    def capturing_calculate(*args, **kwargs):
        result = original_calculate(*args, **kwargs)
        if captured_grids['gpu'] is None:
            captured_grids['gpu'] = {
                'GM': result.GM.copy() if hasattr(result.GM, 'copy') else result.GM,
                'Phase': result.Phase.copy() if hasattr(result.Phase, 'copy') else result.Phase,
                'shape': result.GM.shape
            }
        return result
    
    # Temporarily replace calculate
    import pycalphad
    pycalphad.calculate = capturing_calculate
    
    # Call original
    return original_gpu_equilibrium(wks_obj, verbose, validate_code)

gpu_module.gpu_equilibrium = patched_gpu_equilibrium

def test_grid_comparison():
    """Compare grids used by CPU and GPU."""
    
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    phases = ['AU2BI_C15', 'BCC_A2', 'FCC_A1', 'HCP_A3', 'LIQUID', 'RHOMBOHEDRAL_A7']
    
    conditions = {
        v.X('BI'): 0.1,
        v.T: 400,
        v.P: 101325
    }
    
    print("="*70)
    print("GRID COMPARISON TEST")
    print("="*70)
    
    # Reset captures
    captured_grids['cpu'] = None
    captured_grids['gpu'] = None
    
    # CPU calculation
    print("\n--- CPU CALCULATION ---")
    result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
    
    # GPU calculation
    print("\n--- GPU CALCULATION ---")
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    
    # Compare grids
    print("\n" + "="*50)
    print("GRID COMPARISON")
    print("="*50)
    
    if captured_grids['cpu'] is not None and captured_grids['gpu'] is not None:
        cpu_grid = captured_grids['cpu']
        gpu_grid = captured_grids['gpu']
        
        print(f"CPU grid shape: {cpu_grid['shape']}")
        print(f"GPU grid shape: {gpu_grid['shape']}")
        
        if cpu_grid['shape'] == gpu_grid['shape']:
            print("✅ Same grid shapes")
            
            # Compare GM values
            gm_diff = np.max(np.abs(cpu_grid['GM'] - gpu_grid['GM']))
            print(f"Max GM difference: {gm_diff:.2e}")
            
            if gm_diff < 1e-10:
                print("✅ Identical GM values")
            else:
                print("❌ Different GM values!")
                
            # Compare phases
            phase_match = np.all(cpu_grid['Phase'] == gpu_grid['Phase'])
            if phase_match:
                print("✅ Identical phase assignments")
            else:
                print("❌ Different phase assignments!")
                # Find differences
                diff_mask = cpu_grid['Phase'] != gpu_grid['Phase']
                n_diff = np.sum(diff_mask)
                print(f"   Number of different points: {n_diff}")
                if n_diff < 20:
                    diff_indices = np.where(diff_mask)
                    for i in range(min(n_diff, 10)):
                        idx = tuple(d[i] for d in diff_indices)
                        print(f"   Point {idx}: CPU={cpu_grid['Phase'][idx]} GPU={gpu_grid['Phase'][idx]}")
        else:
            print("❌ Different grid shapes!")
    else:
        print("Failed to capture grids")

if __name__ == "__main__":
    test_grid_comparison()