#!/usr/bin/env python3
"""Detailed comparison between CPU and GPU pycalphad calculations"""

import numpy as np
import pycalphad as pyc
from pycalphad import Database, equilibrium, variables as v
import time
import os
import glob

def compare_arrays(cpu_arr, gpu_arr, name, tolerance=1e-6):
    """Compare two arrays and report differences"""
    print(f"\n=== Comparing {name} ===")
    
    if cpu_arr is None and gpu_arr is None:
        print("Both arrays are None")
        return True
    
    if cpu_arr is None or gpu_arr is None:
        print(f"One array is None: CPU={cpu_arr is not None}, GPU={gpu_arr is not None}")
        return False
    
    cpu_arr = np.asarray(cpu_arr)
    gpu_arr = np.asarray(gpu_arr)
    
    print(f"CPU shape: {cpu_arr.shape}, GPU shape: {gpu_arr.shape}")
    print(f"CPU dtype: {cpu_arr.dtype}, GPU dtype: {gpu_arr.dtype}")
    
    if cpu_arr.shape != gpu_arr.shape:
        print("SHAPES DIFFER!")
        return False
    
    # Compare values
    if np.allclose(cpu_arr, gpu_arr, rtol=tolerance, atol=tolerance):
        print("✓ Arrays match within tolerance")
        return True
    else:
        diff = np.abs(cpu_arr - gpu_arr)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        rel_diff = np.abs((cpu_arr - gpu_arr) / (cpu_arr + 1e-15))
        max_rel_diff = np.max(rel_diff)
        
        print(f"✗ Arrays differ!")
        print(f"  Max absolute difference: {max_diff}")
        print(f"  Mean absolute difference: {mean_diff}")
        print(f"  Max relative difference: {max_rel_diff}")
        
        # Show first few differing values
        flat_cpu = cpu_arr.flatten()
        flat_gpu = gpu_arr.flatten()
        flat_diff = diff.flatten()
        
        print(f"  First 5 CPU values: {flat_cpu[:5]}")
        print(f"  First 5 GPU values: {flat_gpu[:5]}")
        print(f"  First 5 differences: {flat_diff[:5]}")
        
        # Find largest differences
        worst_indices = np.argsort(flat_diff)[-5:]
        print(f"  5 worst differences:")
        for i in worst_indices:
            print(f"    Index {i}: CPU={flat_cpu[i]:.6e}, GPU={flat_gpu[i]:.6e}, diff={flat_diff[i]:.6e}")
        
        return False

def clear_cupy_kernel_cache():
    """Clear CuPy kernel cache to ensure fresh compilation"""
    cache_dir = os.path.expanduser("~/.cupy/kernel_cache")
    if os.path.exists(cache_dir):
        cubin_files = glob.glob(os.path.join(cache_dir, "*.cubin"))
        if cubin_files:
            print(f"[CACHE] Clearing {len(cubin_files)} .cubin files from CuPy kernel cache...")
            for cubin_file in cubin_files:
                try:
                    os.remove(cubin_file)
                except OSError as e:
                    print(f"[CACHE] Warning: Could not remove {cubin_file}: {e}")
            print("[CACHE] CuPy kernel cache cleared.")
        else:
            print("[CACHE] No .cubin files found in CuPy kernel cache.")
    else:
        print("[CACHE] CuPy kernel cache directory not found.")

def run_detailed_comparison():
    """Run a detailed comparison between CPU and GPU calculations"""
    print("=== DETAILED GPU vs CPU COMPARISON ===")
    
    # Clear CuPy kernel cache before testing
    clear_cupy_kernel_cache()
    
    # Load database and set up simple test case
    tdb = Database("NbTi.tdb")
    phases = ["BCC_A2"]  # Start with single phase
    comps = ["NB", "TI", "VA"]
    
    # Simple single-point conditions for easier debugging
    conditions = {
        v.X("TI"): 0.1,
        v.T: 800,
        v.P: 101325
    }
    
    # Also test multi-point conditions that match test_script.py structure
    multi_conditions = {
        v.X("TI"): [0.1, 0.2],  # Two composition points
        v.T: [800, 900],        # Two temperature points
        v.P: 101325
    }
    
    print(f"Database: NbTi.tdb")
    print(f"Phases: {phases}")
    print(f"Components: {comps}")
    print(f"Conditions: {conditions}")
    
    print("\n" + "="*50)
    print("RUNNING CPU CALCULATION")
    print("="*50)
    
    # CPU calculation
    t0 = time.time()
    try:
        cpu_result = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=True)
        t1 = time.time()
        cpu_time = t1 - t0
        print(f"CPU calculation completed in {cpu_time:.3f}s")
        
        # Debug: Print CPU result structure details
        print(f"\n[DEBUG] CPU result type: {type(cpu_result)}")
        print(f"[DEBUG] CPU result dimensions: {cpu_result.dims}")
        print(f"[DEBUG] CPU result coordinates: {list(cpu_result.coords.keys())}")
        print(f"[DEBUG] CPU result data_vars: {list(cpu_result.data_vars.keys())}")
        for coord_name, coord_data in cpu_result.coords.items():
            print(f"[DEBUG] Coordinate {coord_name}: shape={coord_data.shape}, values={coord_data.values}")
        
        cpu_success = True
    except Exception as e:
        print(f"CPU calculation failed: {e}")
        cpu_result = None
        cpu_success = False
        cpu_time = 0
    
    print("\n" + "="*50)
    print("RUNNING GPU CALCULATION")
    print("="*50)
    
    # GPU calculation
    t2 = time.time()
    try:
        gpu_result = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=True)
        t3 = time.time()
        gpu_time = t3 - t2
        print(f"GPU calculation completed in {gpu_time:.3f}s")
        
        # Debug: Print GPU result structure details
        print(f"\n[DEBUG] GPU result type: {type(gpu_result)}")
        if hasattr(gpu_result, 'dims'):
            print(f"[DEBUG] GPU result dimensions: {gpu_result.dims}")
        if hasattr(gpu_result, 'coords'):
            print(f"[DEBUG] GPU result coordinates: {list(gpu_result.coords.keys())}")
            for coord_name, coord_data in gpu_result.coords.items():
                coord_values = coord_data.values if hasattr(coord_data, 'values') else coord_data
                print(f"[DEBUG] Coordinate {coord_name}: shape={coord_values.shape}, values={coord_values}")
        if hasattr(gpu_result, 'data_vars'):
            print(f"[DEBUG] GPU result data_vars: {list(gpu_result.data_vars.keys())}")
            for var_name in ['GM', 'MU', 'NP', 'Phase', 'X']:
                if hasattr(gpu_result, var_name):
                    var_data = getattr(gpu_result, var_name)
                    if hasattr(var_data, 'values'):
                        print(f"[DEBUG] {var_name} shape: {var_data.values.shape}, dtype: {var_data.values.dtype}")
                    else:
                        print(f"[DEBUG] {var_name} shape: {var_data.shape}, dtype: {var_data.dtype}")
        
        gpu_success = True
    except Exception as e:
        print(f"GPU calculation failed: {e}")
        gpu_result = None
        gpu_success = False
        gpu_time = 0
    
    print("\n" + "="*50)
    print("COMPARISON RESULTS")
    print("="*50)
    
    print(f"CPU Success: {cpu_success}")
    print(f"GPU Success: {gpu_success}")
    print(f"CPU Time: {cpu_time:.3f}s")
    print(f"GPU Time: {gpu_time:.3f}s")
    
    if gpu_time > 0 and cpu_time > 0:
        speedup = cpu_time / gpu_time
        print(f"Speedup: {speedup:.2f}x")
    
    if not (cpu_success and gpu_success):
        print("Cannot compare results - one or both calculations failed")
        return
    
    # Compare key outputs
    print(f"\n" + "="*30)
    print("COMPARING RESULT PROPERTIES")
    print("="*30)
    
    # Compare phases present
    if hasattr(cpu_result, 'Phase'):
        cpu_phases = cpu_result.Phase.values if hasattr(cpu_result.Phase, 'values') else cpu_result.Phase
        print(f"\nCPU phases: {np.unique(cpu_phases.flatten()) if hasattr(cpu_phases, 'flatten') else cpu_phases}")
    else:
        print(f"\nCPU phases: N/A")
        
    if hasattr(gpu_result, 'Phase'):
        gpu_phases = gpu_result.Phase.values if hasattr(gpu_result.Phase, 'values') else gpu_result.Phase  
        print(f"GPU phases: {np.unique(gpu_phases.flatten()) if hasattr(gpu_phases, 'flatten') else gpu_phases}")
    else:
        print(f"GPU phases: N/A")
    
    # Compare main thermodynamic properties
    properties_to_compare = ['GM', 'MU', 'NP', 'X']
    
    for prop in properties_to_compare:
        if hasattr(cpu_result, prop) and hasattr(gpu_result, prop):
            cpu_val = getattr(cpu_result, prop)
            gpu_val = getattr(gpu_result, prop)
            
            if hasattr(cpu_val, 'values'):
                cpu_val = cpu_val.values
            if hasattr(gpu_val, 'values'):
                gpu_val = gpu_val.values
                
            compare_arrays(cpu_val, gpu_val, prop)
        else:
            print(f"\nProperty {prop} not found in both results")
    
    # Detailed coordinate comparison
    print(f"\n" + "="*30)
    print("COMPARING COORDINATES")
    print("="*30)
    
    for coord_name in cpu_result.coords:
        if coord_name in gpu_result.coords:
            cpu_coord = cpu_result.coords[coord_name].values
            # GPU coords are stored directly as numpy arrays in LightDataset
            gpu_coord = gpu_result.coords[coord_name]
            if hasattr(gpu_coord, 'values'):
                gpu_coord = gpu_coord.values
            compare_arrays(cpu_coord, gpu_coord, f"coord_{coord_name}")
        else:
            print(f"Coordinate {coord_name} missing in GPU result")
    
    print(f"\n" + "="*30)
    print("SUMMARY")
    print("="*30)
    
    # Final summary
    if cpu_success and gpu_success:
        print("Both calculations completed successfully")
        print("Check individual comparisons above for numerical accuracy")
    else:
        print("One or both calculations failed - need to debug basic functionality first")

if __name__ == "__main__":
    run_detailed_comparison()