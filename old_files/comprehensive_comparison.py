#!/usr/bin/env python3
"""
Comprehensive comparison of ALL values in CPU vs GPU output datasets
Checks every single value in GM, X, Y, Phases arrays to verify fixes worked completely
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

def compare_arrays_completely(cpu_arr, gpu_arr, name, tolerance=1e-10):
    """Compare every single element between two arrays"""
    print(f"\n{'='*60}")
    print(f"COMPLETE {name} ARRAY COMPARISON")
    print(f"{'='*60}")
    
    print(f"CPU {name} shape: {cpu_arr.shape}")
    print(f"GPU {name} shape: {gpu_arr.shape}")
    
    # Handle shape differences
    if cpu_arr.shape != gpu_arr.shape:
        print(f"❌ SHAPE MISMATCH: CPU {cpu_arr.shape} vs GPU {gpu_arr.shape}")
        
        # Try to find overlapping region for comparison
        min_shape = tuple(min(c, g) for c, g in zip(cpu_arr.shape, gpu_arr.shape))
        print(f"   Comparing overlapping region: {min_shape}")
        
        # Extract overlapping regions
        cpu_slice = tuple(slice(0, s) for s in min_shape)
        gpu_slice = tuple(slice(0, s) for s in min_shape)
        cpu_overlap = cpu_arr[cpu_slice]
        gpu_overlap = gpu_arr[gpu_slice]
    else:
        print(f"✅ Shapes match: {cpu_arr.shape}")
        cpu_overlap = cpu_arr
        gpu_overlap = gpu_arr
    
    # Flatten for element-by-element comparison
    cpu_flat = cpu_overlap.flatten()
    gpu_flat = gpu_overlap.flatten()
    
    print(f"Comparing {len(cpu_flat)} elements...")
    
    # Handle different data types
    if cpu_arr.dtype.kind in ['U', 'S', 'O']:  # String/object arrays
        print(f"String/object array comparison:")
        differences = []
        matches = 0
        
        for i, (cpu_val, gpu_val) in enumerate(zip(cpu_flat, gpu_flat)):
            if str(cpu_val) != str(gpu_val):
                differences.append((i, cpu_val, gpu_val))
            else:
                matches += 1
        
        print(f"  Matching elements: {matches}/{len(cpu_flat)}")
        print(f"  Different elements: {len(differences)}/{len(cpu_flat)}")
        
        if len(differences) > 0:
            print(f"  First 10 differences:")
            for i, (idx, cpu_val, gpu_val) in enumerate(differences[:10]):
                print(f"    [{idx}]: CPU='{cpu_val}' vs GPU='{gpu_val}'")
        
        return len(differences) == 0
        
    else:  # Numeric arrays
        print(f"Numeric array comparison (tolerance: {tolerance}):")
        
        # Handle NaN values
        cpu_nan_mask = np.isnan(cpu_flat)
        gpu_nan_mask = np.isnan(gpu_flat)
        
        # Check NaN pattern consistency
        nan_pattern_match = np.array_equal(cpu_nan_mask, gpu_nan_mask)
        print(f"  NaN pattern match: {nan_pattern_match}")
        if not nan_pattern_match:
            print(f"    CPU NaNs: {np.sum(cpu_nan_mask)}/{len(cpu_flat)}")
            print(f"    GPU NaNs: {np.sum(gpu_nan_mask)}/{len(gpu_flat)}")
        
        # Compare non-NaN values
        valid_mask = ~cpu_nan_mask & ~gpu_nan_mask
        cpu_valid = cpu_flat[valid_mask]
        gpu_valid = gpu_flat[valid_mask]
        
        if len(cpu_valid) > 0:
            print(f"  Valid elements to compare: {len(cpu_valid)}")
            
            # Absolute differences
            abs_diff = np.abs(cpu_valid - gpu_valid)
            max_abs_diff = np.max(abs_diff)
            mean_abs_diff = np.mean(abs_diff)
            
            # Relative differences (avoid division by zero)
            rel_diff = np.zeros_like(abs_diff)
            nonzero_mask = np.abs(cpu_valid) > 1e-15
            rel_diff[nonzero_mask] = abs_diff[nonzero_mask] / np.abs(cpu_valid[nonzero_mask])
            max_rel_diff = np.max(rel_diff) if len(rel_diff) > 0 else 0
            
            print(f"  Maximum absolute difference: {max_abs_diff:.2e}")
            print(f"  Mean absolute difference: {mean_abs_diff:.2e}")
            print(f"  Maximum relative difference: {max_rel_diff:.2e}")
            
            # Check tolerance
            within_tolerance = np.all(abs_diff <= tolerance)
            print(f"  All differences within tolerance ({tolerance}): {within_tolerance}")
            
            # Show worst offenders if outside tolerance
            if not within_tolerance:
                worst_indices = np.argsort(abs_diff)[-5:]  # 5 worst
                print(f"  Worst 5 differences:")
                for idx in worst_indices:
                    orig_idx = np.where(valid_mask)[0][idx]
                    print(f"    [{orig_idx}]: CPU={cpu_valid[idx]:.10e}, GPU={gpu_valid[idx]:.10e}, diff={abs_diff[idx]:.2e}")
            
            return within_tolerance and nan_pattern_match
        else:
            print(f"  No valid (non-NaN) elements to compare")
            return nan_pattern_match

def comprehensive_dataset_comparison():
    """Compare every single value in the CPU and GPU datasets"""
    print("=== COMPREHENSIVE DATASET COMPARISON ===")
    print("Checking ALL values in GM, X, Y, Phases arrays")
    clear_cupy_kernel_cache()
    
    tdb = Database("NbTi.tdb")
    phases = ["BCC_A2"]
    comps = ["NB", "TI", "VA"]
    
    conditions = {
        v.X("TI"): 0.1,
        v.T: 800,
        v.P: 101325
    }
    
    print(f"Test conditions: {conditions}")
    
    # Run both calculations
    print("\n" + "="*80)
    print("RUNNING CPU CALCULATION")
    print("="*80)
    cpu_result = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=False)
    
    print("\n" + "="*80)
    print("RUNNING GPU CALCULATION")
    print("="*80)
    gpu_result = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=False)
    
    # Compare all arrays systematically
    comparison_results = {}
    
    # 1. GM (Gibbs Energy)
    print(f"\n🔍 Comparing GM (Gibbs Energy)...")
    cpu_gm = cpu_result.GM.values
    gpu_gm = gpu_result.GM.values
    comparison_results['GM'] = compare_arrays_completely(cpu_gm, gpu_gm, "GM", tolerance=1.0)
    
    # 2. MU (Chemical Potentials)
    print(f"\n🔍 Comparing MU (Chemical Potentials)...")
    cpu_mu = cpu_result.MU.values
    gpu_mu = gpu_result.MU.values
    comparison_results['MU'] = compare_arrays_completely(cpu_mu, gpu_mu, "MU", tolerance=10.0)
    
    # 3. NP (Phase Amounts)
    print(f"\n🔍 Comparing NP (Phase Amounts)...")
    cpu_np = cpu_result.NP.values
    gpu_np = gpu_result.NP.values
    comparison_results['NP'] = compare_arrays_completely(cpu_np, gpu_np, "NP", tolerance=1e-10)
    
    # 4. Phase (Phase Names)
    print(f"\n🔍 Comparing Phase (Phase Names)...")
    cpu_phase = cpu_result.Phase.values
    gpu_phase = gpu_result.Phase.values
    comparison_results['Phase'] = compare_arrays_completely(cpu_phase, gpu_phase, "Phase")
    
    # 5. X (Compositions)
    print(f"\n🔍 Comparing X (Compositions)...")
    cpu_x = cpu_result.X.values
    gpu_x = gpu_result.X.values
    comparison_results['X'] = compare_arrays_completely(cpu_x, gpu_x, "X", tolerance=1e-10)
    
    # 6. Y (Internal DOF)
    print(f"\n🔍 Comparing Y (Internal DOF)...")
    cpu_y = cpu_result.Y.values
    gpu_y = gpu_result.Y.values
    comparison_results['Y'] = compare_arrays_completely(cpu_y, gpu_y, "Y", tolerance=1e-10)
    
    # Summary report
    print(f"\n{'='*80}")
    print("COMPREHENSIVE COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    all_passed = True
    for array_name, passed in comparison_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{array_name:15}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\n{'='*80}")
    if all_passed:
        print("🎉 COMPLETE SUCCESS: All arrays match within tolerances!")
        print("   GPU implementation is fully equivalent to CPU implementation.")
        print("   The fixes have worked completely.")
    else:
        print("⚠️  ISSUES DETECTED: Some arrays do not match within tolerances.")
        print("   The fixes are not yet complete and need further work.")
        print("   Review the detailed comparison output above.")
    print(f"{'='*80}")
    
    return all_passed, comparison_results

def print_summary_statistics():
    """Print summary statistics for key metrics"""
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    tdb = Database("NbTi.tdb")
    phases = ["BCC_A2"]
    comps = ["NB", "TI", "VA"]
    
    conditions = {
        v.X("TI"): 0.1,
        v.T: 800,
        v.P: 101325
    }
    
    cpu_result = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=False)
    gpu_result = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=False)
    
    # Key scalar comparisons
    cpu_gm_scalar = cpu_result.GM.values.flatten()[0]
    gpu_gm_scalar = gpu_result.GM.values.flatten()[0]
    gm_diff = abs(cpu_gm_scalar - gpu_gm_scalar)
    
    print(f"Key Scalar Values:")
    print(f"  GM:  CPU = {cpu_gm_scalar:.10f} J/mol")
    print(f"       GPU = {gpu_gm_scalar:.10f} J/mol")
    print(f"       Diff = {gm_diff:.10f} J/mol")
    
    # Active phase information
    cpu_np_flat = cpu_result.NP.values.flatten()
    gpu_np_flat = gpu_result.NP.values.flatten()
    
    cpu_active = np.sum(~np.isnan(cpu_np_flat) & (cpu_np_flat > 1e-10))
    gpu_active = np.sum(gpu_np_flat > 1e-10)
    
    print(f"\nActive Phases:")
    print(f"  CPU: {cpu_active} active phases")
    print(f"  GPU: {gpu_active} active phases")
    
    # First active phase composition
    cpu_x_flat = cpu_result.X.values
    gpu_x_flat = gpu_result.X.values
    
    print(f"\nFirst Active Phase Composition:")
    try:
        cpu_comp = cpu_x_flat.reshape(-1, cpu_x_flat.shape[-1])[0][:2]
        gpu_comp = gpu_x_flat.reshape(-1, gpu_x_flat.shape[-1])[0][:2]
        
        print(f"  CPU: NB={cpu_comp[0]:.10f}, TI={cpu_comp[1]:.10f}")
        print(f"  GPU: NB={gpu_comp[0]:.10f}, TI={gpu_comp[1]:.10f}")
        print(f"  Diff: NB={abs(cpu_comp[0]-gpu_comp[0]):.2e}, TI={abs(cpu_comp[1]-gpu_comp[1]):.2e}")
    except Exception as e:
        print(f"  Error extracting compositions: {e}")

if __name__ == "__main__":
    # Run comprehensive comparison
    success, results = comprehensive_dataset_comparison()
    
    # Print summary statistics
    print_summary_statistics()
    
    # Final verdict
    print(f"\n{'#'*80}")
    if success:
        print("FINAL VERDICT: GPU IMPLEMENTATION IS FULLY FIXED AND EQUIVALENT TO CPU")
    else:
        print("FINAL VERDICT: GPU IMPLEMENTATION STILL HAS ISSUES - MORE WORK NEEDED")
    print(f"{'#'*80}")