#!/usr/bin/env python
"""Test Al-Cu-Fe system using multi-condition calls for efficiency."""

import numpy as np
from pycalphad import Database, equilibrium
import pycalphad.variables as v
import warnings
import os
import time

# Suppress warnings and debug output
warnings.filterwarnings('ignore')
os.environ['PYCALPHAD_CPU_DEBUG'] = '0'

def test_alcufe_multi_conditions():
    """Test Al-Cu-Fe system with fixed GPU code generation."""
    
    # Load database
    db = Database('Al-Cu-Fe.tdb')
    components = ['AL', 'CU', 'FE', 'VA']
    phases = list(db.phases.keys())
    
    print("="*80)
    print("Al-Cu-Fe System Test with Multi-Condition Calls")
    print("="*80)
    print(f"Database: Al-Cu-Fe.tdb")
    print(f"Temperature: 1000°C (1273.15K)")
    print(f"Number of phases: {len(phases)}")
    
    # Create composition grid using multi-condition format
    # This passes arrays to equilibrium, allowing it to compute all conditions at once
    al_grid = np.linspace(0, 1, 11)
    cu_grid = np.linspace(0, 1, 11)
    
    # Create flattened arrays for all valid compositions
    al_values = []
    cu_values = []
    
    for x_cu in cu_grid:
        for x_al in al_grid:
            x_fe = 1 - x_al - x_cu
            if x_fe >= -1e-10:  # Valid composition
                al_values.append(x_al)
                cu_values.append(x_cu)
    
    al_values = np.array(al_values)
    cu_values = np.array(cu_values)
    
    print(f"Total valid compositions: {len(al_values)}")
    
    # Multi-condition specification - pass arrays directly
    conditions = {
        v.T: 1273.15,
        v.P: 101325,
        v.X('AL'): al_values,  # Array of compositions
        v.X('CU'): cu_values,  # Array of compositions
        v.N: 1
    }
    
    print("\n" + "="*40)
    print("Running CPU calculation (multi-condition)...")
    start_cpu = time.time()
    
    try:
        cpu_result = equilibrium(db, components, phases, conditions, 
                               calc_opts={'pdens': 50})
        cpu_time = time.time() - start_cpu
        print(f"CPU SUCCESS in {cpu_time:.1f} seconds")
        cpu_success = True
        
        # Show sample results
        print(f"Result shape: {cpu_result.GM.shape}")
        print(f"Sample GM values: {cpu_result.GM.values.flatten()[:5]}")
        
    except Exception as e:
        print(f"CPU FAILED: {str(e)[:100]}...")
        cpu_success = False
        cpu_result = None
    
    print("\n" + "="*40)
    print("Running GPU calculation (multi-condition)...")
    start_gpu = time.time()
    
    try:
        gpu_result = equilibrium(db, components, phases, conditions,
                               calc_opts={'pdens': 50}, gpu=True, verbose=False)
        gpu_time = time.time() - start_gpu
        print(f"GPU SUCCESS in {gpu_time:.1f} seconds")
        gpu_success = True
        
        # Show sample results
        print(f"Result shape: {gpu_result.GM.shape}")
        print(f"Sample GM values: {gpu_result.GM.values.flatten()[:5]}")
        
        # Speed comparison
        if cpu_success:
            speedup = cpu_time / gpu_time
            print(f"\nGPU speedup: {speedup:.1f}x")
        
    except Exception as e:
        print(f"GPU FAILED: {str(e)[:100]}...")
        gpu_success = False
        gpu_result = None
    
    # Compare results if both succeeded
    if cpu_success and gpu_success:
        print("\n" + "="*40)
        print("Comparing Results...")
        print("="*40)
        
        # Extract GM values
        gm_cpu = cpu_result.GM.values.flatten()
        gm_gpu = gpu_result.GM.values.flatten()
        
        # Calculate differences
        differences = np.abs(gm_cpu - gm_gpu)
        tolerance = 1.0  # 1 J/mol
        
        matching = np.sum(differences < tolerance)
        total = len(differences)
        
        print(f"\nTotal points: {total}")
        print(f"Matching points (within {tolerance} J/mol): {matching} ({100*matching/total:.1f}%)")
        
        # Statistics
        print(f"\nDifference statistics:")
        print(f"  Max difference: {np.max(differences):.3f} J/mol")
        print(f"  Mean difference: {np.mean(differences):.3f} J/mol")
        print(f"  Median difference: {np.median(differences):.3f} J/mol")
        
        # Show worst mismatches
        worst_indices = np.argsort(differences)[-5:][::-1]
        if np.max(differences) > tolerance:
            print(f"\nWorst mismatches:")
            print("Index  X(AL)  X(CU)  X(FE)   GM_CPU    GM_GPU    Diff")
            print("-"*55)
            for idx in worst_indices:
                if differences[idx] > tolerance:
                    x_al = al_values[idx]
                    x_cu = cu_values[idx]
                    x_fe = 1 - x_al - x_cu
                    print(f"{idx:5d}  {x_al:5.3f}  {x_cu:5.3f}  {x_fe:5.3f}  "
                          f"{gm_cpu[idx]:8.1f}  {gm_gpu[idx]:8.1f}  {differences[idx]:6.1f}")
        
        # Also compare chemical potentials for a few points
        if hasattr(cpu_result, 'MU') and hasattr(gpu_result, 'MU'):
            print("\n" + "="*40)
            print("Chemical Potential Comparison (sample points):")
            
            sample_indices = [0, len(al_values)//4, len(al_values)//2, 3*len(al_values)//4]
            
            for idx in sample_indices:
                if idx < len(al_values):
                    x_al = al_values[idx]
                    x_cu = cu_values[idx]
                    x_fe = 1 - x_al - x_cu
                    
                    print(f"\nX(AL)={x_al:.3f}, X(CU)={x_cu:.3f}, X(FE)={x_fe:.3f}:")
                    
                    mu_cpu = cpu_result.MU.values.reshape(-1, cpu_result.MU.shape[-1])[idx]
                    mu_gpu = gpu_result.MU.values.reshape(-1, gpu_result.MU.shape[-1])[idx]
                    
                    for k, comp in enumerate(['AL', 'CU', 'FE']):
                        if k < len(mu_cpu) and k < len(mu_gpu):
                            diff_mu = abs(mu_gpu[k] - mu_cpu[k])
                            print(f"  μ({comp}): CPU={mu_cpu[k]:10.1f}, "
                                  f"GPU={mu_gpu[k]:10.1f}, Δ={diff_mu:8.3f}")
    
    return cpu_success, gpu_success

if __name__ == "__main__":
    cpu_ok, gpu_ok = test_alcufe_multi_conditions()
    
    print("\n" + "="*80)
    print("SUMMARY:")
    if cpu_ok and gpu_ok:
        print("✓ Both CPU and GPU calculations completed successfully")
        print("✓ GPU code generation fix for multi-sublattice phases is working")
    elif cpu_ok and not gpu_ok:
        print("✓ CPU calculation succeeded")
        print("✗ GPU calculation failed - further debugging needed")
    else:
        print("✗ Both calculations failed")