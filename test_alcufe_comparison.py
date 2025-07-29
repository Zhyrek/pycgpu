#!/usr/bin/env python
"""Compare CPU vs GPU results for Al-Cu-Fe system at 1000C."""

import numpy as np
from pycalphad import Database, equilibrium
import pycalphad.variables as v
import time

# Load database
db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']  # VA is always included
phases = list(db.phases.keys())

# Temperature in Celsius converted to Kelvin
T_celsius = 1000
T_kelvin = T_celsius + 273.15

# Define conditions - scan entire composition space
conditions = {
    v.T: T_kelvin,
    v.P: 101325,
    v.X('AL'): (0, 1, 0.1),  # 11 points from 0 to 1
    v.X('CU'): (0, 1, 0.1),  # 11 points from 0 to 1
    v.N: 1
}

print("=" * 80)
print(f"Al-Cu-Fe System Comparison at {T_celsius}°C ({T_kelvin}K)")
print("=" * 80)
print(f"Database: Al-Cu-Fe.tdb")
print(f"Components: {components}")
print(f"Phases available: {phases}")
print(f"Composition grid: X(AL) and X(CU) from 0 to 1 in steps of 0.1")
print(f"Total points: 11 x 11 = 121 compositions")

# Run CPU calculation
print("\n" + "=" * 40)
print("Running CPU calculation...")
start_cpu = time.time()
try:
    cpu_result = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 50})
    cpu_time = time.time() - start_cpu
    print(f"CPU calculation completed in {cpu_time:.2f} seconds")
    cpu_success = True
except Exception as e:
    print(f"CPU calculation FAILED: {e}")
    cpu_success = False
    cpu_result = None

# Run GPU calculation  
print("\n" + "=" * 40)
print("Running GPU calculation...")
start_gpu = time.time()
try:
    gpu_result = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 50}, gpu=True, verbose=False)
    gpu_time = time.time() - start_gpu
    print(f"GPU calculation completed in {gpu_time:.2f} seconds")
    gpu_success = True
except Exception as e:
    print(f"GPU calculation FAILED: {e}")
    gpu_success = False
    gpu_result = None

# Compare results if both succeeded
if cpu_success and gpu_success:
    print("\n" + "=" * 40)
    print("Comparing Results...")
    print("=" * 40)
    
    # Create composition grids
    al_values = np.linspace(0, 1, 11)
    cu_values = np.linspace(0, 1, 11)
    
    # Compare each point
    total_points = 0
    matching_points = 0
    mismatched_points = []
    
    print("\nDetailed comparison:")
    print("-" * 80)
    print(f"{'X(AL)':<8} {'X(CU)':<8} {'X(FE)':<8} {'GM_CPU':<12} {'GM_GPU':<12} {'ΔGM':<10} {'Status':<10}")
    print("-" * 80)
    
    for i, x_al in enumerate(al_values):
        for j, x_cu in enumerate(cu_values):
            x_fe = 1 - x_al - x_cu
            
            # Skip invalid compositions where X(FE) < 0
            if x_fe < -1e-10:
                continue
                
            total_points += 1
            
            # Get the index in the flattened result array
            # The results are arranged as: [X_CU=0 for all X_AL], [X_CU=0.1 for all X_AL], etc.
            idx = j * len(al_values) + i
            
            # Extract GM values
            gm_cpu = cpu_result.GM.values.flatten()[idx]
            gm_gpu = gpu_result.GM.values.flatten()[idx]
            
            # Calculate difference
            diff = abs(gm_gpu - gm_cpu)
            
            # Check if within tolerance (1 J/mol)
            tolerance = 1.0
            if diff < tolerance:
                status = "MATCH"
                matching_points += 1
            else:
                status = "MISMATCH"
                mismatched_points.append((x_al, x_cu, x_fe, gm_cpu, gm_gpu, diff))
            
            # Print every 10th point and all mismatches
            if (total_points % 10 == 1) or (status == "MISMATCH"):
                print(f"{x_al:<8.3f} {x_cu:<8.3f} {x_fe:<8.3f} {gm_cpu:<12.1f} {gm_gpu:<12.1f} {diff:<10.3f} {status:<10}")
    
    print("-" * 80)
    
    # Summary statistics
    print(f"\nSummary:")
    print(f"Total valid compositions tested: {total_points}")
    print(f"Matching points (ΔGM < {tolerance} J/mol): {matching_points}")
    print(f"Mismatched points: {len(mismatched_points)}")
    print(f"Success rate: {100*matching_points/total_points:.1f}%")
    
    if mismatched_points:
        print(f"\nMismatched points details:")
        print("-" * 80)
        print(f"{'X(AL)':<8} {'X(CU)':<8} {'X(FE)':<8} {'GM_CPU':<12} {'GM_GPU':<12} {'ΔGM':<10}")
        print("-" * 80)
        for x_al, x_cu, x_fe, gm_cpu, gm_gpu, diff in mismatched_points:
            print(f"{x_al:<8.3f} {x_cu:<8.3f} {x_fe:<8.3f} {gm_cpu:<12.1f} {gm_gpu:<12.1f} {diff:<10.3f}")
    
    # Also check chemical potentials for a few key points
    print("\n" + "=" * 40)
    print("Chemical Potential Comparison (selected points):")
    print("-" * 80)
    
    test_points = [(0.333, 0.333), (0.5, 0.3), (0.7, 0.2)]
    for x_al, x_cu in test_points:
        i = int(round(x_al * 10))
        j = int(round(x_cu * 10))
        idx = j * 11 + i
        x_fe = 1 - x_al - x_cu
        
        print(f"\nX(AL)={x_al:.3f}, X(CU)={x_cu:.3f}, X(FE)={x_fe:.3f}:")
        
        # Compare chemical potentials
        for k, comp in enumerate(['AL', 'CU', 'FE']):
            if k < cpu_result.MU.shape[-1] and k < gpu_result.MU.shape[-1]:
                mu_cpu = cpu_result.MU.values.reshape(-1, cpu_result.MU.shape[-1])[idx, k]
                mu_gpu = gpu_result.MU.values.reshape(-1, gpu_result.MU.shape[-1])[idx, k]
                diff_mu = abs(mu_gpu - mu_cpu)
                print(f"  μ({comp}): CPU={mu_cpu:12.1f}, GPU={mu_gpu:12.1f}, Δ={diff_mu:8.3f}")

print("\nTest completed!")