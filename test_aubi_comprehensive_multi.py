#!/usr/bin/env python
"""
Comprehensive multi-condition test for Au-Bi system
Tests GPU vs CPU with multiple conditions in a single call
Starting with just LIQUID and RHOMBOHEDRAL_A7 phases
"""

from pycalphad import Database, equilibrium
import numpy as np
import time
import os

# Clear GPU cache
os.system('rm -f ~/.cupy/kernel_cache/*')

# Load database
db = Database('/mnt/c/users/scott/Documents/pycalphad/AuBi-07Wan.tdb')
components = ['AU', 'BI', 'VA']
phases = ['LIQUID', 'FCC_A1']  # Test with FCC_A1 which has vacancy

# Define test grid
x_bi_values = np.linspace(0.1, 0.9, 8)  # 8 compositions
temperatures = np.array([400, 500, 600, 700, 800])  # 5 temperatures

print(f"Testing {len(x_bi_values)} compositions x {len(temperatures)} temperatures = {len(x_bi_values) * len(temperatures)} total conditions")
print(f"Phases: {phases}")

# Build conditions for multi-point calculation
# Create conditions dict with arrays for X(BI) and T
conditions = {
    'P': 101325,
    'T': temperatures,
    'X(BI)': x_bi_values
}

print("Running calculations with multiple conditions per call...")

# CPU calculation
print("Running CPU calculation...")
start_cpu = time.time()
result_cpu = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 2000})
cpu_time = time.time() - start_cpu
print(f"CPU calculation completed in {cpu_time:.1f} seconds")

# GPU calculation
print("Running GPU calculation...")
start_gpu = time.time()
result_gpu = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 2000}, gpu=True)
gpu_time = time.time() - start_gpu
print(f"GPU calculation completed in {gpu_time:.1f} seconds")

# Extract results
print("\nDebug - Result shapes:")
print(f"  cpu_gm.shape: {result_cpu.GM.values.shape}")
print(f"  gpu_gm.shape: {result_gpu.GM.values.shape}")
print(f"  result_cpu.X shape: {result_cpu.X.values.shape}")
print(f"  result_cpu.T shape: {result_cpu.T.values.shape}")
print(f"  result_cpu dimensions: {result_cpu.dims}")

# Extract values for comparison
cpu_gm = result_cpu.GM.values
gpu_gm = result_gpu.GM.values
cpu_mu = result_cpu.MU.values
gpu_mu = result_gpu.MU.values
cpu_phase = result_cpu.Phase.values
gpu_phase = result_gpu.Phase.values
cpu_np = result_cpu.NP.values
gpu_np = result_gpu.NP.values

# Extract condition values
print("\nExtracting condition values from dataset...")
print(f"  Dimensions in order: {list(result_cpu.dims.keys())}")
print(f"  Temperature coordinates: {result_cpu.coords['T'].values}")
print(f"  X_BI coordinates: {result_cpu.coords['X_BI'].values}")

# Flatten arrays for comparison
cpu_gm_flat = cpu_gm.flatten()
gpu_gm_flat = gpu_gm.flatten()

# Get temperature and composition for each flattened point
temp_coords = result_cpu.coords['T'].values
x_bi_coords = result_cpu.coords['X_BI'].values

# Create pairs of (X_BI, T) for each condition
condition_pairs = []
for t in temp_coords:
    for x in x_bi_coords:
        condition_pairs.append((x, t))

print(f"  Constructed {len(condition_pairs)} condition pairs")
print(f"  cpu_gm flattened shape: {cpu_gm_flat.shape}")

# Ensure we have the right number of results
num_conditions = len(condition_pairs)
print(f"\nFlattened array lengths:")
print(f"  cpu_gm_flat: {len(cpu_gm_flat)}")
print(f"  condition_pairs: {len(condition_pairs)}")

# Save detailed results
with open('aubi_liquid_fcc_comparison_results_multi.txt', 'w') as f:
    f.write("X(BI)\tT(K)\tCPU_GM\tGPU_GM\tGM_DIFF\tCPU_Phases\tGPU_Phases\tPhases_Match\tSTATUS\n")
    
    passed = 0
    failed = 0
    
    for idx, (x_bi, temp) in enumerate(condition_pairs):
        if idx >= len(cpu_gm_flat) or idx >= len(gpu_gm_flat):
            print(f"Warning: Index {idx} out of bounds for result arrays")
            continue
            
        cpu_gm_val = cpu_gm_flat[idx]
        gpu_gm_val = gpu_gm_flat[idx]
        gm_diff = abs(cpu_gm_val - gpu_gm_val)
        
        # Extract phases present at this condition
        # Need to map the flattened index back to multi-dimensional indices
        # The shape is typically (N, P, T, X_BI) for scalars
        # For Phase/NP it's (N, P, T, X_BI, vertex)
        
        # Find indices in the original arrays
        t_idx = list(temp_coords).index(temp)
        x_idx = list(x_bi_coords).index(x_bi)
        
        # Extract phases (assuming shape is compatible)
        try:
            cpu_phases_at_point = []
            gpu_phases_at_point = []
            
            # Get phase data for this condition
            for v_idx in range(cpu_phase.shape[-1]):  # vertex dimension
                cpu_phase_name = cpu_phase[0, 0, t_idx, x_idx, v_idx]
                gpu_phase_name = gpu_phase[0, 0, t_idx, x_idx, v_idx]
                cpu_amount = cpu_np[0, 0, t_idx, x_idx, v_idx]
                gpu_amount = gpu_np[0, 0, t_idx, x_idx, v_idx]
                
                if cpu_phase_name and cpu_phase_name != '' and cpu_amount > 1e-10:
                    cpu_phases_at_point.append(f"{cpu_phase_name}({cpu_amount:.3f})")
                if gpu_phase_name and gpu_phase_name != '' and gpu_amount > 1e-10:
                    gpu_phases_at_point.append(f"{gpu_phase_name}({gpu_amount:.3f})")
            
            cpu_phases_str = ','.join(cpu_phases_at_point) if cpu_phases_at_point else 'None'
            gpu_phases_str = ','.join(gpu_phases_at_point) if gpu_phases_at_point else 'None'
            phases_match = cpu_phases_str == gpu_phases_str
            
        except Exception as e:
            cpu_phases_str = "Error"
            gpu_phases_str = "Error"
            phases_match = False
        
        # Tolerance for GM comparison
        gm_tol = 1e-6
        
        # Status
        if gm_diff < gm_tol and phases_match:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1
        
        f.write(f"{x_bi:.1f}\t{temp:.0f}\t{cpu_gm_val:.12f}\t{gpu_gm_val:.12f}\t"
                f"{gm_diff:.12f}\t{cpu_phases_str}\t{gpu_phases_str}\t"
                f"{phases_match}\t{status}\n")
    
    # Write summary
    f.write(f"\n# SUMMARY\n")
    f.write(f"# Total conditions tested: {num_conditions}\n")
    f.write(f"# Passed: {passed}\n")
    f.write(f"# Failed: {failed}\n")
    f.write(f"# Pass rate: {100 * passed / num_conditions:.1f}%\n")
    f.write(f"# CPU time: {cpu_time:.1f} seconds\n")
    f.write(f"# GPU time: {gpu_time:.1f} seconds\n")
    f.write(f"# Speedup: {cpu_time/gpu_time:.1f}x\n")

print(f"\nReceived {len(cpu_gm_flat)} results (expected {num_conditions})")
print(f"\nTest completed in {time.time() - start_cpu:.1f} seconds")
print(f"Results saved to aubi_liquid_fcc_comparison_results_multi.txt")

print(f"\nSummary:")
print(f"  Total conditions: {num_conditions}")
print(f"  Passed: {passed}")
print(f"  Failed: {failed}")
print(f"  Pass rate: {100 * passed / num_conditions:.1f}%")

print(f"\nPerformance:")
print(f"  CPU time: {cpu_time:.1f} seconds")
print(f"  GPU time: {gpu_time:.1f} seconds")
print(f"  Speedup: {cpu_time/gpu_time:.1f}x")

# Show failed conditions if any
if failed > 0:
    print(f"\nFailed conditions:")
    fail_count = 0
    for idx, (x_bi, temp) in enumerate(condition_pairs):
        if idx >= len(cpu_gm_flat) or idx >= len(gpu_gm_flat):
            continue
        cpu_gm_val = cpu_gm_flat[idx]
        gpu_gm_val = gpu_gm_flat[idx]
        gm_diff = abs(cpu_gm_val - gpu_gm_val)
        if gm_diff >= gm_tol or not phases_match:
            print(f"  X(BI)={x_bi:.1f}, T={temp:.0f}K")
            fail_count += 1
            if fail_count >= 5:
                print(f"  ... and {failed - 5} more")
                break