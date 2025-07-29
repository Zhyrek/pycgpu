#!/usr/bin/env python
"""Test GPU vs CPU equilibrium calculations for the Al-Cu-Fe system."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import warnings
import time
import itertools

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Load database and set up calculation
dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']
phases = filter_phases(dbf, comps)

print(f"Available phases in Al-Cu-Fe system: {phases}")
print()

# Define test conditions - ternary compositions
# We'll test along several composition paths
test_conditions = []

# Test 1: Along Al-Cu binary (Fe=0)
for x_cu in np.arange(0.1, 1.0, 0.2):
    test_conditions.append({
        'X_CU': x_cu,
        'X_FE': 0.0,
        'X_AL': 1.0 - x_cu,
        'T': 800
    })

# Test 2: Along Al-Fe binary (Cu=0)
for x_fe in np.arange(0.1, 1.0, 0.2):
    test_conditions.append({
        'X_CU': 0.0,
        'X_FE': x_fe,
        'X_AL': 1.0 - x_fe,
        'T': 800
    })

# Test 3: Along Cu-Fe binary (Al=0)
for x_fe in np.arange(0.1, 1.0, 0.2):
    test_conditions.append({
        'X_CU': 1.0 - x_fe,
        'X_FE': x_fe,
        'X_AL': 0.0,
        'T': 1200  # Higher temperature for Cu-Fe
    })

# Test 4: Some ternary compositions
ternary_points = [
    (0.33, 0.33, 0.34, 900),  # Near center
    (0.5, 0.3, 0.2, 850),
    (0.2, 0.5, 0.3, 850),
    (0.3, 0.2, 0.5, 850),
    (0.6, 0.2, 0.2, 800),
    (0.2, 0.6, 0.2, 800),
    (0.2, 0.2, 0.6, 800),
]

for x_al, x_cu, x_fe, temp in ternary_points:
    test_conditions.append({
        'X_AL': x_al,
        'X_CU': x_cu,
        'X_FE': x_fe,
        'T': temp
    })

print(f"Testing {len(test_conditions)} conditions in the Al-Cu-Fe system")
print("="*80)

# Test different temperatures for selected compositions
temperatures = [600, 800, 1000, 1200]
pressure = 101325

# Track results
results = []
passed = 0
failed = 0

# Open output file
with open('alcufe_gpu_cpu_comparison.txt', 'w') as f:
    # Write header
    f.write("X_AL\tX_CU\tX_FE\tT(K)\tCPU_GM\tGPU_GM\tGM_DIFF\tCPU_PHASES\tGPU_PHASES\tSTATUS\n")
    
    # Run tests
    start_time = time.time()
    
    for i, cond in enumerate(test_conditions):
        x_al = cond['X_AL']
        x_cu = cond['X_CU']
        x_fe = cond['X_FE']
        temp = cond['T']
        
        print(f"\nTest {i+1}/{len(test_conditions)}: X(AL)={x_al:.2f}, X(CU)={x_cu:.2f}, X(FE)={x_fe:.2f}, T={temp}K")
        
        try:
            # Set up conditions for equilibrium calculation
            # Need to specify only n-1 mole fractions for n components
            conditions = {v.T: temp, v.P: pressure, v.N: 1}
            
            # Add composition constraints - only need 2 for a ternary system
            if x_cu < 0.999:  # Avoid X=1 which can cause issues
                conditions[v.X('CU')] = x_cu
            if x_fe < 0.999 and x_cu + x_fe < 0.999:
                conditions[v.X('FE')] = x_fe
            # AL is calculated by difference
            
            # CPU calculation
            print("  Running CPU calculation...")
            cpu_start = time.time()
            cpu_result = equilibrium(dbf, comps, phases, conditions, 
                                   calc_opts={'pdens': 50}, 
                                   verbose=False, 
                                   grid=True)
            cpu_time = time.time() - cpu_start
            
            # GPU calculation
            print("  Running GPU calculation...")
            gpu_start = time.time()
            gpu_result = equilibrium(dbf, comps, phases, conditions, 
                                   calc_opts={'pdens': 50}, 
                                   verbose=False, 
                                   grid=True, 
                                   gpu=True)
            gpu_time = time.time() - gpu_start
            
            # Extract results
            cpu_gm = float(cpu_result.GM.values)
            gpu_gm = float(gpu_result.GM.values)
            gm_diff = abs(gpu_gm - cpu_gm)
            
            # Get stable phases
            cpu_phases = []
            gpu_phases = []
            
            # Extract phase information
            for phase_idx in range(cpu_result.dims['vertex']):
                cpu_np = float(cpu_result.NP.values[0,0,0,0,phase_idx])
                if cpu_np > 1e-6:
                    phase_name = str(cpu_result.Phase.values[0,0,0,0,phase_idx])
                    if phase_name and phase_name != '':
                        cpu_phases.append(f"{phase_name}({cpu_np:.3f})")
            
            for phase_idx in range(gpu_result.dims['vertex']):
                gpu_np = float(gpu_result.NP.values[0,0,0,0,phase_idx])
                if gpu_np > 1e-6:
                    phase_name = str(gpu_result.Phase.values[0,0,0,0,phase_idx])
                    if phase_name and phase_name != '':
                        gpu_phases.append(f"{phase_name}({gpu_np:.3f})")
            
            # Determine pass/fail
            gm_tolerance = 1.0  # J/mol
            status = "PASS" if gm_diff < gm_tolerance else "FAIL"
            
            if status == "PASS":
                passed += 1
            else:
                failed += 1
            
            # Write results
            cpu_phases_str = "+".join(cpu_phases) if cpu_phases else "NONE"
            gpu_phases_str = "+".join(gpu_phases) if gpu_phases else "NONE"
            
            f.write(f"{x_al:.3f}\t{x_cu:.3f}\t{x_fe:.3f}\t{temp}\t")
            f.write(f"{cpu_gm:.6f}\t{gpu_gm:.6f}\t{gm_diff:.6f}\t")
            f.write(f"{cpu_phases_str}\t{gpu_phases_str}\t{status}\n")
            f.flush()
            
            print(f"  CPU: GM={cpu_gm:.2f} J/mol, phases={cpu_phases_str} (time={cpu_time:.2f}s)")
            print(f"  GPU: GM={gpu_gm:.2f} J/mol, phases={gpu_phases_str} (time={gpu_time:.2f}s)")
            print(f"  Difference: {gm_diff:.6f} J/mol - {status}")
            
            if status == "FAIL":
                print(f"  WARNING: Large difference detected!")
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            f.write(f"{x_al:.3f}\t{x_cu:.3f}\t{x_fe:.3f}\t{temp}\t")
            f.write(f"ERROR\tERROR\tERROR\tERROR\tERROR\tERROR\n")
            f.flush()
            failed += 1
    
    # Write summary
    total_time = time.time() - start_time
    f.write(f"\n# SUMMARY\n")
    f.write(f"# Total conditions tested: {len(test_conditions)}\n")
    f.write(f"# Passed: {passed}\n")
    f.write(f"# Failed: {failed}\n")
    f.write(f"# Pass rate: {100*passed/len(test_conditions):.1f}%\n")
    f.write(f"# Total time: {total_time:.1f} seconds\n")

print(f"\nTest completed in {total_time:.1f} seconds")
print(f"Results saved to alcufe_gpu_cpu_comparison.txt")
print(f"\nSummary:")
print(f"  Total conditions: {len(test_conditions)}")
print(f"  Passed: {passed}")
print(f"  Failed: {failed}")
print(f"  Pass rate: {100*passed/len(test_conditions):.1f}%")