#!/usr/bin/env python
"""Test with ALL phases from the Al-Cu-Fe database."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
import time
import glob
warnings.filterwarnings("ignore")

def check_kernel_cache():
    """Check if there are cached kernels."""
    cache_dirs = [
        '.pycalphad_gpu_kernels',
        'pycgpu_kernels',
        '../.pycalphad_gpu_kernels',
        '../pycgpu_kernels'
    ]

    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            files = glob.glob(os.path.join(cache_dir, '*.cu'))
            if files:
                return True, cache_dir, len(files)

    return False, None, 0

def main():
    """Test with all phases."""

    # Check for cached kernels before starting
    has_cache, cache_dir, num_files = check_kernel_cache()
    if has_cache:
        print(f"Found {num_files} cached kernel files in {cache_dir}")
        print("Kernel will be reused if hash matches")
    else:
        print("No cached kernels found - will compile on first run")

    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']

    # Get ALL phases from the database
    all_phases = list(dbf.phases.keys())
    
    print("=" * 80)
    print("TEST WITH ALL PHASES FROM Al-Cu-Fe.tdb")
    print("=" * 80)
    print(f"\nTotal phases in database: {len(all_phases)}")
    print(f"Phases: {all_phases}")
    
    # Test conditions - including the problematic X(CU)=0.5
    test_conditions = [
        (0.2, 0.5, 900, "Problematic condition"),
        (0.3, 0.3, 900, "Equal Al-Cu"),
        (0.1, 0.1, 900, "Low Al-Cu"),
        (0.4, 0.4, 900, "High Al-Cu"),
    ]
    
    print("\n" + "-" * 80)
    print("Testing with ALL phases:")
    print("-" * 80)
    print("\nX(AL) | X(CU) | X(FE) | T(K) | CPU GM    | GPU GM    | Diff    | Compile | Status")
    print("------|-------|-------|------|-----------|-----------|---------|---------|--------")
    
    for x_al, x_cu, temp, desc in test_conditions:
        x_fe = 1.0 - x_al - x_cu
        
        conditions = {
            v.X('AL'): x_al,
            v.X('CU'): x_cu,
            v.T: temp,
            v.P: 101325
        }
        
        try:
            # CPU calculation
            cpu_result = equilibrium(dbf, comps, all_phases, conditions,
                                    calc_opts={'pdens': 50},
                                    gpu=False, verbose=False)
            cpu_gm = cpu_result.GM.values.item()

            # Check cache before GPU run
            cache_before = check_kernel_cache()

            # GPU calculation with timing
            gpu_start = time.time()
            gpu_result = equilibrium(dbf, comps, all_phases, conditions,
                                    calc_opts={'pdens': 50},
                                    gpu=True, verbose=False)
            gpu_time = time.time() - gpu_start
            gpu_gm = gpu_result.GM.values.item()

            # Check cache after GPU run
            cache_after = check_kernel_cache()

            # Determine if compilation happened
            if not cache_before[0] and cache_after[0]:
                compile_status = "NEW"
            elif cache_before[0] and cache_before[2] < cache_after[2]:
                compile_status = "ADD"
            else:
                compile_status = "REUSE"
            
            diff = abs(gpu_gm - cpu_gm)
            status = "✓" if diff < 100 else "✗"

            print(f" {x_al:.2f}  | {x_cu:.2f}  | {x_fe:.2f}  | {temp:4d} | {cpu_gm:9.1f} | {gpu_gm:9.1f} | {diff:7.1f} | {compile_status:7s} | {status}")
            
            # Show which phases are stable
            cpu_np = cpu_result.NP.values.flatten()
            gpu_np = gpu_result.NP.values.flatten()
            
            cpu_stable = []
            gpu_stable = []
            
            for i, phase in enumerate(all_phases):
                if i < len(cpu_np) and cpu_np[i] > 0.001:
                    cpu_stable.append(phase)
                if i < len(gpu_np) and gpu_np[i] > 0.001:
                    gpu_stable.append(phase)
            
            if cpu_stable != gpu_stable or diff > 100:
                print(f"       CPU stable: {', '.join(cpu_stable)}")
                print(f"       GPU stable: {', '.join(gpu_stable)}")
                
        except Exception as e:
            print(f" {x_al:.2f}  | {x_cu:.2f}  | {x_fe:.2f}  | {temp:4d} | ERROR: {str(e)[:42]}")
    
    print("\n" + "=" * 80)
    print("KERNEL TIMING TEST:")
    print("-" * 80)

    # Run the same condition multiple times to test caching
    print("\nTesting kernel reuse with same condition (X(AL)=0.2, X(CU)=0.5, T=900K):")
    print("Run # | GPU Time (s) | Compile Status")
    print("------|--------------|---------------")

    test_cond = {
        v.X('AL'): 0.2,
        v.X('CU'): 0.5,
        v.T: 900,
        v.P: 101325
    }

    for i in range(3):
        cache_before = check_kernel_cache()
        start = time.time()

        gpu_result = equilibrium(dbf, comps, all_phases, test_cond,
                               calc_opts={'pdens': 50},
                               gpu=True, verbose=False)

        elapsed = time.time() - start
        cache_after = check_kernel_cache()

        if not cache_before[0] and cache_after[0]:
            compile_status = "COMPILED"
        elif cache_before[0] and cache_before[2] < cache_after[2]:
            compile_status = "RECOMPILED"
        else:
            compile_status = "REUSED"

        print(f"  {i+1:2d}  |    {elapsed:8.3f}  | {compile_status}")

    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("-" * 80)

    if len(all_phases) > 8:
        print(f"⚠ Database has {len(all_phases)} phases, which exceeds GPU limit of 8 phases")
        print("  GPU may be truncating the phase list or failing")
    else:
        print(f"✓ Database has {len(all_phases)} phases, within GPU limit")

    # Final cache check
    has_cache, cache_dir, num_files = check_kernel_cache()
    if has_cache:
        print(f"\n✓ Kernel cache exists in {cache_dir} with {num_files} files")
        print("  Kernels should be reused on subsequent runs")

    print("=" * 80)

if __name__ == "__main__":
    main()