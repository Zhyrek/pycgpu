#!/usr/bin/env python
"""Analyze GPU memory usage for pycalphad equilibrium calculations."""

import numpy as np
import cupy as cp
from pycalphad import Database, variables as v
from pycalphad.core.workspace import Workspace

# Memory analysis
def analyze_memory_requirements():
    """Calculate memory requirements for GPU equilibrium calculations."""
    
    print("GPU Memory Analysis for Pycalphad Equilibrium Calculations")
    print("=" * 60)
    
    # Get GPU info
    device = cp.cuda.Device()
    total_memory = device.mem_info[1] / (1024**3)  # Total memory in GB
    free_memory = device.mem_info[0] / (1024**3)   # Free memory in GB
    
    print(f"\nGPU Device Information:")
    print(f"  Device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"  Total Memory: {total_memory:.2f} GB")
    print(f"  Free Memory: {free_memory:.2f} GB")
    
    # Load database to get actual sizes
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    phases = list(dbf.phases.keys())
    
    # Get dynamic sizes from workspace
    test_conditions = {v.T: 500, v.P: 101325, v.X('BI'): 0.5}
    wks = Workspace(dbf, comps, phases, test_conditions)
    
    # Extract key dimensions
    num_phases = len(phases)
    num_components = len(comps)
    max_phase_dof = max([phase_rec.phase_dof for phase_rec in wks.phase_record_factory.values()])
    max_statevars = len(wks.phase_record_factory.state_variables)
    
    print(f"\nSystem Configuration:")
    print(f"  Number of phases: {num_phases}")
    print(f"  Number of components: {num_components}")
    print(f"  Max phase DOF: {max_phase_dof}")
    print(f"  Number of state variables: {max_statevars}")
    
    # Memory per condition calculation
    print(f"\nPer-Condition Memory Requirements:")
    
    # Major data structures per condition
    arrays_per_condition = {
        # From gpu_equilibrium.py - initial phase data arrays
        "initial_phase_data (phase_indices)": num_phases * 4,  # int32
        "initial_phase_data (phase_amt)": num_phases * 8,  # float64
        "initial_phase_data (phase_dof)": num_phases * max_phase_dof * 8,  # float64
        "initial_phase_data (phase_ids)": num_phases * 4,  # int32
        
        # SystemSpecification (one per condition)
        "SystemSpecification": 2048,  # Estimated struct size with padding
        
        # Results arrays
        "results_array (GM)": 8,  # float64
        "results_array (MU)": num_components * 8,  # float64
        "results_array (NP)": num_phases * 8,  # float64
        "results_array (X)": num_phases * num_components * 8,  # float64
        "results_array (Y)": num_phases * max_phase_dof * 8,  # float64
        "results_array (Phase)": num_phases * 4,  # int32 for phase ID
        
        # Working memory in kernel (per thread)
        "CompositionSet array": num_phases * 512,  # Estimated struct size
        "Solver working memory": 2048,  # Newton solver workspace
    }
    
    total_per_condition = sum(arrays_per_condition.values())
    
    for name, size in arrays_per_condition.items():
        print(f"  {name}: {size} bytes")
    
    print(f"\nTotal per condition: {total_per_condition} bytes ({total_per_condition/1024:.2f} KB)")
    
    # Calculate maximum conditions based on available memory
    print(f"\nMaximum Conditions Calculation:")
    
    # Additional global memory requirements
    global_memory = {
        "PhaseRecord array": num_phases * 4096,  # Estimated PhaseRecord size
        "Grid data": 100000 * (8 + 8 + 8),  # GM, X, Y for grid points
        "Kernel code": 1024 * 1024,  # 1 MB for compiled kernel
        "CuPy overhead": 500 * 1024 * 1024,  # 500 MB CuPy overhead
    }
    
    total_global = sum(global_memory.values())
    print(f"  Global memory requirements: {total_global/1024/1024:.2f} MB")
    
    # Calculate conditions for different memory limits
    memory_limits = [2, 4, 8, 16, 24]  # GB
    
    print(f"\nMaximum conditions for different GPU memory sizes:")
    for limit_gb in memory_limits:
        available_for_conditions = (limit_gb * 1024**3 - total_global) * 0.8  # Use 80% of available
        max_conditions = int(available_for_conditions / total_per_condition)
        print(f"  {limit_gb:2d} GB GPU: ~{max_conditions:,} conditions")
    
    # Actual available on current GPU
    available_for_conditions = (free_memory * 1024**3 - total_global) * 0.8
    max_conditions_current = int(available_for_conditions / total_per_condition)
    
    print(f"\nCurrent GPU can handle approximately {max_conditions_current:,} conditions")
    
    # Memory usage for specific grid sizes
    print(f"\nMemory requirements for specific grid sizes:")
    
    test_cases = [
        (20, 20),    # 400 conditions
        (32, 37),    # 1,184 conditions  
        (50, 50),    # 2,500 conditions
        (100, 75),   # 7,500 conditions
        (100, 150),  # 15,000 conditions
    ]
    
    for n_comp, n_temp in test_cases:
        n_conditions = n_comp * n_temp
        memory_mb = (n_conditions * total_per_condition + total_global) / (1024**2)
        print(f"  {n_comp}x{n_temp} grid ({n_conditions:,} conditions): {memory_mb:.1f} MB")
        
    # Identify memory bottlenecks
    print(f"\nMemory Bottlenecks:")
    print(f"  1. CompositionSet array scales with num_phases: {num_phases * 512 * 1000 / 1024**2:.1f} MB per 1000 conditions")
    print(f"  2. Results arrays scale with phases*components: {(num_phases * num_components * 8 * 2 * 1000) / 1024**2:.1f} MB per 1000 conditions")
    print(f"  3. SystemSpecification array: {2048 * 1000 / 1024**2:.1f} MB per 1000 conditions")
    
    # Thread and block limits
    print(f"\nGPU Thread/Block Limits:")
    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"  Max threads per block: {props['maxThreadsPerBlock']}")
    print(f"  Max blocks per grid: {props['maxGridSize'][0]:,}")
    print(f"  Max threads total: {props['maxThreadsPerBlock'] * min(65535, props['maxGridSize'][0]):,}")
    
    return max_conditions_current

if __name__ == "__main__":
    max_conditions = analyze_memory_requirements()
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("=" * 60)
    print(f"1. For stable operation, limit to {int(max_conditions * 0.7):,} conditions")
    print(f"2. Use batch processing for larger datasets")
    print(f"3. Consider reducing phases or using phase selection")
    print(f"4. Optimize data structures to reduce per-condition memory")