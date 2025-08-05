#!/usr/bin/env python
"""Debug phase selection and equilibrium for AuBi system with all phases."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

def debug_phase_selection():
    """Debug what phases are being selected in equilibrium calculations."""
    
    # Load database and set up calculation
    dbf = Database('important_tests/AuBi-07Wan.tdb')  
    comps = ['AU', 'BI', 'VA']
    all_phases = filter_phases(dbf, comps)
    
    print(f"All available phases: {all_phases}")
    
    # Test a single condition first
    conditions = {
        v.X('BI'): 0.1,
        v.T: 400,
        v.P: 101325
    }
    
    try:
        print("\n=== Testing CPU calculation ===")
        result_cpu = equilibrium(dbf, comps, all_phases, conditions, gpu=False, verbose=True)
        
        print(f"\nCPU Results:")
        print(f"  GM: {result_cpu.GM.values}")
        print(f"  Phase fractions (NP): {result_cpu.NP.values}")
        print(f"  Phase names: {result_cpu.Phase.values}")
        print(f"  Compositions (X): {result_cpu.X.values}")
        
        # Check which phases have non-zero amounts
        cpu_np = result_cpu.NP.values.flatten()
        cpu_phases = result_cpu.Phase.values.flatten()
        
        print(f"\nActive phases (CPU):")
        for i, (phase, amount) in enumerate(zip(cpu_phases, cpu_np)):
            if amount > 1e-8 and phase != '':
                print(f"  {phase}: {amount:.6f}")
        
    except Exception as e:
        print(f"CPU calculation failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        print("\n=== Testing GPU calculation ===")
        result_gpu = equilibrium(dbf, comps, all_phases, conditions, gpu=True, verbose=True)
        
        print(f"\nGPU Results:")
        print(f"  GM: {result_gpu.GM.values}")
        print(f"  Phase fractions (NP): {result_gpu.NP.values}")
        print(f"  Phase names: {result_gpu.Phase.values}")
        print(f"  Compositions (X): {result_gpu.X.values}")
        
        # Check which phases have non-zero amounts
        gpu_np = result_gpu.NP.values.flatten()
        gpu_phases = result_gpu.Phase.values.flatten()
        
        print(f"\nActive phases (GPU):")
        for i, (phase, amount) in enumerate(zip(gpu_phases, gpu_np)):
            if amount > 1e-8 and phase != '':
                print(f"  {phase}: {amount:.6f}")
        
    except Exception as e:
        print(f"GPU calculation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_phase_selection()