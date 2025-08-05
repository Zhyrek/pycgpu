#!/usr/bin/env python
"""Test to see GPU initial phase setup."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def test_initial_phases():
    """Check GPU initial phases."""
    
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    phases = ['AU2BI_C15', 'BCC_A2', 'FCC_A1', 'HCP_A3', 'LIQUID', 'RHOMBOHEDRAL_A7']
    
    conditions = {
        v.X('BI'): 0.1,
        v.T: 400,
        v.P: 101325
    }
    
    print("="*70)
    print("GPU INITIAL PHASE SETUP TEST")
    print("="*70)
    
    # GPU calculation with verbose
    print("\n--- GPU VERBOSE OUTPUT ---")
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)
    
    gpu_gm = result_gpu.GM.values[0,0,0,0]
    gpu_phases = result_gpu.Phase.values[0,0,0,0]
    gpu_np = result_gpu.NP.values[0,0,0,0]
    
    print(f"\nGPU Final result:")
    print(f"  GM: {gpu_gm:.6f}")
    print("  Active phases:")
    for phase, amount in zip(gpu_phases, gpu_np):
        if phase != '' and amount > 1e-8:
            print(f"    {phase}: {amount:.6f}")

if __name__ == "__main__":
    test_initial_phases()