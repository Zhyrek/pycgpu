#!/usr/bin/env python
"""Debug initial phase selection."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def main():
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2', 'L12', 'ALCU_THETA', 'AL13FE4', 'AL5FE2']
    
    conditions = {
        v.X('AL'): 0.60,
        v.X('CU'): 0.10,
        v.T: 600,
        v.P: 101325
    }
    
    print("=" * 80)
    print("DEBUGGING INITIAL PHASE SELECTION")
    print("=" * 80)
    
    # Enable debug mode
    os.environ['PYCALPHAD_DEBUG_MODE'] = '1'
    
    print("\n1. Running CPU equilibrium (should show 2 phases in matrix)...")
    print("-" * 80)
    cpu_result = equilibrium(dbf, comps, phases, conditions,
                            gpu=False, verbose=False)
    
    cpu_phases = cpu_result.Phase.values.flatten()
    cpu_np = cpu_result.NP.values.flatten()
    
    print("\nCPU final phases:")
    for i, (phase, amount) in enumerate(zip(cpu_phases, cpu_np)):
        if not np.isnan(amount) and amount > 0.001:
            print(f"  {phase}: {amount:.4f}")
    
    print("\n2. Running GPU equilibrium (should show 3 phases in matrix)...")
    print("-" * 80)
    gpu_result = equilibrium(dbf, comps, phases, conditions,
                            gpu=True, verbose=True)
    
    gpu_phases = gpu_result.Phase.values.flatten()
    gpu_np = gpu_result.NP.values.flatten()
    
    print("\nGPU final phases:")
    for i, (phase, amount) in enumerate(zip(gpu_phases, gpu_np)):
        if not np.isnan(amount) and amount > 0.001:
            print(f"  {phase}: {amount:.4f}")
    
    print("\n" + "=" * 80)
    print("KEY OBSERVATION")
    print("=" * 80)
    print("""
The GPU starts with 3 phases at iteration 0:
- Phase 0: amount=0.345223
- Phase 1: amount=0.128662
- Phase 2: amount=0.526115

But the CPU matrix at iteration 0 only has 2 phases.

This suggests the CPU is either:
1. Consolidating phases before the first iteration
2. Using a different starting point calculation
3. Has a pre-filtering step that removes one phase
""")

if __name__ == "__main__":
    main()