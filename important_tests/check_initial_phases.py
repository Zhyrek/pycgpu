#!/usr/bin/env python
"""Check what phases are initially passed to CPU vs GPU."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def check_initial_setup():
    """Check the initial phases for the problematic condition."""
    
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2', 'L12', 'ALCU_THETA', 'AL13FE4', 'AL5FE2']
    
    print("=" * 80)
    print("CHECKING INITIAL PHASE SETUP")
    print("=" * 80)
    
    conditions = {
        v.X('AL'): 0.60,
        v.X('CU'): 0.10,
        v.T: 600,
        v.P: 101325
    }
    
    # Enable debug mode to see what's happening
    os.environ['PYCALPHAD_DEBUG_MODE'] = '1'
    
    print("\n1. CPU Calculation (to see what phases it starts with):")
    print("-" * 80)
    cpu_result = equilibrium(dbf, comps, phases, conditions,
                            gpu=False, verbose=False)
    
    print("\nCPU Final Result:")
    cpu_phases = cpu_result.Phase.values.flatten()
    cpu_np = cpu_result.NP.values.flatten()
    for i, (phase, amount) in enumerate(zip(cpu_phases, cpu_np)):
        if not np.isnan(amount) and amount > 0.001:
            print(f"  Phase {i}: {phase} = {amount:.4f}")
    
    print("\n2. GPU Calculation (with verbose to see initial setup):")
    print("-" * 80)
    
    # Run GPU with verbose to see what it's doing
    gpu_result = equilibrium(dbf, comps, phases, conditions,
                            gpu=True, verbose=True)
    
    print("\nGPU Final Result:")
    gpu_phases = gpu_result.Phase.values.flatten()
    gpu_np = gpu_result.NP.values.flatten()
    for i, (phase, amount) in enumerate(zip(gpu_phases, gpu_np)):
        if not np.isnan(amount) and amount > 0.001:
            print(f"  Phase {i}: {phase} = {amount:.4f}")
    
    print("\n" + "=" * 80)
    print("KEY QUESTION: Why does GPU have 3 phases in its matrix at iteration 0?")
    print("=" * 80)
    print("""
The GPU equilibrium matrix at iteration 0 has 7 rows:
- 3 phase energy equations (rows 0-2) 
- 2 mole fraction constraints (rows 3-4)
- 1 system amount constraint (row 5)

While CPU has 6 rows:
- 2 phase energy equations (rows 0-1)
- 2 mole fraction constraints (rows 2-3) 
- 1 system amount constraint (row 4)

This means GPU is starting with 3 active phases while CPU starts with 2.
The initial phase selection is different!
""")

if __name__ == "__main__":
    check_initial_setup()