#!/usr/bin/env python
"""Print the equilibrium matrix to debug constraint enforcement."""

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
    phases = ['BCC_B2', 'AL5FE2']
    
    conditions = {
        v.X('AL'): 0.60,
        v.X('CU'): 0.10,
        v.T: 600,
        v.P: 101325
    }
    
    print("=" * 80)
    print("CONSTRAINT MATRIX DEBUG")
    print("=" * 80)
    
    print("\nRunning with verbose=True to see equilibrium matrix...")
    
    # Set environment variable to enable CPU debug output
    os.environ['PYCALPHAD_DEBUG_MODE'] = '1'
    
    # Run CPU first
    print("\n" + "=" * 80)
    print("CPU EQUILIBRIUM MATRIX")
    print("=" * 80)
    
    cpu_result = equilibrium(dbf, comps, phases, conditions,
                            gpu=False, verbose=False)
    
    # Check CPU result
    cpu_phases = cpu_result.Phase.values.flatten()
    cpu_np = cpu_result.NP.values.flatten()
    
    print("\nCPU Result:")
    for i, (phase, amount) in enumerate(zip(cpu_phases, cpu_np)):
        if not np.isnan(amount) and amount > 0.001:
            print(f"  {phase}: {amount:.4f}")
    
    print("\n" + "=" * 80)
    print("GPU EQUILIBRIUM MATRIX")
    print("=" * 80)
    
    # Run GPU with verbose to see matrix
    gpu_result = equilibrium(dbf, comps, phases, conditions,
                            gpu=True, verbose=True)
    
    # Check GPU result
    gpu_phases = gpu_result.Phase.values.flatten()
    gpu_np = gpu_result.NP.values.flatten()
    
    print("\nGPU Result:")
    for i, (phase, amount) in enumerate(zip(gpu_phases, gpu_np)):
        if not np.isnan(amount) and amount > 0.001:
            print(f"  {phase}: {amount:.4f}")

if __name__ == "__main__":
    main()