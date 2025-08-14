#!/usr/bin/env python
"""Debug CPU constraint matrix."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['PYCALPHAD_DEBUG_MODE'] = '1'

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.minimizer import set_debug_mode
import warnings
warnings.filterwarnings("ignore")

# Enable debug mode
set_debug_mode(True)

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
    print("CPU CONSTRAINT MATRIX DEBUG")
    print("=" * 80)
    
    cpu_result = equilibrium(dbf, comps, phases, conditions, gpu=False)
    
    print("\nCPU Result:")
    cpu_phases = cpu_result.Phase.values.flatten()
    cpu_np = cpu_result.NP.values.flatten()
    
    for i, (phase, amount) in enumerate(zip(cpu_phases, cpu_np)):
        if not np.isnan(amount) and amount > 0.001:
            print(f"  {phase}: {amount:.4f}")

if __name__ == "__main__":
    main()