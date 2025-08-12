#!/usr/bin/env python
"""Extract CPU equilibrium matrix."""

from pycalphad import Database, equilibrium, variables as v
import numpy as np
import os

def run_cpu_test():
    """Run CPU and extract matrix."""
    
    tdb = Database('Al-Cu-Fe.tdb')
    conditions = {v.T: 1200, v.P: 101325, v.X('CU'): 0.3, v.X('FE'): 0.2}
    
    print("CPU CALCULATION WITH DEBUG:")
    print("-" * 40)
    
    # Set DEBUG_MODE environment variable for CPU
    os.environ['DEBUG_MODE'] = '1'
    
    # Run CPU calculation
    eq_cpu = equilibrium(tdb, ['AL', 'CU', 'FE', 'VA'], ['LIQUID'], conditions, gpu=False, verbose=False)
    cpu_gm = float(eq_cpu.GM.values.flatten()[0])
    
    print(f"\nCPU GM: {cpu_gm:.8f} J/mol")

if __name__ == "__main__":
    run_cpu_test()