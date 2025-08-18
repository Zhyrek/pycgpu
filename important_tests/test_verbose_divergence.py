#!/usr/bin/env python
"""Test with verbose output to find exact divergence point."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def test_verbose_divergence():
    """Run with verbose to find divergence."""
    
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    
    # All 21 phases
    phases = ['AL13FE4', 'AL2FE', 'AL5FE2', 'AL5FE4', 'ALCU_DELTA', 'ALCU_EPSILON', 
              'ALCU_ETA', 'ALCU_PRIME', 'ALCU_THETA', 'ALCU_ZETA', 'BCC_A2', 'BCC_B2', 
              'FCC_A1', 'GAMMA_D83', 'GAMMA_H', 'L12', 'LIQUID', 'TS01T1', 'TS01T2', 
              'TS01T3', 'TS01TI']
    
    conditions = {
        v.X('AL'): 0.40,
        v.X('CU'): 0.25,
        v.T: 700,
        v.P: 101325
    }
    
    print("=" * 80)
    print("Running with verbose=True to capture full debug output")
    print("=" * 80)
    print(f"\nCondition: X(AL)=0.40, X(CU)=0.25, X(FE)=0.35, T=700K")
    
    # Save outputs to files for easier analysis
    print("\n--- CPU Calculation (verbose) ---")
    cpu_result = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=True)
    
    print("\n" + "=" * 80)
    print("--- GPU Calculation (verbose) ---")
    gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)

if __name__ == "__main__":
    test_verbose_divergence()