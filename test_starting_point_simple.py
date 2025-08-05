#!/usr/bin/env python
"""Simple test to understand starting point differences."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def test_phase_order_effect():
    """Test if phase order affects starting point."""
    
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    
    conditions = {
        v.X('BI'): 0.1,
        v.T: 400,
        v.P: 101325
    }
    
    print("="*70)
    print("PHASE ORDER EFFECT TEST")
    print("="*70)
    
    # Test 1: Original order
    phases1 = ['AU2BI_C15', 'BCC_A2', 'FCC_A1', 'HCP_A3', 'LIQUID', 'RHOMBOHEDRAL_A7']
    
    print("\nTest 1: Original phase order")
    print(f"Phases: {phases1}")
    
    result1 = equilibrium(dbf, comps, phases1, conditions, gpu=False, verbose=False)
    gm1 = result1.GM.values[0,0,0,0]
    phases_result1 = result1.Phase.values[0,0,0,0]
    np_result1 = result1.NP.values[0,0,0,0]
    
    print(f"GM: {gm1:.6f}")
    print("Active phases:")
    for phase, amount in zip(phases_result1, np_result1):
        if phase != '' and amount > 1e-8:
            print(f"  {phase}: {amount:.6f}")
    
    # Test 2: Different order (HCP_A3 earlier)
    phases2 = ['HCP_A3', 'FCC_A1', 'AU2BI_C15', 'BCC_A2', 'LIQUID', 'RHOMBOHEDRAL_A7']
    
    print("\n\nTest 2: Different phase order (HCP_A3 first)")
    print(f"Phases: {phases2}")
    
    result2 = equilibrium(dbf, comps, phases2, conditions, gpu=False, verbose=False)
    gm2 = result2.GM.values[0,0,0,0]
    phases_result2 = result2.Phase.values[0,0,0,0]
    np_result2 = result2.NP.values[0,0,0,0]
    
    print(f"GM: {gm2:.6f}")
    print("Active phases:")
    for phase, amount in zip(phases_result2, np_result2):
        if phase != '' and amount > 1e-8:
            print(f"  {phase}: {amount:.6f}")
    
    # Compare
    print("\n" + "="*50)
    print("COMPARISON")
    print("="*50)
    
    if abs(gm1 - gm2) < 1e-6:
        print("✅ Same GM values - phase order doesn't affect result")
    else:
        print(f"❌ Different GM values: {gm1:.6f} vs {gm2:.6f}")
        print(f"   Difference: {abs(gm1 - gm2):.2e}")
    
    # Extract active phases
    active1 = set(p for p, a in zip(phases_result1, np_result1) if p != '' and a > 1e-8)
    active2 = set(p for p, a in zip(phases_result2, np_result2) if p != '' and a > 1e-8)
    
    if active1 == active2:
        print("✅ Same active phases")
    else:
        print("❌ Different active phases!")
        print(f"   Order 1: {active1}")
        print(f"   Order 2: {active2}")

if __name__ == "__main__":
    test_phase_order_effect()