#!/usr/bin/env python
"""Analyze phases at failing conditions."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

dbf = Database('AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
phases = ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'HCP_A3', 'RHOMBOHEDRAL_A7', 'BCC_A2']

# Test the failing conditions specifically
failing_conditions = [
    {'X(BI)': 0.3, 'T': 500, 'desc': 'Condition 10'},
    {'X(BI)': 0.2, 'T': 600, 'desc': 'Condition 17'}
]

# Also test neighboring conditions
neighbor_conditions = [
    {'X(BI)': 0.2, 'T': 500, 'desc': 'Before Cond 10'},
    {'X(BI)': 0.4, 'T': 500, 'desc': 'After Cond 10'},
    {'X(BI)': 0.1, 'T': 600, 'desc': 'Before Cond 17'},
    {'X(BI)': 0.3, 'T': 600, 'desc': 'After Cond 17'}
]

def analyze_condition(cond_dict, calc_type='CPU'):
    result = equilibrium(dbf, comps, phases, 
                        {v.X('BI'): cond_dict['X(BI)'], v.T: cond_dict['T'], v.P: 101325}, 
                        gpu=(calc_type=='GPU'), verbose=False)
    
    print(f"\n{calc_type} - {cond_dict['desc']}: X(BI)={cond_dict['X(BI)']}, T={cond_dict['T']}K")
    print(f"  GM: {result.GM.values[0,0,0,0]:.2f}")
    print(f"  MU(AU): {result.MU.values[0,0,0,0,0]:.2f}")
    print(f"  MU(BI): {result.MU.values[0,0,0,0,1]:.2f}")
    
    print("  Stable phases:")
    # Get the actual phase names from the result
    phase_names = result.Phase.values[0,0,0,0]
    np_values = result.NP.values[0,0,0,0]
    x_values = result.X.values[0,0,0,0]
    
    for i in range(len(phase_names)):
        if phase_names[i] and phase_names[i] != '' and np_values[i] > 1e-6:
            x_bi = x_values[i,1] if x_values.shape[-1] > 1 else 0
            print(f"    {phase_names[i]}: {np_values[i]:.4f} moles, X(BI)={x_bi:.4f}")

print("=== FAILING CONDITIONS ===")
for cond in failing_conditions:
    analyze_condition(cond, 'CPU')
    analyze_condition(cond, 'GPU')

print("\n=== NEIGHBORING CONDITIONS ===")
for cond in neighbor_conditions:
    analyze_condition(cond, 'CPU')