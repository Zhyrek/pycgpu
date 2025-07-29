#!/usr/bin/env python
"""Test GPU with ALCU_ZETA at various conditions to find where CPU converges."""

from pycalphad import Database, equilibrium, variables as v
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# Load database
dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']

# Test with LIQUID and ALCU_ZETA
phases = ['LIQUID', 'ALCU_ZETA']

print("Testing various conditions to find where both CPU and GPU converge")
print("="*70)

# Try different conditions
test_conditions = [
    # Lower temperatures where phases are more stable
    {'T': 800, 'X_CU': 0.2, 'X_FE': 0.1, 'desc': 'Low T, Al-rich'},
    {'T': 900, 'X_CU': 0.3, 'X_FE': 0.1, 'desc': 'Med T, balanced'},
    {'T': 1000, 'X_CU': 0.4, 'X_FE': 0.1, 'desc': 'Higher T, Cu-rich'},
    # Binary edges (simpler)
    {'T': 850, 'X_CU': 0.3, 'X_FE': 0.0, 'desc': 'Al-Cu binary'},
    {'T': 900, 'X_CU': 0.0, 'X_FE': 0.3, 'desc': 'Al-Fe binary'},
    # Lower total solute
    {'T': 950, 'X_CU': 0.15, 'X_FE': 0.15, 'desc': 'Low solute'},
]

for test in test_conditions:
    T = test['T']
    X_CU = test['X_CU']
    X_FE = test['X_FE']
    X_AL = 1.0 - X_CU - X_FE
    desc = test['desc']
    
    print(f"\nTest: {desc}")
    print(f"Conditions: X(AL)={X_AL:.2f}, X(CU)={X_CU:.2f}, X(FE)={X_FE:.2f}, T={T}K")
    
    conditions = {v.T: T, v.P: 101325, v.N: 1}
    if X_CU > 1e-6:
        conditions[v.X('CU')] = X_CU
    if X_FE > 1e-6:
        conditions[v.X('FE')] = X_FE
    
    # CPU test
    try:
        cpu_result = equilibrium(dbf, comps, phases, conditions, 
                               calc_opts={'pdens': 10}, verbose=False)
        cpu_gm = float(cpu_result.GM.values)
        
        if np.isnan(cpu_gm):
            print(f"  CPU: FAILED (nan)")
            continue
            
        print(f"  CPU: GM={cpu_gm:.2f} J/mol", end='')
        
        # Check phases
        cpu_phases = []
        for idx in range(cpu_result.dims['vertex']):
            np_val = float(cpu_result.NP.values[0,0,0,0,0,idx])
            if np_val > 1e-6:
                phase_name = str(cpu_result.Phase.values[0,0,0,0,0,idx])
                if phase_name and phase_name != '':
                    cpu_phases.append(phase_name)
        print(f" [{', '.join(cpu_phases)}]")
        
    except Exception as e:
        print(f"  CPU: ERROR - {type(e).__name__}")
        continue
    
    # GPU test
    try:
        gpu_result = equilibrium(dbf, comps, phases, conditions, 
                               calc_opts={'pdens': 10}, verbose=False, gpu=True)
        gpu_gm = float(gpu_result.GM.values)
        
        print(f"  GPU: GM={gpu_gm:.2f} J/mol", end='')
        
        # Check phases
        gpu_phases = []
        for idx in range(gpu_result.dims['vertex']):
            np_val = float(gpu_result.NP.values[0,0,0,0,0,idx])
            if np_val > 1e-6:
                phase_name = str(gpu_result.Phase.values[0,0,0,0,0,idx])
                if phase_name and phase_name != '':
                    gpu_phases.append(phase_name)
        print(f" [{', '.join(gpu_phases)}]")
        
        # Compare
        diff = abs(cpu_gm - gpu_gm)
        print(f"  Difference: {diff:.6f} J/mol - {'PASS' if diff < 1.0 else 'FAIL'}")
        
        # If ALCU_ZETA is present and results match, this is a good test case
        if 'ALCU_ZETA' in cpu_phases and diff < 1.0:
            print("  *** Good test case - ALCU_ZETA stable and both converged! ***")
        
    except Exception as e:
        print(f"  GPU: ERROR - {type(e).__name__}")
        
print("\n" + "="*70)