#!/usr/bin/env python
"""Basic test of GPU vs CPU for Al-Cu-Fe system."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import warnings

warnings.filterwarnings('ignore')

# Load database
dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']
phases = filter_phases(dbf, comps)

print(f'Testing Al-Cu-Fe system')
print(f'Available phases: {len(phases)} phases')
print()

# Test conditions - need to specify n-1 mole fractions for n components
test_cases = [
    {'desc': 'Al-Cu binary (X_CU=0.3)', 'conds': {v.T: 800, v.P: 101325, v.N: 1, v.X('CU'): 0.3, v.X('FE'): 0.0}},
    {'desc': 'Al-Fe binary (X_FE=0.3)', 'conds': {v.T: 800, v.P: 101325, v.N: 1, v.X('CU'): 0.0, v.X('FE'): 0.3}},
    {'desc': 'Ternary (X_CU=0.3, X_FE=0.3)', 'conds': {v.T: 900, v.P: 101325, v.N: 1, v.X('CU'): 0.3, v.X('FE'): 0.3}},
]

passed = 0
failed = 0

for test in test_cases:
    print(f"Testing {test['desc']}...")
    conditions = test['conds']
    
    try:
        # CPU calculation
        cpu_result = equilibrium(dbf, comps, phases, conditions, calc_opts={'pdens': 20}, verbose=False)
        cpu_gm = float(cpu_result.GM.values)
        
        # Count CPU phases
        cpu_phases = []
        for idx in range(cpu_result.dims['vertex']):
            np_val = float(cpu_result.NP.values[0,0,0,0,idx])
            if np_val > 1e-6:
                phase_name = str(cpu_result.Phase.values[0,0,0,0,idx])
                if phase_name and phase_name != '':
                    cpu_phases.append(phase_name)
        
        # GPU calculation
        gpu_result = equilibrium(dbf, comps, phases, conditions, calc_opts={'pdens': 20}, verbose=False, gpu=True)
        gpu_gm = float(gpu_result.GM.values)
        
        # Count GPU phases
        gpu_phases = []
        for idx in range(gpu_result.dims['vertex']):
            np_val = float(gpu_result.NP.values[0,0,0,0,idx])
            if np_val > 1e-6:
                phase_name = str(gpu_result.Phase.values[0,0,0,0,idx])
                if phase_name and phase_name != '':
                    gpu_phases.append(phase_name)
        
        diff = abs(gpu_gm - cpu_gm)
        status = 'PASS' if diff < 1.0 else 'FAIL'
        
        if status == 'PASS':
            passed += 1
        else:
            failed += 1
            
        print(f'  CPU: GM={cpu_gm:.2f} J/mol, {len(cpu_phases)} phases: {cpu_phases}')
        print(f'  GPU: GM={gpu_gm:.2f} J/mol, {len(gpu_phases)} phases: {gpu_phases}')
        print(f'  Difference: {diff:.6f} J/mol - {status}')
        print()
        
    except Exception as e:
        print(f'  ERROR: {str(e)}')
        failed += 1
        print()

print("="*60)
print("SUMMARY:")
print(f"  Total tests: {len(test_cases)}")
print(f"  Passed: {passed}")
print(f"  Failed: {failed}")
print(f"  Pass rate: {100*passed/len(test_cases):.1f}%")