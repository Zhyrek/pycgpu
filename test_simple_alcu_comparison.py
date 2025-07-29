#!/usr/bin/env python
"""Simple test of AlCu GPU vs CPU without debug output."""

import os
os.environ['PYCALPHAD_DEBUG_EQSOLVER'] = ''  # Disable debug output

from pycalphad import Database, equilibrium, variables as v

# Suppress numpy warnings
import warnings
warnings.filterwarnings('ignore')

# Load AlCuFe database
dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']
phases = ['LIQUID', 'ALCU_ZETA']

# Test condition that previously showed 806 J/mol error
conditions = {
    v.T: 900,  
    v.P: 101325, 
    v.N: 1, 
    v.X('AL'): 0.6,
    v.X('CU'): 0.3
}

print("AlCu GPU vs CPU Test")
print("="*50)

# CPU calculation
cpu_result = equilibrium(dbf, comps, phases, conditions, 
                       calc_opts={'pdens': 100}, verbose=False)
cpu_gm = float(cpu_result.GM.values.item())

# GPU calculation  
gpu_result = equilibrium(dbf, comps, phases, conditions, 
                       calc_opts={'pdens': 100}, verbose=False, gpu=True)
gpu_gm = float(gpu_result.GM.values.item())

# Compare
diff = abs(cpu_gm - gpu_gm)

print(f"CPU GM: {cpu_gm:.2f} J/mol")
print(f"GPU GM: {gpu_gm:.2f} J/mol")
print(f"Difference: {diff:.2f} J/mol")

if diff < 10:
    print("\n✓ EXCELLENT! GPU matches CPU within 10 J/mol")
    print(f"  (Previous error was 806 J/mol)")
elif diff < 100:
    print("\n✓ Good accuracy (< 100 J/mol)")
else:
    print(f"\n⚠️  Still have large error")
    
# Check phases
cpu_phases = [(p, amt) for p, amt in zip(cpu_result.Phase.values.squeeze(), 
                                         cpu_result.NP.values.squeeze()) if amt > 0.01]
gpu_phases = [(p, amt) for p, amt in zip(gpu_result.Phase.values.squeeze(), 
                                         gpu_result.NP.values.squeeze()) if amt > 0.01]

print(f"\nPhases with >1% amount:")
print(f"CPU: {cpu_phases}")
print(f"GPU: {gpu_phases}")