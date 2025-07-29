#!/usr/bin/env python
"""Test if phase consolidation fix resolved the multi-sublattice issue."""

import os
os.environ['PYCALPHAD_DEBUG_EQSOLVER'] = ''  # Disable debug output

from pycalphad import Database, equilibrium, variables as v
import numpy as np

# Load AlCuFe database
dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']
phases = ['LIQUID', 'ALCU_ZETA']

print("Testing Phase Consolidation Fix for AlCu System")
print("="*70)

# Test the condition that had the largest error before
conditions = {
    v.T: 900,  # Med T, balanced
    v.P: 101325, 
    v.N: 1, 
    v.X('AL'): 0.6,
    v.X('CU'): 0.3
}

print("Test condition: X(AL)=0.6, X(CU)=0.3, T=900K")
print("Previous error: ~806 J/mol")
print("="*70)

try:
    # CPU calculation
    cpu_result = equilibrium(dbf, comps, phases, conditions, 
                           calc_opts={'pdens': 100}, verbose=False)
    cpu_gm = float(cpu_result.GM.values.item())
    cpu_phases = list(cpu_result.Phase.values.squeeze())
    cpu_np = list(cpu_result.NP.values.squeeze())
    
    print(f"\nCPU Result:")
    print(f"  GM: {cpu_gm:.6f} J/mol")
    print(f"  Phases present: {[p for p, amt in zip(cpu_phases, cpu_np) if amt > 0.01]}")
    print(f"  Phase amounts: {[f'{amt:.4f}' for p, amt in zip(cpu_phases, cpu_np) if amt > 0.01]}")
    
    # GPU calculation
    gpu_result = equilibrium(dbf, comps, phases, conditions, 
                           calc_opts={'pdens': 100}, verbose=False, gpu=True)
    gpu_gm = float(gpu_result.GM.values.item())
    gpu_phases = list(gpu_result.Phase.values.squeeze())
    gpu_np = list(gpu_result.NP.values.squeeze())
    
    print(f"\nGPU Result:")
    print(f"  GM: {gpu_gm:.6f} J/mol")
    print(f"  Phases present: {[p for p, amt in zip(gpu_phases, gpu_np) if amt > 0.01]}")
    print(f"  Phase amounts: {[f'{amt:.4f}' for p, amt in zip(gpu_phases, gpu_np) if amt > 0.01]}")
    
    # Compare
    diff = abs(cpu_gm - gpu_gm)
    print(f"\nDifference: {diff:.2e} J/mol")
    
    # Check phase amounts
    print(f"\nPhase amount comparison:")
    for i, (phase, cpu_amt, gpu_amt) in enumerate(zip(cpu_phases, cpu_np, gpu_np)):
        if cpu_amt > 0.01 or gpu_amt > 0.01:
            print(f"  {phase}: CPU={cpu_amt:.4f}, GPU={gpu_amt:.4f}, diff={abs(cpu_amt-gpu_amt):.4f}")
    
    # Success criteria
    if diff < 10:
        print(f"\n✓ EXCELLENT! Error reduced from ~806 J/mol to {diff:.2e} J/mol")
        print(f"  Improvement factor: {806/diff:.1f}x")
    elif diff < 100:
        print(f"\n✓ Good! Error reduced from ~806 J/mol to {diff:.2e} J/mol")
        print(f"  Improvement factor: {806/diff:.1f}x")
    else:
        print(f"\n⚠️  Error still large: {diff:.2e} J/mol")
        print(f"  (Previous error was ~806 J/mol)")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()