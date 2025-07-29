#!/usr/bin/env python
"""Test if GPU now includes all phases like CPU does."""

import os
os.environ['PYCALPHAD_DEBUG_EQSOLVER'] = ''  # Disable debug output

from pycalphad import Database, equilibrium, variables as v
import numpy as np

# Load AlCuFe database
dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']
phases = ['LIQUID', 'ALCU_ZETA']

print("Testing GPU Phase Inclusion Fix")
print("="*70)
print("The GPU should now include ALL phases, even with zero amount")
print("="*70)

# Test multiple conditions
test_conditions = [
    ("Low T, Al-rich", 0.7, 0.2, 600),
    ("Med T, balanced", 0.6, 0.3, 900),  
    ("High T, Cu-rich", 0.3, 0.6, 1200),
]

for name, x_al, x_cu, T in test_conditions:
    print(f"\n{name}: X(AL)={x_al}, X(CU)={x_cu}, T={T}K")
    print("-" * 50)
    
    conditions = {
        v.T: T, 
        v.P: 101325, 
        v.N: 1, 
        v.X('AL'): x_al,
        v.X('CU'): x_cu
    }
    
    try:
        # CPU calculation
        cpu_result = equilibrium(dbf, comps, phases, conditions, 
                               calc_opts={'pdens': 100}, verbose=False)
        cpu_gm = float(cpu_result.GM.values.item())
        cpu_phases = list(cpu_result.Phase.values.squeeze())
        cpu_np = list(cpu_result.NP.values.squeeze())
        
        # GPU calculation
        gpu_result = equilibrium(dbf, comps, phases, conditions, 
                               calc_opts={'pdens': 100}, verbose=False, gpu=True)
        gpu_gm = float(gpu_result.GM.values.item())
        gpu_phases = list(gpu_result.Phase.values.squeeze())
        gpu_np = list(gpu_result.NP.values.squeeze())
        
        # Compare number of phases
        print(f"Number of phases in result:")
        print(f"  CPU: {len(cpu_phases)} phases")
        print(f"  GPU: {len(gpu_phases)} phases")
        
        if len(cpu_phases) != len(gpu_phases):
            print("  ⚠️  MISMATCH in number of phases!")
        else:
            print("  ✓ Same number of phases")
        
        # Compare phase amounts
        print(f"\nPhase amounts:")
        for i, (cpu_p, gpu_p, cpu_amt, gpu_amt) in enumerate(zip(cpu_phases, gpu_phases, cpu_np, gpu_np)):
            print(f"  Phase {i}: {cpu_p}")
            print(f"    CPU NP: {cpu_amt:.6f}")
            print(f"    GPU NP: {gpu_amt:.6f}")
            print(f"    Diff: {abs(cpu_amt - gpu_amt):.6f}")
        
        # Compare GM
        diff = abs(cpu_gm - gpu_gm)
        print(f"\nGM comparison:")
        print(f"  CPU: {cpu_gm:.6f} J/mol")
        print(f"  GPU: {gpu_gm:.6f} J/mol")
        print(f"  Difference: {diff:.2e} J/mol")
        
        if diff < 10:
            print(f"  ✓ EXCELLENT accuracy!")
        elif diff < 100:
            print(f"  ✓ Good accuracy")
        else:
            print(f"  ⚠️  Large error")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("If the GPU now includes all phases (even with zero amount),")
print("the multi-sublattice accuracy should be greatly improved.")