#!/usr/bin/env python
"""Detailed comparison of CPU vs GPU with exactly 5 phases."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

# Load database
dbf = Database('AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']

# Test with 5 phases
phases = ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'RHOMBOHEDRAL_A7', 'BCC_A2']
print(f"Testing with 5 phases: {phases}")

# Simple test condition - single point
conditions = {
    v.X('BI'): 0.3,
    v.T: 600,
    v.P: 101325
}

print(f"\nTest condition: X(BI)={conditions[v.X('BI')]}, T={conditions[v.T]}K")

try:
    # CPU calculation
    print("\nRunning CPU calculation...")
    result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
    
    # GPU calculation with verbose
    print("\nRunning GPU calculation with verbose=True...")
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)
    
    # Extract results
    cpu_gm = float(result_cpu.GM.values)
    gpu_gm = float(result_gpu.GM.values)
    
    print(f"\n{'='*60}")
    print(f"RESULTS COMPARISON:")
    print(f"{'='*60}")
    
    print(f"\nGibbs energy:")
    print(f"  CPU: {cpu_gm:.6f} J/mol")
    print(f"  GPU: {gpu_gm:.6f} J/mol")
    print(f"  Difference: {abs(cpu_gm - gpu_gm):.6f} J/mol")
    
    # Check phases
    cpu_phases = result_cpu.Phase.values.flatten()
    gpu_phases = result_gpu.Phase.values.flatten()
    cpu_np = result_cpu.NP.values.flatten()
    gpu_np = result_gpu.NP.values.flatten()
    
    print(f"\nPhases and amounts:")
    print(f"  CPU phases:")
    for i, (phase, amount) in enumerate(zip(cpu_phases, cpu_np)):
        if phase and phase != '' and phase != '_FAKE_' and amount > 1e-6:
            print(f"    {phase}: {amount:.4f}")
    
    print(f"  GPU phases:")
    for i, (phase, amount) in enumerate(zip(gpu_phases, gpu_np)):
        if phase and phase != '' and phase != '_FAKE_' and amount > 1e-6:
            print(f"    {phase}: {amount:.4f}")
    
    # Check chemical potentials
    cpu_mu_au = float(result_cpu.MU.sel(component='AU').values)
    cpu_mu_bi = float(result_cpu.MU.sel(component='BI').values)
    gpu_mu_au = float(result_gpu.MU.sel(component='AU').values)
    gpu_mu_bi = float(result_gpu.MU.sel(component='BI').values)
    
    print(f"\nChemical potentials:")
    print(f"  CPU: MU(AU)={cpu_mu_au:.2f}, MU(BI)={cpu_mu_bi:.2f}")
    print(f"  GPU: MU(AU)={gpu_mu_au:.2f}, MU(BI)={gpu_mu_bi:.2f}")
    
except Exception as e:
    print(f"\n✗ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()