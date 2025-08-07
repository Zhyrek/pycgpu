#!/usr/bin/env python
"""Test Au-Bi system with all 6 phases after fixing site_fractions_offset."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.gpu.gpu_codegen import compute_dynamic_kernel_sizes
from pycalphad import Workspace

# Load database
dbf = Database('AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']

# Full Au-Bi system with all phases
phases = ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'AU2BI_HEX', 'RHOMBOHEDRAL_A7', 'BCC_A2']
print(f"Testing full Au-Bi system with {len(phases)} phases: {phases}")

# Check what MAX_PHASES will be with the fixed code
wks = Workspace(dbf, comps, phases, {v.X('BI'): 0.3, v.T: 600, v.P: 101325})
sizes = compute_dynamic_kernel_sizes(wks)
print(f"\nComputed kernel sizes:")
print(f"  MAX_PHASES = {sizes['MAX_PHASES']} (for {len(phases)} actual phases)")
print(f"  MAX_COMPONENTS = {sizes['MAX_COMPONENTS']}")
print(f"  MAX_DOF_PER_PHASE = {sizes['MAX_DOF_PER_PHASE']}")

# Test with varying number of conditions
for num_conds in [1, 5, 10, 15]:
    print(f"\n{'='*60}")
    print(f"Testing with {num_conds} conditions")
    
    if num_conds == 1:
        conditions = {
            v.X('BI'): 0.3,
            v.T: 600,
            v.P: 101325
        }
    else:
        # Create multiple conditions
        x_vals = np.linspace(0.1, 0.9, num_conds)
        conditions = {
            v.X('BI'): x_vals,
            v.T: 600,
            v.P: 101325
        }
    
    try:
        # CPU calculation
        print(f"  Running CPU calculation...")
        result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
        print(f"  ✓ CPU calculation completed")
        
        # GPU calculation
        print(f"  Running GPU calculation...")
        result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
        print(f"  ✓ GPU calculation completed")
        
        # Compare results
        cpu_gm = result_cpu.GM.values.flatten()
        gpu_gm = result_gpu.GM.values.flatten()
        
        max_diff = np.max(np.abs(cpu_gm - gpu_gm))
        avg_diff = np.mean(np.abs(cpu_gm - gpu_gm))
        
        print(f"\n  Results comparison:")
        print(f"    Maximum GM difference: {max_diff:.6f} J/mol")
        print(f"    Average GM difference: {avg_diff:.6f} J/mol")
        
        if max_diff < 1.0:
            print(f"  ✓ TEST PASSED with {num_conds} conditions")
        else:
            print(f"  ✗ TEST FAILED with {num_conds} conditions - diff = {max_diff:.1f} J/mol")
            
            # Show details for single condition failure
            if num_conds == 1:
                cpu_phases = result_cpu.Phase.values.flatten()
                gpu_phases = result_gpu.Phase.values.flatten()
                cpu_np = result_cpu.NP.values.flatten()
                gpu_np = result_gpu.NP.values.flatten()
                
                print(f"\n  CPU phases:")
                for i, (phase, amount) in enumerate(zip(cpu_phases, cpu_np)):
                    if phase and phase != '' and phase != '_FAKE_' and amount > 1e-6:
                        print(f"    {phase}: {amount:.4f}")
                
                print(f"\n  GPU phases:")
                for i, (phase, amount) in enumerate(zip(gpu_phases, gpu_np)):
                    if phase and phase != '' and phase != '_FAKE_' and amount > 1e-6:
                        print(f"    {phase}: {amount:.4f}")
            
    except Exception as e:
        print(f"  ✗ ERROR with {num_conds} conditions: {type(e).__name__}: {e}")
        if "cudaErrorIllegalAddress" in str(e):
            print("    GPU memory access error - array bounds likely exceeded")

print(f"\n{'='*60}")
print("SUMMARY: Testing full 6-phase Au-Bi system with fixed site_fractions_offset")