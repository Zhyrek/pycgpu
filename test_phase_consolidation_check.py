#!/usr/bin/env python
"""Check if phase consolidation is occurring during GPU solve."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import os

# Load database
db = Database('NbTi.tdb')
components = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'HCP_A3']

# Test conditions - in miscibility gap
conditions = {
    v.T: 500,
    v.P: 101325,
    v.X('TI'): 0.1,
    v.N: 1
}

print("Testing phase consolidation...")
print("="*80)

# Set environment variable to enable GPU debug output
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['GPU_DEBUG'] = '1'

# Run GPU equilibrium with verbose output
print("\nRunning GPU equilibrium with debug output...")
print("(Look for consolidation messages in the output)")
print("-"*80)

try:
    gpu_eq = equilibrium(db, components, phases, conditions, 
                        calc_opts={'pdens': 50}, verbose=True, gpu=True)
    
    # Extract results
    gpu_np = gpu_eq.NP.values[0, 0, 0, 0]
    gpu_phases = gpu_eq.Phase.values[0, 0, 0, 0]
    gpu_gm = gpu_eq.GM.values[0, 0, 0, 0]
    
    # Count active phases
    active_phases = 0
    for i in range(len(gpu_np)):
        if gpu_np[i] > 0:
            active_phases += 1
    
    print("-"*80)
    print(f"\nGPU Result: {active_phases} active phases, GM = {gpu_gm:.6f}")
    
    if active_phases == 1:
        print("\n✗ GPU consolidated phases into single phase")
        print("  This confirms the immiscibility gap handling issue")
    else:
        print("\n✓ GPU maintained phase separation")
        
except Exception as e:
    print(f"\nERROR: {type(e).__name__}: {e}")

print("\n" + "="*80)