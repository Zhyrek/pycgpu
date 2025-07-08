#!/usr/bin/env python
"""
Debug phase detection issue - GPU only finds 1 phase vs CPU multiple phases.
"""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings('ignore')

def debug_phase_detection():
    """Debug why GPU only detects 1 phase vs CPU multiple phases"""
    print("="*60)
    print("PHASE DETECTION DEBUG: CPU vs GPU")
    print("="*60)
    
    dbf = Database("NbTi.tdb")
    
    # Test the first condition that showed big deviation
    conditions = {v.X("TI"): 1e-10, v.T: 600, v.P: 101325}
    
    print("Condition: T=600K, X_TI=1e-10 (pure NB system)")
    
    # CPU calculation with detailed output
    print(f"\n1. CPU EQUILIBRIUM:")
    cpu_result = equilibrium(dbf, ['NB', 'TI', 'VA'], ['LIQUID', 'BCC_A2'], 
                           conditions, gpu=False, verbose=False)
    
    cpu_gm = float(cpu_result.GM.values[0,0,0,0])
    cpu_mu = cpu_result.MU.values[0,0,0,0,:]
    cpu_np = cpu_result.NP.values[0,0,0,0,:]
    cpu_phases = cpu_result.Phase.values[0,0,0,0,:]
    
    print(f"  GM: {cpu_gm:.6f}")
    print(f"  MU: {cpu_mu}")
    print(f"  NP: {cpu_np}")
    print(f"  Phases: {cpu_phases}")
    
    # Find active phases
    cpu_active_mask = cpu_np > 1e-10
    cpu_active_phases = cpu_phases[cpu_active_mask]
    cpu_active_amounts = cpu_np[cpu_active_mask]
    
    print(f"  Active phases: {cpu_active_phases}")
    print(f"  Active amounts: {cpu_active_amounts}")
    print(f"  Number of active phases: {len(cpu_active_phases)}")
    
    # Now check what the GPU starting point should be
    print(f"\n2. EXPECTED GPU STARTING POINT:")
    print(f"  Based on CPU result, GPU should start with {len(cpu_active_phases)} active phases")
    for i, (phase, amount) in enumerate(zip(cpu_active_phases, cpu_active_amounts)):
        print(f"    Phase {i}: {phase} with amount {amount:.6f}")
    
    # Check if this is a single-phase equilibrium or multi-phase
    if len(cpu_active_phases) == 1:
        print(f"  → This is a SINGLE-PHASE equilibrium")
        print(f"  → GPU should also find 1 phase, but chemical potentials must match")
    else:
        print(f"  → This is a MULTI-PHASE equilibrium")
        print(f"  → GPU starting with only 1 phase will give wrong results")
    
    # Check what the starting point procedure would generate
    print(f"\n3. ANALYSIS:")
    if len(cpu_active_phases) == 1 and 'BCC_A2' in str(cpu_active_phases[0]):
        print(f"  Pure NB at 600K should be single BCC_A2 phase")
        print(f"  GPU finding 1 phase is correct")
        print(f"  Issue must be in chemical potential calculation")
        print(f"  CPU MU: {cpu_mu[:2]}")
        print(f"  Expected GPU to match this exactly")
    else:
        print(f"  Unexpected phase configuration!")

if __name__ == "__main__":
    debug_phase_detection()