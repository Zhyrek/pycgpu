#!/usr/bin/env python
"""Check the size of InitialPhaseData structure for the AuBi system."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

def check_data_size():
    """Check the InitialPhaseData size for AuBi system."""
    
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    phases = filter_phases(dbf, comps)
    
    print("Checking InitialPhaseData size for AuBi system...")
    print("=" * 60)
    
    # The system has:
    # - 3 components (AU, BI, VA)
    # - 6 phases available
    
    # From gpu_equilibrium.py line 1427:
    # doubles_per_struct = MAX_PHASES + MAX_PHASES + (MAX_PHASES * MAX_DOF_PER_PHASE) + 
    #                      (MAX_PHASES * MAX_COMPONENTS) + MAX_COMPONENTS + 1
    
    # Let's determine the actual values
    MAX_COMPONENTS = 3  # AU, BI, VA
    MAX_PHASES = 6      # From the available phases list
    MAX_DOF_PER_PHASE = 3  # Typical value
    
    print(f"System parameters:")
    print(f"  MAX_COMPONENTS: {MAX_COMPONENTS}")
    print(f"  MAX_PHASES: {MAX_PHASES}")
    print(f"  MAX_DOF_PER_PHASE: {MAX_DOF_PER_PHASE}")
    print(f"  Available phases: {phases}")
    
    # Calculate size
    doubles_per_struct = (
        MAX_PHASES +                          # phase_indices: 6
        MAX_PHASES +                          # phase_amounts: 6
        (MAX_PHASES * MAX_DOF_PER_PHASE) +    # phase_dof: 6*3 = 18
        (MAX_PHASES * MAX_COMPONENTS) +       # compositions: 6*3 = 18
        MAX_COMPONENTS +                      # chemical_potentials: 3
        1                                      # num_phases: 1
    )
    
    print(f"\nInitialPhaseData structure size:")
    print(f"  phase_indices: {MAX_PHASES} doubles")
    print(f"  phase_amounts: {MAX_PHASES} doubles")
    print(f"  phase_dof: {MAX_PHASES * MAX_DOF_PER_PHASE} doubles")
    print(f"  compositions: {MAX_PHASES * MAX_COMPONENTS} doubles")
    print(f"  chemical_potentials: {MAX_COMPONENTS} doubles")
    print(f"  num_phases: 1 double")
    print(f"  TOTAL: {doubles_per_struct} doubles = {doubles_per_struct * 8} bytes")
    
    # Check if this triggers the padding
    if doubles_per_struct == 65:
        print(f"\n⚠️  Size = 65 doubles - This triggers padding to 80 doubles!")
        print(f"    Affected threads: those where thread_id % 7 = 3")
        print(f"    In 32-thread batch: threads 3, 10, 17, 24, 31")
    elif doubles_per_struct == 75:
        print(f"\n⚠️  Size = 75 doubles - This triggers padding to 80 doubles!")
        print(f"    Affected threads: those where thread_id % 7 = 3")
    else:
        print(f"\n✓  Size = {doubles_per_struct} doubles - No special padding needed")
    
    # Test with verbose mode to see the actual padding
    print(f"\n" + "=" * 60)
    print("Running test with verbose mode to confirm padding...")
    
    conditions = {
        v.X('BI'): [0.1, 0.2],
        v.T: [400, 500],
        v.P: 101325
    }
    
    # Run with verbose to see padding messages
    result = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)
    
if __name__ == "__main__":
    check_data_size()