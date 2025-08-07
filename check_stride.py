#!/usr/bin/env python
"""Check what stride is being calculated for AuBi system."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Load database and set up calculation
dbf = Database('important_tests/AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
phases = filter_phases(dbf, comps)

print(f"Phases: {phases}")
print(f"Number of phases: {len(phases)}")
print(f"Number of components: {len(comps)}")

# Calculate struct size
# Assuming MAX_DOF_PER_PHASE = 4 (default)
MAX_PHASES = len(phases) + 1  # Account for _FAKE_ phase
MAX_COMPONENTS = 4  # AU, BI, VA, plus one for normalization
MAX_DOF_PER_PHASE = 4

doubles_per_struct = MAX_PHASES + MAX_PHASES + (MAX_PHASES * MAX_DOF_PER_PHASE) + (MAX_PHASES * MAX_COMPONENTS) + MAX_COMPONENTS + 1

print(f"\nCalculated sizes:")
print(f"MAX_PHASES: {MAX_PHASES}")
print(f"MAX_COMPONENTS: {MAX_COMPONENTS}")
print(f"MAX_DOF_PER_PHASE: {MAX_DOF_PER_PHASE}")
print(f"doubles_per_struct: {doubles_per_struct}")
print(f"  = {MAX_PHASES} + {MAX_PHASES} + {MAX_PHASES*MAX_DOF_PER_PHASE} + {MAX_PHASES*MAX_COMPONENTS} + {MAX_COMPONENTS} + 1")
print(f"  = {MAX_PHASES} + {MAX_PHASES} + {MAX_PHASES*MAX_DOF_PER_PHASE} + {MAX_PHASES*MAX_COMPONENTS} + {MAX_COMPONENTS} + 1")
print(f"  = 7 + 7 + 28 + 28 + 4 + 1 = {doubles_per_struct}")

# Test if padding would be applied
if doubles_per_struct == 65:
    print("\n✗ This struct size (65) would trigger padding to 80")
else:
    print(f"\n✓ This struct size ({doubles_per_struct}) does NOT trigger padding")