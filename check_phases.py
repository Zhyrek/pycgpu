#!/usr/bin/env python
"""Check which phases are included by filter_phases."""

from pycalphad import Database
from pycalphad.core.utils import filter_phases

# Load database
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']

# Get all phases
all_phases = filter_phases(dbf, comps)
print(f"filter_phases returns: {all_phases}")

# Compare with explicit list
explicit_phases = ['BCC_A2', 'HCP_A3']
print(f"Explicit phases: {explicit_phases}")

# Show difference
extra_phases = set(all_phases) - set(explicit_phases)
if extra_phases:
    print(f"\nExtra phases in filter_phases: {extra_phases}")
    
missing_phases = set(explicit_phases) - set(all_phases)
if missing_phases:
    print(f"\nMissing phases in filter_phases: {missing_phases}")