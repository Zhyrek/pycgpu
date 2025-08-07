#!/usr/bin/env python
"""Check phase indices to understand the pattern."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Load database
dbf = Database('important_tests/AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
phases = filter_phases(dbf, comps)

print("Phase list:")
for i, phase in enumerate(phases):
    print(f"  {i}: {phase}")

# Check if phase order affects the issue
import pycalphad.variables as v
from pycalphad import Model

# Check models for each phase
print("\nPhase models:")
for phase in phases:
    mod = Model(dbf, comps, phase)
    print(f"\n{phase}:")
    print(f"  phase_dof: {mod.phase_dof}")
    print(f"  site_ratios: {mod.site_ratios}")

# Check if there's something special about FCC_A1 and AU2BI_C15
fcc_idx = phases.index('FCC_A1') if 'FCC_A1' in phases else -1
au2bi_idx = phases.index('AU2BI_C15') if 'AU2BI_C15' in phases else -1

print(f"\nFCC_A1 index: {fcc_idx}")
print(f"AU2BI_C15 index: {au2bi_idx}")

# Check if indices have any relation to 7
if fcc_idx >= 0:
    print(f"FCC_A1 index % 7 = {fcc_idx % 7}")
if au2bi_idx >= 0:
    print(f"AU2BI_C15 index % 7 = {au2bi_idx % 7}")
    
# Sum of indices
if fcc_idx >= 0 and au2bi_idx >= 0:
    print(f"Sum of indices: {fcc_idx + au2bi_idx}")
    print(f"Sum % 7 = {(fcc_idx + au2bi_idx) % 7}")