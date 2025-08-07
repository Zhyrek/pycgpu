#!/usr/bin/env python
"""Debug test to check site_fractions_offset calculation."""

import numpy as np
from pycalphad import Database, Workspace, variables as v
from pycalphad.gpu.gpu_codegen import compute_dynamic_kernel_sizes

# Load database
dbf = Database('AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']

# Test with 4 phases
phases = ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'RHOMBOHEDRAL_A7']

# Create workspace
wks = Workspace(dbf, comps, phases, {v.X('BI'): 0.3, v.T: 600, v.P: 101325})
sizes = compute_dynamic_kernel_sizes(wks)

print(f"With {len(phases)} phases:")
print(f"  MAX_PHASES = {sizes['MAX_PHASES']}")
print(f"  MAX_COMPONENTS = {sizes['MAX_COMPONENTS']}")
print(f"  MAX_DOF_PER_PHASE = {sizes['MAX_DOF_PER_PHASE']}")

# Calculate offsets
max_phases = sizes['MAX_PHASES']
print(f"\nOffsets:")
print(f"  site_fractions_offset = MAX_PHASES + MAX_PHASES = {max_phases} + {max_phases} = {max_phases + max_phases}")
print(f"  (old hard-coded value was 8)")

# Test with 6 phases
phases_6 = ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'AU2BI_HEX', 'RHOMBOHEDRAL_A7', 'BCC_A2']
wks_6 = Workspace(dbf, comps, phases_6, {v.X('BI'): 0.3, v.T: 600, v.P: 101325})
sizes_6 = compute_dynamic_kernel_sizes(wks_6)

print(f"\nWith {len(phases_6)} phases:")
print(f"  MAX_PHASES = {sizes_6['MAX_PHASES']}")
print(f"  site_fractions_offset = {sizes_6['MAX_PHASES']} + {sizes_6['MAX_PHASES']} = {sizes_6['MAX_PHASES'] + sizes_6['MAX_PHASES']}")