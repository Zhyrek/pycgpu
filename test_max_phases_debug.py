#!/usr/bin/env python
"""Debug MAX_PHASES calculation in detail."""

from pycalphad import Database, Workspace, variables as v
from pycalphad.gpu.gpu_codegen import compute_dynamic_kernel_sizes

# Load database
dbf = Database('AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']

# Test with 5 phases specifically
all_phases = ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'AU2BI_HEX', 'RHOMBOHEDRAL_A7']
phases = all_phases[:5]

print(f"Testing with phases: {phases}")
print(f"Number of phases: {len(phases)}")

wks = Workspace(dbf, comps, phases, {v.X('BI'): 0.3, v.T: 600, v.P: 101325})

# Check what wks.phases actually contains
print(f"\nWorkspace phases: {wks.phases}")
print(f"Number of workspace phases: {len(wks.phases)}")

# Manual calculation
actual_phases = len(wks.phases)
padding_factor = 1.2
safety_minimum = 4

calc_value = int(actual_phases * padding_factor)
max_phases = max(safety_minimum, calc_value)

print(f"\nManual calculation:")
print(f"  actual_phases = {actual_phases}")
print(f"  int({actual_phases} * {padding_factor}) = {calc_value}")
print(f"  max({safety_minimum}, {calc_value}) = {max_phases}")

# Get the computed sizes
sizes = compute_dynamic_kernel_sizes(wks)
print(f"\nComputed MAX_PHASES = {sizes['MAX_PHASES']}")

# Check if there's filtering happening
print(f"\nChecking phase models:")
for phase in wks.phases:
    if phase in wks.models:
        print(f"  {phase}: model exists")
    else:
        print(f"  {phase}: NO MODEL")