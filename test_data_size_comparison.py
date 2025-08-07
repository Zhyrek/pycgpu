#!/usr/bin/env python
"""Compare data structure sizes between 4 and 6 phases."""

from pycalphad import Database, Workspace, variables as v
from pycalphad.gpu.gpu_codegen import compute_dynamic_kernel_sizes

# Load database
dbf = Database('AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']

# Test with 4 and 6 phases
test_cases = [
    (4, ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'RHOMBOHEDRAL_A7']),
    (6, ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'AU2BI_HEX', 'RHOMBOHEDRAL_A7', 'BCC_A2'])
]

for num_phases, phases in test_cases:
    print(f"\n{'='*60}")
    print(f"Testing with {num_phases} phases")
    
    wks = Workspace(dbf, comps, phases[:6], {v.X('BI'): 0.3, v.T: 600, v.P: 101325})
    sizes = compute_dynamic_kernel_sizes(wks)
    
    print(f"\nDynamic sizes:")
    print(f"  MAX_PHASES = {sizes['MAX_PHASES']}")
    print(f"  MAX_COMPONENTS = {sizes['MAX_COMPONENTS']}")
    print(f"  MAX_DOF_PER_PHASE = {sizes['MAX_DOF_PER_PHASE']}")
    print(f"  MAX_INTERNAL_CONSTRAINTS = {sizes['MAX_INTERNAL_CONSTRAINTS']}")
    
    # Calculate data structure sizes
    doubles_per_condition = (sizes['MAX_PHASES'] + sizes['MAX_PHASES'] + 
                           (sizes['MAX_PHASES'] * sizes['MAX_DOF_PER_PHASE']) + 
                           (sizes['MAX_PHASES'] * sizes['MAX_COMPONENTS']) + 
                           sizes['MAX_COMPONENTS'] + 1)
    
    print(f"\nMemory layout per condition:")
    print(f"  phase_indices: {sizes['MAX_PHASES']} doubles")
    print(f"  phase_amounts: {sizes['MAX_PHASES']} doubles")
    print(f"  site_fractions: {sizes['MAX_PHASES'] * sizes['MAX_DOF_PER_PHASE']} doubles")
    print(f"  compositions: {sizes['MAX_PHASES'] * sizes['MAX_COMPONENTS']} doubles")
    print(f"  chemical_potentials: {sizes['MAX_COMPONENTS']} doubles")
    print(f"  num_phases: 1 double")
    print(f"  TOTAL: {doubles_per_condition} doubles = {doubles_per_condition * 8} bytes")
    
    # Calculate offsets
    print(f"\nOffsets:")
    offset = 0
    print(f"  phase_indices: {offset}")
    offset += sizes['MAX_PHASES']
    print(f"  phase_amounts: {offset}")
    offset += sizes['MAX_PHASES']
    print(f"  site_fractions: {offset}")
    offset += sizes['MAX_PHASES'] * sizes['MAX_DOF_PER_PHASE']
    print(f"  compositions: {offset}")
    offset += sizes['MAX_PHASES'] * sizes['MAX_COMPONENTS']
    print(f"  chemical_potentials: {offset}")
    offset += sizes['MAX_COMPONENTS']
    print(f"  num_phases: {offset}")
    
    # Check for potential alignment issues
    print(f"\nPotential issues:")
    if doubles_per_condition % 8 != 0:
        print(f"  WARNING: Total size {doubles_per_condition} is not 8-double aligned")
    if offset != doubles_per_condition - 1:
        print(f"  ERROR: Offset calculation mismatch!")
        
    # Check if structure size changed
    if num_phases == 4:
        size_4_phases = doubles_per_condition
    elif num_phases == 6:
        size_6_phases = doubles_per_condition
        
# Compare
print(f"\n{'='*60}")
print(f"COMPARISON:")
print(f"  4 phases: {size_4_phases} doubles per condition")
print(f"  6 phases: {size_6_phases} doubles per condition")
print(f"  Difference: {size_6_phases - size_4_phases} doubles")

# Check for potential memory access patterns
print(f"\nMemory access pattern analysis:")
print(f"  If code assumes 4-phase structure size ({size_4_phases} doubles):")
print(f"    Condition 0 offset: 0 (correct)")
print(f"    Condition 1 offset: {size_4_phases} (should be {size_6_phases})")
print(f"    Error: Reading {size_4_phases} but should read {size_6_phases}")
print(f"    This means condition 1 reads {size_6_phases - size_4_phases} doubles into condition 0's data!")