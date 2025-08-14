#!/usr/bin/env python
"""Check the stride calculation for initial phase data with 3 components."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Simulate the dynamic sizes that would be computed for Al-Cu-Fe
# Based on the compute_dynamic_kernel_sizes function logic
num_components = 3
num_phases = 8
num_statevars = 2
num_site_fractions = 40

# Add padding as done in compute_dynamic_kernel_sizes
component_padding_factor = 1.2
phase_padding_factor = 1.2
safety_minimum = 4

dynamic_sizes = {
    "MAX_COMPONENTS": max(int(num_components * component_padding_factor), safety_minimum),
    "MAX_PHASES": max(int(num_phases * phase_padding_factor), 6),
    "MAX_DOF_PER_PHASE": max(int(num_site_fractions * 1.25 / num_phases), 6),
    "MAX_STATEVARS": max(int(num_statevars * 1.5), 3),
}

print("Dynamic kernel sizes for Al-Cu-Fe system:")
for key, value in dynamic_sizes.items():
    print(f"  {key}: {value}")

# Calculate the initial phase data stride
MAX_PHASES = int(dynamic_sizes["MAX_PHASES"])
MAX_COMPONENTS = int(dynamic_sizes["MAX_COMPONENTS"])
MAX_DOF_PER_PHASE = int(dynamic_sizes["MAX_DOF_PER_PHASE"])

# Calculate raw size
doubles_per_struct = (MAX_PHASES + MAX_PHASES + 
                     (MAX_PHASES * MAX_DOF_PER_PHASE) + 
                     (MAX_PHASES * MAX_COMPONENTS) + 
                     MAX_COMPONENTS + 1)

print(f"\nInitial phase data structure:")
print(f"  phase_indices[{MAX_PHASES}]: {MAX_PHASES} doubles")
print(f"  phase_amounts[{MAX_PHASES}]: {MAX_PHASES} doubles")
print(f"  site_fractions[{MAX_PHASES}*{MAX_DOF_PER_PHASE}]: {MAX_PHASES * MAX_DOF_PER_PHASE} doubles")
print(f"  compositions[{MAX_PHASES}*{MAX_COMPONENTS}]: {MAX_PHASES * MAX_COMPONENTS} doubles")
print(f"  chemical_potentials[{MAX_COMPONENTS}]: {MAX_COMPONENTS} doubles")
print(f"  num_phases: 1 double")
print(f"  Raw total: {doubles_per_struct} doubles")

# Check if padding is applied
if doubles_per_struct == 65:
    padded_size = 80
    print(f"  PADDING APPLIED: {doubles_per_struct} -> {padded_size} doubles")
elif doubles_per_struct == 75:
    padded_size = 80
    print(f"  PADDING APPLIED: {doubles_per_struct} -> {padded_size} doubles")
else:
    padded_size = doubles_per_struct
    print(f"  No padding needed")

print(f"\nFinal stride: {padded_size} doubles = {padded_size * 8} bytes")

# Check memory alignment
cache_line_size = 64  # bytes
bytes_per_struct = padded_size * 8
cache_lines = bytes_per_struct / cache_line_size
print(f"Memory alignment: {bytes_per_struct} bytes = {cache_lines:.1f} cache lines")

if bytes_per_struct % cache_line_size == 0:
    print("✓ Perfectly aligned to cache line boundaries")
else:
    print(f"✗ Not aligned (off by {bytes_per_struct % cache_line_size} bytes)")