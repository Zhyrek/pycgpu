#!/usr/bin/env python
"""Check phase structures in Al-Cu-Fe.tdb"""

from pycalphad import Database

db = Database('Al-Cu-Fe.tdb')

print("Phase structures in Al-Cu-Fe.tdb:")
print("="*60)

# Check all phases for sublattice structure
phase_info = []
for phase_name in sorted(db.phases.keys()):
    phase = db.phases[phase_name]
    num_sublattices = len(phase.sublattices)
    has_va = any('VA' in str(constituents) for constituents in phase.constituents)
    phase_info.append((phase_name, num_sublattices, has_va, phase.sublattices, phase.constituents))

# Sort by number of sublattices
phase_info.sort(key=lambda x: (x[1], x[0]))

for name, num_sub, has_va, sublattices, constituents in phase_info:
    print(f"\n{name}:")
    print(f"  Sublattices: {sublattices} (count: {num_sub})")
    print(f"  Constituents: {constituents}")
    print(f"  Has VA: {has_va}")

# Find phases with multiple sublattices but no VA
print("\n" + "="*60)
print("Phases with multiple sublattices but NO vacancy:")
for name, num_sub, has_va, sublattices, constituents in phase_info:
    if num_sub > 1 and not has_va:
        print(f"- {name}: {sublattices} with {constituents}")

print("\nPhases with multiple sublattices AND vacancy:")
for name, num_sub, has_va, sublattices, constituents in phase_info:
    if num_sub > 1 and has_va:
        print(f"- {name}: {sublattices} with {constituents}")