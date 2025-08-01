#\!/usr/bin/env python
"""
Check available phases in AuBi database
"""

from pycalphad import Database, Model

# Load database
db = Database('/mnt/c/users/scott/Documents/pycalphad/AuBi-07Wan.tdb')
components = ['AU', 'BI', 'VA']

print("Available phases in AuBi database:")
for phase_name in sorted(db.phases.keys()):
    print(f"\n{phase_name}:")
    try:
        mod = Model(db, components, phase_name)
        print(f"  Site ratios: {mod.site_ratios}")
        print(f"  Sum of site ratios: {sum(mod.site_ratios)}")
        print(f"  Sublattices: {len(mod.site_ratios)}")
        # Check if it has vacancy
        has_vacancy = False
        for const in db.phases[phase_name].constituents:
            if any('VA' in str(species) for species in const):
                has_vacancy = True
                break
        print(f"  Has vacancy: {has_vacancy}")
    except Exception as e:
        print(f"  Error: {e}")
