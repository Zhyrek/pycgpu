#!/usr/bin/env python
"""Check number of components."""

from pycalphad import Database

# AU-BI
db = Database('important_tests/AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
non_va = [c for c in comps if c != 'VA']
print(f"Au-Bi components: {comps}")
print(f"Au-Bi non-VA components: {non_va}, count: {len(non_va)}")

# Al-Cu-Fe
db2 = Database('Al-Cu-Fe.tdb')
comps2 = ['AL', 'CU', 'FE', 'VA']
non_va2 = [c for c in comps2 if c != 'VA']
print(f"Al-Cu-Fe components: {comps2}")
print(f"Al-Cu-Fe non-VA components: {non_va2}, count: {len(non_va2)}")