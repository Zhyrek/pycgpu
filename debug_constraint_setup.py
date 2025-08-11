#!/usr/bin/env python
"""Debug constraint setup for binary vs ternary."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

# Test Au-Bi binary
print("=" * 60)
print("TESTING Au-Bi (BINARY)")
print("=" * 60)
dbf = Database('important_tests/AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
phases = ['LIQUID', 'FCC_A1']
conditions = {
    v.T: 600,
    v.P: 101325,
    v.X('BI'): 0.3
}

print(f"Components: {comps}")
non_va = [c for c in comps if c != 'VA']
print(f"Non-VA components: {non_va}, count: {len(non_va)}")
print(f"Condition: X(BI) = 0.3")
print()

# Run with verbose to see constraint setup
result = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True, calc_opts={'pdens': 50})