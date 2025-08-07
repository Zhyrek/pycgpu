#!/usr/bin/env python
"""Minimal test for 6 phases debug."""

from pycalphad import Database, equilibrium, variables as v

dbf = Database('AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
phases = ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'AU2BI_HEX', 'RHOMBOHEDRAL_A7', 'BCC_A2']

result = equilibrium(dbf, comps, phases, 
                    {v.X('BI'): [0.3, 0.3], v.T: 600, v.P: 101325}, 
                    gpu=True, verbose=True)

print(f"GPU GM: {result.GM.values.flatten()}")

result_cpu = equilibrium(dbf, comps, phases, 
                        {v.X('BI'): [0.3, 0.3], v.T: 600, v.P: 101325}, 
                        gpu=False, verbose=False)

print(f"CPU GM: {result_cpu.GM.values.flatten()}")
print(f"Difference: {result.GM.values.flatten() - result_cpu.GM.values.flatten()}")