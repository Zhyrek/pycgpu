#!/usr/bin/env python
"""Test 6 phases with multiple conditions."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

dbf = Database('AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
phases = ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'AU2BI_HEX', 'RHOMBOHEDRAL_A7', 'BCC_A2']

# Test with varying compositions
x_bi_values = [0.1, 0.2, 0.3, 0.4, 0.5]

result_gpu = equilibrium(dbf, comps, phases, 
                        {v.X('BI'): x_bi_values, v.T: 600, v.P: 101325}, 
                        gpu=True, verbose=False)

result_cpu = equilibrium(dbf, comps, phases, 
                        {v.X('BI'): x_bi_values, v.T: 600, v.P: 101325}, 
                        gpu=False, verbose=False)

print(f"GPU GM: {result_gpu.GM.values.flatten()}")
print(f"CPU GM: {result_cpu.GM.values.flatten()}")
print(f"Max difference: {np.max(np.abs(result_gpu.GM.values.flatten() - result_cpu.GM.values.flatten()))}")
print(f"All differences < 1e-6: {np.all(np.abs(result_gpu.GM.values.flatten() - result_cpu.GM.values.flatten()) < 1e-6)}")