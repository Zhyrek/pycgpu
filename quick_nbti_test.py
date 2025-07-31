#!/usr/bin/env python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from pycalphad import Database, equilibrium
import numpy as np

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['LIQUID', 'BCC_A2']
conds = {'T': 1500, 'P': 101325, 'X(TI)': 0.5}

# CPU
eq_cpu = equilibrium(dbf, comps, phases, conds, calc_opts={'pdens': 50}, verbose=False)
cpu_gm = float(eq_cpu.GM.values.flatten()[0])

# GPU  
eq_gpu = equilibrium(dbf, comps, phases, conds, calc_opts={'pdens': 50}, verbose=False, gpu=True)
gpu_gm = float(eq_gpu.GM.values.flatten()[0])

diff = abs(gpu_gm - cpu_gm)
print(f"CPU: {cpu_gm:.6f} J/mol")
print(f"GPU: {gpu_gm:.6f} J/mol")
print(f"Difference: {diff:.6f} J/mol")