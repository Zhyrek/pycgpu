#!/usr/bin/env python3
import os
import sys

# Force reimport of pycalphad
if 'pycalphad' in sys.modules:
    del sys.modules['pycalphad']

os.environ['PYCALPHAD_GPU_DEBUG'] = '0'

import numpy as np
from pycalphad import Database, equilibrium
import warnings
warnings.filterwarnings('ignore')

print("Running test with freshly compiled GPU kernel...")

# Test X(TI)=0.1, T=600K
db = Database('NbTi.tdb')
cond = {'T': 600, 'P': 101325, 'X(TI)': 0.1}

gpu_eq = equilibrium(db, ['NB', 'TI'], 'BCC_A2', cond,
                   model=None, verbose=False,
                   calc_opts={'pdens': 50}, gpu=True)

cpu_eq = equilibrium(db, ['NB', 'TI'], 'BCC_A2', cond,
                   model=None, verbose=False,
                   calc_opts={'pdens': 50}, gpu=False)

gpu_gm = float(gpu_eq.GM.values[0])
cpu_gm = float(cpu_eq.GM.values[0])

print(f"\nResults for X(TI)=0.1, T=600K:")
print(f"CPU GM: {cpu_gm:.1f} J/mol")
print(f"GPU GM: {gpu_gm:.1f} J/mol")
print(f"Difference: {abs(gpu_gm - cpu_gm):.1f} J/mol")
print(f"\nPrevious error: 8.67 J/mol")
print(f"Fix applied: System amount constraint moved outside phase loops")
print(f"Status: {'SUCCESS - Error reduced!' if abs(gpu_gm - cpu_gm) < 8.67 else 'No improvement'}")