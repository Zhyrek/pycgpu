#!/usr/bin/env python3
import os
os.environ['PYCALPHAD_GPU_DEBUG'] = '0'

import numpy as np
from pycalphad import Database, equilibrium
import warnings
warnings.filterwarnings('ignore')

# Test X(TI)=0.1, T=600K
db = Database('NbTi.tdb')
cond = {'T': 600, 'P': 101325, 'X(TI)': 0.1}

print("Testing system amount constraint fix...")

gpu_eq = equilibrium(db, ['NB', 'TI'], 'BCC_A2', cond,
                   model=None, verbose=False,
                   calc_opts={'pdens': 50}, gpu=True)

cpu_eq = equilibrium(db, ['NB', 'TI'], 'BCC_A2', cond,
                   model=None, verbose=False,
                   calc_opts={'pdens': 50}, gpu=False)

gpu_gm = float(gpu_eq.GM.values[0])
cpu_gm = float(cpu_eq.GM.values[0])

print(f"\nX(TI)=0.1, T=600K:")
print(f"CPU GM: {cpu_gm:.1f} J/mol")
print(f"GPU GM: {gpu_gm:.1f} J/mol")
print(f"Difference: {abs(gpu_gm - cpu_gm):.1f} J/mol")
print(f"Previous error: 8.67 J/mol")
print(f"Status: {'FIXED!' if abs(gpu_gm - cpu_gm) < 1.0 else 'Still failing'}")