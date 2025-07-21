#!/usr/bin/env python3
import os
import sys

# Redirect all output
class DevNull:
    def write(self, msg):
        pass
    def flush(self):
        pass

os.environ['PYCALPHAD_GPU_DEBUG'] = '0'
old_stdout = sys.stdout
old_stderr = sys.stderr
sys.stdout = DevNull()
sys.stderr = DevNull()

import numpy as np
from pycalphad import Database, equilibrium
import warnings
warnings.filterwarnings('ignore')

db = Database('NbTi.tdb')
cond = {'T': 600, 'P': 101325, 'X(TI)': 0.1}

gpu_eq = equilibrium(db, ['NB', 'TI'], 'BCC_A2', cond,
                   model=None, verbose=False,
                   calc_opts={'pdens': 50}, gpu=True)

cpu_eq = equilibrium(db, ['NB', 'TI'], 'BCC_A2', cond,
                   model=None, verbose=False,
                   calc_opts={'pdens': 50}, gpu=False)

sys.stdout = old_stdout
sys.stderr = old_stderr

gpu_x = float(gpu_eq.isel(vertex=0).X_BCC_A2_TI.values)
cpu_x = float(cpu_eq.isel(vertex=0).X_BCC_A2_TI.values)

print('Final Results After GPU Fixes:')
print(f'X(TI)=0.1, T=600K')
print(f'CPU: {cpu_x:.6f}') 
print(f'GPU: {gpu_x:.6f}')
print(f'Error: {100*abs(gpu_x - cpu_x)/cpu_x:.3f}%')
print(f'Previous error was 2.653%')