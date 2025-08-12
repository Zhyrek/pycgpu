import numpy as np
from pycalphad import Database, equilibrium
import warnings
import os

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Enable debug mode for CPU
os.environ['PYCALPHAD_DEBUG_MODE'] = '1'

# Load Al-Cu-Fe database
dbf = Database('Al-Cu-Fe.tdb')

# Test single condition
conditions = {
    'T': 973.15,
    'P': 101325,
    'X(AL)': 0.5,
    'X(CU)': 0.2
}

print('Testing Al-Cu-Fe ternary system...')
print(f'Conditions: T={conditions["T"]:.2f}K, X(AL)={conditions["X(AL)"]}, X(CU)={conditions["X(CU)"]}, X(FE)=0.3')
print('='*80)

# CPU calculation
print('\n=== CPU CALCULATION ===')
cpu_result = equilibrium(dbf, ['AL', 'CU', 'FE', 'VA'], ['LIQUID'], conditions, 
                        verbose=False, calc_opts={'pdens': 50})
cpu_gm = float(cpu_result.GM.values.squeeze())
print(f'\nCPU GM: {cpu_gm:.8f} J/mol')
print('='*80)

# GPU calculation with verbose to enable VERBOSE_DEBUG
print('\n=== GPU CALCULATION ===')
gpu_result = equilibrium(dbf, ['AL', 'CU', 'FE', 'VA'], ['LIQUID'], conditions, 
                        verbose=True, calc_opts={'pdens': 50}, gpu=True)
gpu_gm = float(gpu_result.GM.values.squeeze())
print(f'\nGPU GM: {gpu_gm:.8f} J/mol')
print('='*80)

# Compare
diff = abs(cpu_gm - gpu_gm)
print(f'\nDifference: {diff:.8f} J/mol')
if diff < 0.001:
    print('SUCCESS: GPU and CPU match to within 0.001 J/mol\!')
else:
    print(f'Still diverging by {diff:.2f} J/mol')
ENDOFFILE < /dev/null
