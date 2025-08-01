from pycalphad import Database, equilibrium
import numpy as np
import os

os.system('rm -f ~/.cupy/kernel_cache/*')

db = Database('/mnt/c/users/scott/Documents/pycalphad/tests/databases/Ni-Ti_Dupin_2018.tdb')
components = ['NI', 'TI', 'VA']
phases = ['LIQUID', 'FCC_L12']
conditions = {'T': 500, 'P': 101325, 'X(TI)': 0.2}

print('Testing single condition that failed: T=500K, X(TI)=0.2')
print('=' * 60)

# GPU
result_gpu = equilibrium(db, components, phases, conditions, 
                       calc_opts={'pdens': 2000}, gpu=True, verbose=True)
print('\nGPU Results:')
for phase in np.unique(result_gpu.Phase.values):
    if phase != '':
        mask = result_gpu.Phase.values == phase
        amount = result_gpu.NP.values[mask][0]
        print(f'  {phase}: NP = {amount:.6f}')
print(f'GPU GM: {result_gpu.GM.values[0]}')

# CPU
result_cpu = equilibrium(db, components, phases, conditions, 
                       calc_opts={'pdens': 2000})
print('\nCPU Results:')
for phase in np.unique(result_cpu.Phase.values):
    if phase != '':
        mask = result_cpu.Phase.values == phase
        amount = result_cpu.NP.values[mask][0]
        print(f'  {phase}: NP = {amount:.6f}')
print(f'CPU GM: {result_cpu.GM.values[0]}')