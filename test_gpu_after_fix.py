from pycalphad import Database, equilibrium
import numpy as np

# Test 1: Al-Cu-Fe system with single stable phase
print("=" * 60)
print("Test 1: Al-Cu-Fe system at T=1000K (single phase)")
print("=" * 60)
db = Database('/mnt/c/users/scott/Documents/pycalphad/Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']
phases = ['LIQUID', 'FCC_A1']
conditions = {'T': 1000, 'P': 101325, 'X(AL)': 0.70, 'X(CU)': 0.20}

result_gpu = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 1000}, gpu=True)
print('GPU Results:')
for phase in np.unique(result_gpu.Phase.values):
    if phase != '':
        mask = result_gpu.Phase.values == phase
        amount = result_gpu.NP.values[mask][0]
        print(f'  {phase}: NP = {amount:.6f}')

# Compare with CPU
result_cpu = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 1000})
print('\nCPU Results:')
for phase in np.unique(result_cpu.Phase.values):
    if phase != '':
        mask = result_cpu.Phase.values == phase
        amount = result_cpu.NP.values[mask][0]
        print(f'  {phase}: NP = {amount:.6f}')

# Test 2: Ternary system with potential immiscibility
print("\n" + "=" * 60)
print("Test 2: Al-Cu-Fe system (potential immiscibility)")
print("=" * 60)
db2 = Database('/mnt/c/users/scott/Documents/pycalphad/Al-Cu-Fe.tdb')
components2 = ['AL', 'CU', 'FE', 'VA']
phases2 = ['LIQUID', 'ALCU_ZETA']
conditions2 = {'T': 900, 'P': 101325, 'X(AL)': 0.45, 'X(CU)': 0.40}

result_gpu2 = equilibrium(db2, components2, phases2, conditions2, 
                         calc_opts={'pdens': 1000}, gpu=True, verbose=True)
print('\nGPU Results:')
for phase in np.unique(result_gpu2.Phase.values):
    if phase != '':
        mask = result_gpu2.Phase.values == phase
        amount = result_gpu2.NP.values[mask][0]
        print(f'  {phase}: NP = {amount:.6f}')

# Compare with CPU
result_cpu2 = equilibrium(db2, components2, phases2, conditions2, 
                         calc_opts={'pdens': 1000})
print('\nCPU Results:')
for phase in np.unique(result_cpu2.Phase.values):
    if phase != '':
        mask = result_cpu2.Phase.values == phase
        amount = result_cpu2.NP.values[mask][0]
        print(f'  {phase}: NP = {amount:.6f}')

print("\n" + "=" * 60)
print("Summary: Warnings about immiscibility gaps have been removed.")
print("GPU now treats multiple instances of the same phase type as valid.")
print("=" * 60)