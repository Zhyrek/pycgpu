from pycalphad import Database, equilibrium
import numpy as np
import os

# Clear cache
os.system('rm -f ~/.cupy/kernel_cache/*')

# Load Au-Bi database
db = Database('/mnt/c/users/scott/Documents/pycalphad/AuBi-07Wan.tdb')
components = ['AU', 'BI', 'VA']

# Test with just two phases to simplify
print("Testing Au-Bi with limited phases")
print("=" * 60)

# Test 1: LIQUID + FCC_A1 at high temperature
print("\nTest 1: LIQUID + FCC_A1 at 1000K, X(BI)=0.3")
phases = ['LIQUID', 'FCC_A1']
conditions = {'T': 1000, 'P': 101325, 'X(BI)': 0.3}

# GPU
print("\nGPU:")
result_gpu = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 500}, gpu=True)
for phase in np.unique(result_gpu.Phase.values):
    if phase != '':
        mask = result_gpu.Phase.values == phase
        amount = result_gpu.NP.values[mask][0]
        print(f"  {phase}: NP = {amount:.6f}")

# CPU
print("\nCPU:")
result_cpu = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 500})
for phase in np.unique(result_cpu.Phase.values):
    if phase != '':
        mask = result_cpu.Phase.values == phase
        amount = result_cpu.NP.values[mask][0]
        print(f"  {phase}: NP = {amount:.6f}")

# Test 2: AU2BI_C15 + RHOMBOHEDRAL_A7 at lower temperature
print("\n" + "-" * 60)
print("\nTest 2: AU2BI_C15 + RHOMBOHEDRAL_A7 at 500K, X(BI)=0.6")
phases = ['AU2BI_C15', 'RHOMBOHEDRAL_A7']
conditions = {'T': 500, 'P': 101325, 'X(BI)': 0.6}

# GPU
print("\nGPU:")
result_gpu = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 500}, gpu=True)
for phase in np.unique(result_gpu.Phase.values):
    if phase != '':
        mask = result_gpu.Phase.values == phase
        amount = result_gpu.NP.values[mask][0]
        print(f"  {phase}: NP = {amount:.6f}")

# CPU
print("\nCPU:")
result_cpu = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 500})
for phase in np.unique(result_cpu.Phase.values):
    if phase != '':
        mask = result_cpu.Phase.values == phase
        amount = result_cpu.NP.values[mask][0]
        print(f"  {phase}: NP = {amount:.6f}")

# Test 3: Single phase region
print("\n" + "-" * 60)
print("\nTest 3: LIQUID only at 1200K, X(BI)=0.5")
phases = ['LIQUID']
conditions = {'T': 1200, 'P': 101325, 'X(BI)': 0.5}

# GPU
print("\nGPU:")
result_gpu = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 500}, gpu=True)
for phase in np.unique(result_gpu.Phase.values):
    if phase != '':
        mask = result_gpu.Phase.values == phase
        amount = result_gpu.NP.values[mask][0]
        print(f"  {phase}: NP = {amount:.6f}")

# CPU
print("\nCPU:")
result_cpu = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 500})
for phase in np.unique(result_cpu.Phase.values):
    if phase != '':
        mask = result_cpu.Phase.values == phase
        amount = result_cpu.NP.values[mask][0]
        print(f"  {phase}: NP = {amount:.6f}")