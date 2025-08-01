from pycalphad import Database, equilibrium
import numpy as np
import os

# Enable GPU debug
os.environ['GPU_DEBUG'] = '1'

db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']
phases = ['LIQUID', 'ALCU_ZETA']

conditions = {
    'T': 900,
    'P': 101325,
    'X(AL)': 0.45,
    'X(CU)': 0.40
}

print("Running GPU calculation with verbose output...")
try:
    result_gpu = equilibrium(db, components, phases, conditions, 
                           calc_opts={'pdens': 1000},
                           gpu=True,
                           verbose=True)
    
    print("\nGPU Results:")
    for phase in np.unique(result_gpu.Phase.values):
        if phase != '':
            mask = result_gpu.Phase.values == phase
            amount = result_gpu.NP.values[mask][0]
            print(f"  {phase}: NP = {amount:.6f}")
            
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Also test with more extreme conditions to force both phases
print("\n\nTrying different conditions to force both phases...")
conditions2 = {
    'T': 850,
    'P': 101325,
    'X(AL)': 0.42,
    'X(CU)': 0.42
}

try:
    result_gpu2 = equilibrium(db, components, phases, conditions2, 
                            calc_opts={'pdens': 1000},
                            gpu=True)
    
    print("\nGPU Results (attempt 2):")
    for phase in np.unique(result_gpu2.Phase.values):
        if phase != '':
            mask = result_gpu2.Phase.values == phase
            amount = result_gpu2.NP.values[mask][0]
            print(f"  {phase}: NP = {amount:.6f}")
            
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test CPU for comparison
print("\n\nCPU comparison:")
result_cpu = equilibrium(db, components, phases, conditions, 
                        calc_opts={'pdens': 1000})

print("CPU Results:")
for phase in np.unique(result_cpu.Phase.values):
    if phase != '':
        mask = result_cpu.Phase.values == phase
        amount = result_cpu.NP.values[mask][0]
        print(f"  {phase}: NP = {amount:.6f}")