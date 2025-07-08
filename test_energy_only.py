import numpy as np
from pycalphad import Database, equilibrium, variables as v

# Setup
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']
conds = {v.T: 1800, v.P: 101325, v.X('TI'): 0.3}

print("Running GPU equilibrium...")
try:
    eq_gpu = equilibrium(dbf, comps, phases, conds, calc_opts={'pdens': 2}, gpu=True)
    print(f"GPU completed successfully")
    print(f"GPU GM: {float(eq_gpu.GM.values[0]):.6f}")
except Exception as e:
    print(f"GPU failed with error: {e}")

print("\nRunning CPU equilibrium...")
eq_cpu = equilibrium(dbf, comps, phases, conds, calc_opts={'pdens': 2})
print(f"CPU GM: {float(eq_cpu.GM.values[0]):.6f}")