import numpy as np
from pycalphad import Database, equilibrium, variables as v

# Setup
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']
conds = {v.T: 1800, v.P: 101325, v.X('TI'): 0.3}

print("Running GPU equilibrium...")
eq_gpu = equilibrium(dbf, comps, phases, conds, calc_opts={'pdens': 10}, gpu=True)
print(f"GPU GM: {float(eq_gpu.GM.values[0]):.6f}")

print("\nRunning CPU equilibrium...")
eq_cpu = equilibrium(dbf, comps, phases, conds, calc_opts={'pdens': 10})
print(f"CPU GM: {float(eq_cpu.GM.values[0]):.6f}")

print(f"\nGM Difference: {abs(float(eq_cpu.GM.values[0]) - float(eq_gpu.GM.values[0])):.6e} J/mol")
if abs(float(eq_cpu.GM.values[0]) - float(eq_gpu.GM.values[0])) > 0.001:
    print("ERROR: Difference exceeds 0.001 J/mol threshold!")
    
    # Print more details
    print("\nCPU Results:")
    print(f"  Phases present: {list(eq_cpu.Phase.values[0])}")
    print(f"  Phase amounts: {eq_cpu.NP.values[0]}")
    
    print("\nGPU Results:")
    print(f"  Phases present: {list(eq_gpu.Phase.values[0])}")
    print(f"  Phase amounts: {eq_gpu.NP.values[0]}")