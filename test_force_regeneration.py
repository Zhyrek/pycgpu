#!/usr/bin/env python3
"""Force GPU code regeneration"""
from pycalphad import Database, equilibrium, variables as v
from pycalphad.gpu.gpu_equilibrium import clear_gpu_cache

# Clear the cache
clear_gpu_cache()
print("GPU cache cleared")

# Now run equilibrium - this should trigger code generation
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2'] 
conditions = {v.N: 1, v.P: 101325, v.T: 1000, v.X('TI'): 0.4}

print("\nRunning equilibrium with GPU (should regenerate code)...")
result = equilibrium(db, comps, phases, conditions, gpu=True, calc_opts={'pdens': 5})
print(f"GPU GM: {float(result.GM.values.flatten()[0])}")