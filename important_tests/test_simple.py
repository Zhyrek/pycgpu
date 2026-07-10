import os
os.environ['HIP_LAUNCH_BLOCKING'] = '1'

# Minimal test to reproduce the crash
from pycalphad import Database, equilibrium, variables as v
import numpy as np

# Use the Al-Cu-Fe database that exists
db = Database("Al-Cu-Fe.tdb")

# Single condition - as minimal as possible
print("Running minimal GPU test with 1 condition...")
result = equilibrium(
    db, 
    ['AL', 'CU', 'FE', 'VA'], 
    'LIQUID',
    {v.T: 1000, v.P: 101325, v.X('AL'): 0.3, v.X('CU'): 0.3},
    gpu=True,
    verbose=True
)

print("Success! GPU equilibrium completed.")
print(f"GM = {result.GM.values[0]}")
