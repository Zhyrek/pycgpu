import pycalphad as pycal
from pycalphad import Database, equilibrium
import numpy as np

# Load the Al-Cu-Fe database
db = Database('Al-Cu-Fe.tdb')

# Define components
components = ['AL', 'CU', 'FE', 'VA']

# Define phases to consider
phases = list(db.phases.keys())
print(f"Available phases: {phases}")

# Test conditions - Al-Cu-Fe ternary at 1000K
# Note: Only specify 2 compositions for a 3-component system
conditions = {
    'T': 1000,
    'P': 101325,
    'X(AL)': 0.3,
    'X(CU)': 0.3
}

print("\nRunning equilibrium calculation with verbose=True...")
print("Looking for phase-local conditions...\n")

# Run equilibrium calculation with verbose mode
result = equilibrium(db, components, phases, conditions, 
                    verbose=True,
                    calc_opts={'pdens': 2000})

print("\n\nEquilibrium phases found:")
for phase in np.unique(result.Phase.values[result.NP.values > 1e-6]):
    if phase != '':
        print(f"  {phase}")

print("\nPhase amounts:")
for phase in np.unique(result.Phase.values):
    if phase != '':
        mask = result.Phase.values == phase
        amount = result.NP.values[mask][0]
        if amount > 1e-10:
            print(f"  {phase}: {amount:.6f}")