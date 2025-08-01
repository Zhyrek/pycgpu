from pycalphad import Database, equilibrium, variables as v
import numpy as np

# Load database
db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']
phases = ['BCC_B2', 'L12', 'LIQUID']

# Standard conditions
conditions = {
    'T': 1000,
    'P': 101325,
    'X(AL)': 0.3,
    'X(CU)': 0.3
}

print("Test 1: Standard equilibrium (no phase-local conditions)")
result1 = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 500})
print(f"Phases: {list(set(result1.Phase.values.flatten()) - {''})}")

# Try to add a phase-local condition (fix site fraction in BCC_B2)
print("\nTest 2: With phase-local condition Y(BCC_B2,AL,0)=0.5")
conditions_with_local = conditions.copy()
conditions_with_local[v.Y('BCC_B2', 0, 'AL')] = 0.5

try:
    result2 = equilibrium(db, components, phases, conditions_with_local, calc_opts={'pdens': 500})
    print(f"Phases: {list(set(result2.Phase.values.flatten()) - {''})}")
    print("Phase-local condition successfully applied!")
except Exception as e:
    print(f"Error with phase-local condition: {type(e).__name__}: {e}")

# Check in solver if local conditions are detected
import pycalphad.core.solver as solver_module
print("\nChecking solver code for phase-local condition handling...")

# Read solver source to see get_system_spec
import inspect
source = inspect.getsource(solver_module.Solver.get_system_spec)
if 'local_conditions' in source:
    print("Found 'local_conditions' in solver.get_system_spec")
    # Find the relevant lines
    lines = source.split('\n')
    for i, line in enumerate(lines):
        if 'local_conditions' in line:
            print(f"  Line {i}: {line.strip()}")