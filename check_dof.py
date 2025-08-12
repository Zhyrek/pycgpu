from pycalphad import Database
import pycalphad.variables as v
dbf = Database('Al-Cu-Fe.tdb')
components = ['AL','CU','FE','VA']
phases = ['LIQUID']
conditions = {'T': 973.15, 'P': 101325, 'X_AL': 0.5, 'X_CU': 0.2}

# Calculate degrees of freedom
num_components = len([c for c in components if c != 'VA'])
num_phases = len(phases)
num_conditions = len([k for k in conditions if k.startswith('X_')])

print(f'Components (non-VA): {num_components}')
print(f'Phases: {num_phases}')
print(f'Fixed mole fractions: {num_conditions}')
print(f'Degrees of freedom (Gibbs phase rule): F = C - P + 2 - fixed = {num_components} - {num_phases} + 2 - {num_conditions} = {num_components - num_phases + 2 - num_conditions}')
print(f'With T and P fixed: F = C - P - fixed_X = {num_components} - {num_phases} - {num_conditions} = {num_components - num_phases - num_conditions}')