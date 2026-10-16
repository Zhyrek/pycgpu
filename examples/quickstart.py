"""Al-Fe: an equilibrium grid with pycalphad_cg, used exactly like pycalphad's."""
from importlib.resources import files

import numpy as np
from pycalphad import Database, variables as v

from pycalphad_cg import CGWorkspace, equilibrium

dbf = Database(str(files('pycalphad.tests.databases') / 'alfe.tdb'))
comps = ['AL', 'FE', 'VA']
phases = list(dbf.phases)
conds = {v.T: (600, 1801, 100), v.X('AL'): (0.05, 0.96, 0.05), v.P: 101325, v.N: 1}

eq = equilibrium(dbf, comps, phases, conds, output='HM')
print(eq)

# as a Workspace, everything after the equilibrium (properties, derivatives) is pycalphad's
wks = CGWorkspace(dbf, comps, phases, {**conds, v.T: 1000})
x_al, cp = wks.get('X(AL)', 'HM.T')
print('heat capacity at 1000 K:')
for x, c in zip(np.ravel(x_al)[::3], np.ravel(cp)[::3]):
    print(f'  x(Al) = {x:.2f}: {c:.2f} J/mol/K')
