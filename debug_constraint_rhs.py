#!/usr/bin/env python

import warnings
warnings.filterwarnings('ignore')
from pycalphad import Database, equilibrium

dbf = Database('Al-Cu-Fe.tdb')
result = equilibrium(dbf, ['AL','CU','FE','VA'], ['LIQUID'], 
                    {'T': 973.15, 'P': 101325, 'X_AL': 0.5, 'X_CU': 0.2}, 
                    verbose=True, calc_opts={'pdens': 50}, gpu=True)