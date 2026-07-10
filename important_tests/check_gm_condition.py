#!/usr/bin/env python
"""Detail dump of one bad condition: X_BI=0.32, T=625K."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from pycalphad import Database, equilibrium, variables as v

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dbf = Database(os.path.join(ROOT, 'AuBi-07Wan.tdb'))
comps = ['AU', 'BI', 'VA']
phases = sorted(dbf.phases.keys())
conds = {v.X('BI'): 0.32, v.T: 625, v.P: 101325, v.N: 1}

for label, use_gpu in [('CPU', False), ('GPU', True)]:
    r = equilibrium(dbf, comps, phases, conds, gpu=use_gpu)
    gm = float(r.GM.values.squeeze())
    mu = r.MU.values.squeeze()
    npv = r.NP.values.squeeze()
    ph = r.Phase.values.squeeze()
    x = r.X.values.squeeze()
    print(f"--- {label} ---")
    print(f"GM = {gm:.6f}")
    print(f"MU = {mu}")
    for j in range(len(ph)):
        if isinstance(ph[j], str) and ph[j]:
            print(f"  phase {ph[j]:12s} NP={npv[j]:.9f} X={x[j]}")
    print(f"sum(x*mu) = {np.dot([0.68, 0.32], mu):.6f}")
