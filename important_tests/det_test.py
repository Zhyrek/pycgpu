#!/usr/bin/env python
"""Determinism test: run GPU equilibrium twice, compare GM. --no-pool disables CuPy memory pool."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import cupy as cp

if '--no-pool' in sys.argv:
    cp.cuda.set_allocator(None)
    print("CuPy memory pool DISABLED")

from pycalphad import Database, equilibrium, variables as v

nx, nt = 17, 15
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dbf = Database(os.path.join(ROOT, 'AuBi-07Wan.tdb'))
comps = ['AU', 'BI', 'VA']
phases = sorted(dbf.phases.keys())
conds = {v.X('BI'): (0.05, 0.951, 0.9 / (nx - 1)),
         v.T: (500, 1000.1, 500.0 / (nt - 1)),
         v.P: 101325, v.N: 1}

g1 = equilibrium(dbf, comps, phases, conds, gpu=True).GM.values.flatten()
g2 = equilibrium(dbf, comps, phases, conds, gpu=True).GM.values.flatten()
m = ~(np.isnan(g1) | np.isnan(g2))
d = np.abs(g1[m] - g2[m])
ndiff = int((d > 1e-9).sum())
print(f"run-to-run: n={m.sum()} max|dGM|={d.max():.4g} conditions differing={ndiff}")
if ndiff:
    idx = np.where(m)[0][d > 1e-9][:10]
    print("differing condition indices:", idx.tolist())
