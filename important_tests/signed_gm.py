#!/usr/bin/env python
"""Signed GM comparison (GPU - CPU): who finds the lower energy, and by how much?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from pycalphad import Database, equilibrium, variables as v

nx = int(sys.argv[1]) if len(sys.argv) > 1 else 7
nt = int(sys.argv[2]) if len(sys.argv) > 2 else 5
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dbf = Database(os.path.join(ROOT, 'Al-Cu-Fe.tdb'))
comps = ['AL', 'CU', 'FE', 'VA']
phases = sorted(dbf.phases.keys())
conds = {v.X('AL'): (0.2, 0.61, 0.4 / max(nx - 1, 1)),
         v.X('CU'): (0.1, 0.41, 0.3 / max(nx - 1, 1)),
         v.T: (700, 1400.1, 700.0 / max(nt - 1, 1)), v.P: 101325, v.N: 1}

cpu = equilibrium(dbf, comps, phases, conds, gpu=False)
gpu = equilibrium(dbf, comps, phases, conds, gpu=True)

c = cpu.GM.values.flatten()
g = gpu.GM.values.flatten()
m = ~(np.isnan(c) | np.isnan(g))
sd = g[m] - c[m]  # negative => GPU found LOWER (better) energy
idx = np.where(m)[0]

tol = 1e-4
gpu_lower = sd < -tol
cpu_lower = sd > tol
print(f"n={m.sum()}  |ties within {tol}|: {np.sum(~gpu_lower & ~cpu_lower)}")
print(f"GPU strictly lower (GPU better): {gpu_lower.sum()}  total advantage {-sd[gpu_lower].sum():.4g} J/mol")
print(f"CPU strictly lower (CPU better): {cpu_lower.sum()}  total advantage {sd[cpu_lower].sum():.4g} J/mol")
print("\nTop |diff| conditions (signed, GPU-CPU; negative = GPU deeper):")
order = np.argsort(-np.abs(sd))[:12]
for k in order:
    if abs(sd[k]) < tol:
        break
    print(f"  cond {idx[k]:3d}: dGM = {sd[k]:+12.5g}")
