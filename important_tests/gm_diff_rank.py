#!/usr/bin/env python
"""Rank per-condition GM diffs between CPU and GPU for the alcufe bench grid.

Usage: python gm_diff_rank.py [nx] [nt]   (defaults 4 3, matching bench_baseline)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from pycalphad import Database, equilibrium, variables as v

nx = int(sys.argv[1]) if len(sys.argv) > 1 else 4
nt = int(sys.argv[2]) if len(sys.argv) > 2 else 3
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
Ts = cpu.T.values
xals = cpu.X_AL.values if hasattr(cpu, 'X_AL') else None
# reconstruct condition tuples in flatten order from coords
coord_names = [d for d in cpu.GM.dims]
shape = cpu.GM.shape
idx = np.array(np.unravel_index(np.arange(c.size), shape)).T
coords = {d: cpu.GM.coords[d].values for d in coord_names}

cph = cpu.Phase.values.reshape(-1, cpu.Phase.shape[-1])
gph = gpu.Phase.values.reshape(-1, gpu.Phase.shape[-1])
cnp = cpu.NP.values.reshape(-1, cpu.NP.shape[-1])
gnp = gpu.NP.values.reshape(-1, gpu.NP.shape[-1])

rows = []
for i in range(c.size):
    if np.isnan(c[i]) or np.isnan(g[i]):
        continue
    cs = sorted({cph[i, j] for j in range(cnp.shape[1]) if cnp[i, j] > 1e-6 and cph[i, j] != ''})
    gs = sorted({gph[i, j] for j in range(gnp.shape[1]) if gnp[i, j] > 1e-6 and gph[i, j] != ''})
    cond = {d: coords[d][idx[i][k]] for k, d in enumerate(coord_names)}
    rows.append((abs(c[i] - g[i]), i, cond, cs, gs))

rows.sort(reverse=True)
for d, i, cond, cs, gs in rows[:12]:
    cstr = ' '.join(f"{k}={v}" for k, v in cond.items() if k not in ('N', 'P'))
    print(f"dGM={d:12.6g}  cond {i:3d}  {cstr}  CPU={cs} GPU={gs}")
