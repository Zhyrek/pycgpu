#!/usr/bin/env python
"""Compare FULL outputs (GM, MU, NP, Phase, X, Y) between CPU and GPU on a small alcufe grid."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from pycalphad import Database, equilibrium, variables as v

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dbf = Database(os.path.join(ROOT, 'Al-Cu-Fe.tdb'))
comps = ['AL', 'CU', 'FE', 'VA']
phases = sorted(dbf.phases.keys())
conds = {v.X('AL'): (0.2, 0.61, 0.4 / 3), v.X('CU'): (0.1, 0.41, 0.3 / 3),
         v.T: (700, 1400.1, 700.0 / 2), v.P: 101325, v.N: 1}

cpu = equilibrium(dbf, comps, phases, conds, gpu=False)
gpu = equilibrium(dbf, comps, phases, conds, gpu=True)

print(f"Y dims:  CPU {cpu.Y.shape}  GPU {gpu.Y.shape}")
print(f"X dims:  CPU {cpu.X.shape}  GPU {gpu.X.shape}")
print(f"NP dims: CPU {cpu.NP.shape}  GPU {gpu.NP.shape}")

c_gm = cpu.GM.values.flatten()
g_gm = gpu.GM.values.flatten()
valid = ~(np.isnan(c_gm) | np.isnan(g_gm))

# Per-condition, phase-set-aware comparison: sort vertices by phase name so
# vertex ordering differences don't count as mismatches.
cph = cpu.Phase.values.reshape(-1, cpu.Phase.shape[-1])
gph = gpu.Phase.values.reshape(-1, gpu.Phase.shape[-1])
cnp = cpu.NP.values.reshape(-1, cpu.NP.shape[-1])
gnp = gpu.NP.values.reshape(-1, gpu.NP.shape[-1])
cy = cpu.Y.values.reshape(cnp.shape[0], cnp.shape[1], -1)
gy = gpu.Y.values.reshape(gnp.shape[0], gnp.shape[1], -1)
cx = cpu.X.values.reshape(cnp.shape[0], cnp.shape[1], -1)
gx = gpu.X.values.reshape(gnp.shape[0], gnp.shape[1], -1)

max_np = max_y = max_x = 0.0
n_cmp = 0
skipped = 0
for i in range(cnp.shape[0]):
    if not valid[i]:
        continue
    cset = sorted([(cph[i, j], j) for j in range(cnp.shape[1]) if isinstance(cph[i, j], str) and cph[i, j]])
    gset = sorted([(gph[i, j], j) for j in range(gnp.shape[1]) if isinstance(gph[i, j], str) and gph[i, j]])
    if [p for p, _ in cset] != [p for p, _ in gset]:
        skipped += 1  # known phase-set mismatch class; counted by bench already
        continue
    n_cmp += 1
    for (pn, cj), (_, gj) in zip(cset, gset):
        max_np = max(max_np, abs(cnp[i, cj] - gnp[i, gj]))
        ndof = min(cy.shape[2], gy.shape[2])
        cyv, gyv = cy[i, cj, :ndof], gy[i, gj, :ndof]
        m = ~(np.isnan(cyv) | np.isnan(gyv))
        # NaN-pattern must match too
        if not np.array_equal(np.isnan(cyv), np.isnan(gyv)):
            print(f"  cond {i} phase {pn}: NaN pattern differs: CPU {np.isnan(cyv)} GPU {np.isnan(gyv)}")
        if m.any():
            max_y = max(max_y, np.max(np.abs(cyv[m] - gyv[m])))
        nx = min(cx.shape[2], gx.shape[2])
        cxv, gxv = cx[i, cj, :nx], gx[i, gj, :nx]
        mx = ~(np.isnan(cxv) | np.isnan(gxv))
        if mx.any():
            max_x = max(max_x, np.max(np.abs(cxv[mx] - gxv[mx])))

print(f"compared {n_cmp} conditions (skipped {skipped} known set-mismatches)")
print(f"max |dNP| = {max_np:.6g}")
print(f"max |dY|  = {max_y:.6g}")
print(f"max |dX|  = {max_x:.6g}")
