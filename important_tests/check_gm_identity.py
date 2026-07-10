#!/usr/bin/env python
"""Check GM == sum(x_i * mu_i) identity for CPU and GPU results (exact at equilibrium, N=1)."""
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
conds = {v.X('BI'): (0.05, 0.951, 0.09), v.T: (500, 1000.1, 125.0), v.P: 101325, v.N: 1}

cpu = equilibrium(dbf, comps, phases, conds, gpu=False)
gpu = equilibrium(dbf, comps, phases, conds, gpu=True)

xbi = cpu.X_BI.values if hasattr(cpu, 'X_BI') else None
# overall composition per condition from conditions coords
xbi_coord = cpu.coords['X_BI'].values
T_coord = cpu.coords['T'].values

c_gm = cpu.GM.values.squeeze()   # shape (T, X_BI) likely
g_gm = gpu.GM.values.squeeze()
c_mu = cpu.MU.values.squeeze()   # (..., comp)
g_mu = gpu.MU.values.squeeze()

print("CPU GM shape", c_gm.shape, "MU shape", c_mu.shape, "dims", cpu.GM.dims)

# iterate
dims = c_gm.shape
worst = []
for it in np.ndindex(*dims):
    # squeezed array is (T, X_BI); X_BI is the last axis
    x_bi = xbi_coord[it[-1]]
    x = np.array([1.0 - x_bi, x_bi])
    cgm, ggm = c_gm[it], g_gm[it]
    cmu, gmu = c_mu[it], g_mu[it]
    c_res = cgm - np.dot(x, cmu)
    g_res = ggm - np.dot(x, gmu)
    worst.append((abs(g_res), abs(c_res), abs(cgm - ggm), it, x_bi))

worst.sort(reverse=True)
print(f"{'gpu_resid':>12} {'cpu_resid':>12} {'gm_diff':>12}  idx  x_bi")
for w in worst[:12]:
    print(f"{w[0]:12.4g} {w[1]:12.4g} {w[2]:12.4g}  {w[3]}  {w[4]:.3f}")

g_res_all = np.array([w[0] for w in worst])
c_res_all = np.array([w[1] for w in worst])
gm_diff_all = np.array([w[2] for w in worst])
print(f"\nmax GPU residual: {g_res_all.max():.4g}   max CPU residual: {c_res_all.max():.4g}")
print(f"max GM diff: {gm_diff_all.max():.4g}")
print(f"corr(gm_diff, gpu_resid): {np.corrcoef(gm_diff_all, g_res_all)[0,1]:.3f}")
