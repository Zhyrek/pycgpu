#!/usr/bin/env python
"""NbTi CPU-vs-GPU comparison sweep (third system for correctness validation)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from pycalphad import Database, equilibrium, variables as v

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dbf = Database(os.path.join(ROOT, 'NbTi.tdb'))
comps = ['NB', 'TI', 'VA']
phases = sorted(dbf.phases.keys())
print(f"NbTi phases: {phases}")
conds = {v.X('TI'): (0.05, 0.951, 0.09), v.T: (800, 2400.1, 200.0),
         v.P: 101325, v.N: 1}

t0 = time.time()
cpu = equilibrium(dbf, comps, phases, conds, gpu=False)
t_cpu = time.time() - t0
t0 = time.time()
gpu = equilibrium(dbf, comps, phases, conds, gpu=True)
t_gpu = time.time() - t0

c = cpu.GM.values.flatten()
g = gpu.GM.values.flatten()
m = ~(np.isnan(c) | np.isnan(g))
d = np.abs(c[m] - g[m])
print(f"n={m.sum()}  CPU {t_cpu:.2f}s  GPU {t_gpu:.2f}s")
print(f"GM max diff={d.max():.6g} mean={d.mean():.6g}")

cph = cpu.Phase.values.reshape(-1, cpu.Phase.shape[-1])
gph = gpu.Phase.values.reshape(-1, gpu.Phase.shape[-1])
cnp = cpu.NP.values.reshape(-1, cpu.NP.shape[-1])
gnp = gpu.NP.values.reshape(-1, gpu.NP.shape[-1])
mis = 0
for i in range(cnp.shape[0]):
    if np.isnan(c[i]) or np.isnan(g[i]):
        continue
    cs = sorted({cph[i, j] for j in range(cnp.shape[1]) if cnp[i, j] > 1e-6 and cph[i, j]})
    gs = sorted({gph[i, j] for j in range(gnp.shape[1]) if gnp[i, j] > 1e-6 and gph[i, j]})
    if cs != gs:
        mis += 1
        if mis <= 8:
            print(f"  cond {i}: CPU={cs} GPU={gs} dGM={abs(c[i]-g[i]):.4g}")
print(f"Stable-phase sets: {m.sum()-mis}/{m.sum()} match")
