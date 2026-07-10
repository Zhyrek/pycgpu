#!/usr/bin/env python
"""Save (or compare against) a CPU-only run of the alcufe 7x7x5 grid.

Usage: python cpu_ab.py save FILE.npz | python cpu_ab.py compare FILE.npz
Env (e.g. PYCALPHAD_ROBUST_REMOVAL=1) controls solver variant per invocation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from pycalphad import Database, equilibrium, variables as v

mode, path = sys.argv[1], sys.argv[2]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dbf = Database(os.path.join(ROOT, 'Al-Cu-Fe.tdb'))
comps = ['AL', 'CU', 'FE', 'VA']
phases = sorted(dbf.phases.keys())
conds = {v.X('AL'): (0.2, 0.61, 0.4 / 6), v.X('CU'): (0.1, 0.41, 0.3 / 6),
         v.T: (700, 1400.1, 700.0 / 4), v.P: 101325, v.N: 1}

r = equilibrium(dbf, comps, phases, conds, gpu=False)
gm = r.GM.values.flatten()
ph = r.Phase.values.reshape(-1, r.Phase.shape[-1]).astype(str)
np_ = r.NP.values.reshape(-1, r.NP.shape[-1])

if mode == 'save':
    np.savez(path, gm=gm, ph=ph, np_=np_)
    print(f"saved {gm.size} conditions to {path}")
else:
    ref = np.load(path, allow_pickle=True)
    gm0, ph0, np0 = ref['gm'], ref['ph'], ref['np_']
    m = ~(np.isnan(gm) | np.isnan(gm0))
    sd = gm[m] - gm0[m]  # negative => THIS run (robust) found lower energy
    tol = 1e-4
    print(f"n={m.sum()}  robust lower: {(sd < -tol).sum()} (total {-sd[sd < -tol].sum():.4g} J/mol)  "
          f"baseline lower: {(sd > tol).sum()} (total {sd[sd > tol].sum():.4g} J/mol)")
    idx = np.where(m)[0]
    mism = 0
    for k, i in enumerate(idx):
        s0 = sorted({p for j, p in enumerate(ph0[i]) if p and np0[i, j] > 1e-6})
        s1 = sorted({p for j, p in enumerate(ph[i]) if p and np_[i, j] > 1e-6})
        if s0 != s1:
            mism += 1
            if mism <= 8:
                print(f"  cond {i}: baseline={s0} robust={s1} dGM={sd[k]:+.4g}")
    print(f"phase-set changes vs baseline: {mism}/{m.sum()}")
