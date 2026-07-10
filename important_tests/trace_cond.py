#!/usr/bin/env python
"""Trace single AlCuFe condition: CPU debug vs GPU verbose.

Usage: python trace_cond.py <T> <X_AL> <X_CU>
Runs CPU then GPU with verbose=True so debug_output writes CPU_VS_GPU_TRACE.txt.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from pycalphad import Database, equilibrium, variables as v

T, xal, xcu = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dbf = Database(os.path.join(ROOT, 'Al-Cu-Fe.tdb'))
comps = ['AL', 'CU', 'FE', 'VA']
phases = sorted(dbf.phases.keys())
conds = {v.X('AL'): xal, v.X('CU'): xcu, v.T: T, v.P: 101325, v.N: 1}

for mode, use_gpu in [('CPU', False), ('GPU', True)]:
    r = equilibrium(dbf, comps, phases, conds, gpu=use_gpu, verbose=True)
    gm = float(r.GM.values.squeeze())
    ph = r.Phase.values.squeeze()
    npv = r.NP.values.squeeze()
    yv = r.Y.values.squeeze()
    print(f"=== {mode} RESULT ===")
    print(f"GM = {gm:.6f}")
    for j in range(len(ph)):
        if isinstance(ph[j], str) and ph[j]:
            ys = ' '.join(f"{x:.9f}" for x in np.atleast_1d(yv[j]) if not np.isnan(x))
            print(f"  phase {ph[j]:12s} NP={npv[j]:.9f} Y=[{ys}]")
