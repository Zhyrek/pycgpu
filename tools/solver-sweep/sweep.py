"""Equilibria over fixed condition grids for comparing solvers.
usage: sweep.py LABEL OUT.npz [cg]   (run with the checkout under test on PYTHONPATH; 'cg' selects the cg backend)"""
import sys, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
from pycalphad import Database, Workspace, variables as v
import pycalphad.tests.databases as tdbs
import os
D = os.path.dirname(tdbs.__file__) + '/'
g = lambda a, b, s: np.round(np.arange(a, b, s), 3)
CASES = {
    'alfe_sundman': ('Al-Fe_sundman2009.tdb', ['AL', 'FE'], [700, 800, 900, 1000, 1100, 1200, 1300], {'FE': g(0.5, 0.96, 0.025)}, {}),
    'alfe': ('alfe.tdb', ['AL', 'FE'], [600, 800, 1000, 1200, 1400], {'FE': g(0.03, 0.98, 0.04)}, {}),
    'alni_dupin': ('alni_dupin_2001.tdb', ['AL', 'NI'], [600, 900, 1200, 1500], {'NI': g(0.03, 0.98, 0.04)}, {}),
    'cov': ('CoV-20Wan.tdb', ['CO', 'V'], [800, 1000, 1200, 1400], {'V': g(0.03, 0.98, 0.04)}, {}),
    'mnti': ('MnTi-23Wal-3g.tdb', ['MN', 'TI'], [900, 1200, 1500], {'TI': g(0.03, 0.98, 0.04)}, {}),
    'cuy': ('Al-Cu-Y.tdb', ['CU', 'Y'], [900, 1200], {'Y': g(0.03, 0.98, 0.04)}, {}),
    'crti': ('crtiv_ghosh.tdb', ['CR', 'TI'], [900, 1200, 1500], {'TI': g(0.03, 0.98, 0.04)}, {}),
    'nbre': ('nbre_liu.tdb', ['NB', 'RE'], [1200, 1800], {'RE': g(0.03, 0.98, 0.04)}, {}),
    'alcrni1550': ('alcrni.tdb', ['AL', 'CR', 'NI'], [1550], {'AL': g(0.05, 0.9, 0.1), 'CR': g(0.05, 0.9, 0.1)}, {}),
    'alcrni1200': ('alcrni.tdb', ['AL', 'CR', 'NI'], [1200], {'AL': g(0.05, 0.9, 0.1), 'CR': g(0.05, 0.9, 0.1)}, {}),
    'alcocrni1300': ('alcocrni.tdb', ['AL', 'CR', 'NI'], [1300], {'AL': g(0.05, 0.9, 0.1), 'CR': g(0.05, 0.9, 0.1)}, {}),
    'alcocrni1300_fine': ('alcocrni.tdb', ['AL', 'CR', 'NI'], [1300], {'AL': g(0.02, 0.5, 0.04), 'CR': g(0.02, 0.5, 0.04)}, {}),
    'crfenb': ('CrFeNb_Jacob2016.tdb', ['CR', 'FE', 'NB'], [1373], {'CR': g(0.05, 0.9, 0.1), 'NB': g(0.05, 0.9, 0.1)}, {}),
    'alcumg': ('COST507.tdb', ['AL', 'CU', 'MG'], [700], {'CU': g(0.05, 0.9, 0.1), 'MG': g(0.05, 0.9, 0.1)}, {}),
    'cost5': ('COST507.tdb', ['AL', 'CU', 'MG', 'SI', 'ZN'], [500], {'CU': g(0.02, 0.2, 0.03), 'MG': g(0.01, 0.1, 0.03)}, {'SI': 0.021, 'ZN': 0.052}),
    'cost10': ('COST507.tdb', ['AL', 'MG', 'SI', 'CU', 'ZN', 'FE', 'MN', 'NI', 'CR', 'TI'], [750],
               {'CU': np.array([0.005, 0.03, 0.06]), 'MG': np.array([0.005, 0.03, 0.06]), 'SI': np.array([0.005, 0.03, 0.06]), 'ZN': np.array([0.005, 0.03, 0.06])},
               {'FE': 0.005, 'MN': 0.005, 'NI': 0.005, 'CR': 0.005, 'TI': 0.001}),
}
CASES2 = {
    'cost5_T': ('COST507.tdb', ['AL', 'CU', 'MG', 'SI', 'ZN'], [450, 500, 600, 700], {'CU': g(0.01, 0.2, 0.03), 'MG': g(0.01, 0.12, 0.02)}, {'SI': 0.021, 'ZN': 0.052}),
    'cost5_sizn': ('COST507.tdb', ['AL', 'CU', 'MG', 'SI', 'ZN'], [500], {'SI': g(0.005, 0.1, 0.015), 'ZN': g(0.01, 0.1, 0.015)}, {'CU': 0.04, 'MG': 0.02}),
    'cost10_T': ('COST507.tdb', ['AL', 'MG', 'SI', 'CU', 'ZN', 'FE', 'MN', 'NI', 'CR', 'TI'], [650, 850],
                 {'CU': np.array([0.005, 0.03, 0.06]), 'MG': np.array([0.005, 0.03, 0.06]), 'SI': np.array([0.005, 0.03, 0.06]), 'ZN': np.array([0.005, 0.03, 0.06])},
                 {'FE': 0.005, 'MN': 0.005, 'NI': 0.005, 'CR': 0.005, 'TI': 0.001}),
    'alcocrni4': ('alcocrni.tdb', ['AL', 'CO', 'CR', 'NI'], [1100, 1300, 1500], {'AL': g(0.05, 0.3, 0.05), 'CO': g(0.05, 0.3, 0.05), 'CR': g(0.05, 0.3, 0.05)}, {}),
}
import os as _os
if _os.environ.get('SWEEP_SET') == '2':
    CASES = CASES2
label, out = sys.argv[1], sys.argv[2]
use_cg = len(sys.argv) > 3 and sys.argv[3] == 'cg'
if use_cg:
    import pycalphad
    pycalphad.set_backend('cg')
res = {}
for name, (f, els, Ts, axes, fixed) in CASES.items():
    dbf = Database(D + f)
    conds = {v.T: np.array(Ts, float), v.P: 101325, v.N: 1, **{v.X(k): val for k, val in axes.items()}, **{v.X(k): val for k, val in fixed.items()}}
    t0 = time.time()
    eq = Workspace(dbf, els + ['VA'], list(dbf.phases.keys()), conds).eq
    dt = time.time() - t0
    gm = np.asarray(eq.GM, float).reshape(-1)
    phases = np.asarray(eq.Phase).reshape(gm.size, -1)
    res[name + '/GM'] = gm
    res[name + '/Phase'] = np.array(['+'.join(sorted(str(p) for p in row if str(p))) for row in phases])
    res[name + '/time'] = dt
    print(f'{label} {name}: {gm.size} conditions, {int(np.isnan(gm).sum())} NaN, {dt:.1f} s', flush=True)
np.savez(out, **res)
