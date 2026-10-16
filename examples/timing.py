"""
pycalphad's Workspace against CGWorkspace on a few systems: wall time (warm, best of two) and the largest Gibbs
energy difference. The first call of each compiles the phases' functions and is not timed.
"""
import time
from importlib.resources import files

import numpy as np
from pycalphad import Database, Workspace, variables as v
from pycalphad.core.utils import filter_phases, unpack_species

from pycalphad_cg import CGWorkspace

DB = files('pycalphad.tests.databases')
COST_MINOR = {'MG': 0.03, 'CU': 0.02, 'ZN': 0.02, 'FE': 0.01, 'MN': 0.01, 'NI': 0.01, 'CR': 0.01, 'TI': 0.001}
CASES = [
    ('Al-Ni, 1200 K, 500 compositions', 'alni_dupin_2001.tdb', ['AL', 'NI', 'VA'],
     {v.T: 1200, v.X('AL'): (0.02, 0.98, 0.96 / 500)}),
    ('Al-Fe (Sundman), 900 K, 500 compositions', 'Al-Fe_sundman2009.tdb', ['AL', 'FE', 'VA'],
     {v.T: 900, v.X('AL'): (0.02, 0.6, 0.58 / 500)}),
    ('Al-Fe, 15 T x 25 compositions', 'alfe.tdb', ['AL', 'FE', 'VA'],
     {v.T: (500, 1901, 100), v.X('AL'): (0.02, 0.99, 0.04)}),
    ('Al-Cr-Ni, 1273 K, 25 x 25 compositions', 'alcrni.tdb', ['AL', 'CR', 'NI', 'VA'],
     {v.T: 1273, v.X('AL'): (0.02, 0.5, 0.02), v.X('CR'): (0.02, 0.5, 0.02)}),
    ('COST507 Al-Mg-Si-Cu-Zn, 800 K, 100 compositions', 'COST507.tdb', ['AL', 'MG', 'SI', 'CU', 'ZN', 'VA'],
     {v.T: 800, v.X('MG'): 0.03, v.X('CU'): 0.02, v.X('ZN'): 0.02, v.X('SI'): (0.01, 0.1, 0.0009)}),
    ('COST507 10 elements, 800 K, 100 compositions', 'COST507.tdb',
     ['AL', 'MG', 'SI', 'CU', 'ZN', 'FE', 'MN', 'NI', 'CR', 'TI', 'VA'],
     {v.T: 800, **{v.X(e): x for e, x in COST_MINOR.items()}, v.X('SI'): (0.01, 0.1, 0.0009)}),
]


def timed(cls, *args, **kwargs):
    cls(*args, **kwargs).eq
    best, gm = np.inf, None
    for _ in range(2):
        start = time.perf_counter()
        gm = np.asarray(cls(*args, **kwargs).get('GM'))
        best = min(best, time.perf_counter() - start)
    return best, gm


print(f'{"system":50s} {"pycalphad":>10s} {"cg, 1 thr":>10s} {"cg, all":>10s}  {"max |dGM|":>10s} {"cg lower by":>12s}')
for label, db, comps, conds in CASES:
    dbf = Database(str(DB / db))
    phases = filter_phases(dbf, unpack_species(dbf, comps))
    conds = {v.P: 101325, v.N: 1, **conds}
    t_ref, gm_ref = timed(Workspace, dbf, comps, phases, conds)
    t_one, _ = timed(CGWorkspace, dbf, comps, phases, conds, cg_threads=1)
    t_all, gm_cg = timed(CGWorkspace, dbf, comps, phases, conds)
    d = gm_cg - gm_ref
    print(f'{label:50s} {t_ref:9.2f}s {t_one:9.2f}s {t_all:9.2f}s  {np.nanmax(np.abs(d)):10.2g} {max(0.0, -np.nanmin(d)):12.2g}',
          flush=True)
