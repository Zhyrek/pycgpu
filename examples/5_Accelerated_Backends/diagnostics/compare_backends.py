"""Time and cross-check every backend available on this machine.

    python compare_backends.py                  # Al-Ni binary, 100 x 100 grid
    python compare_backends.py 200 200          # nT nX
    python compare_backends.py 100 100 --threads 1,4,0

Each backend solves the grid once untimed (absorbing kernel compilation,
which happens once per system and machine), then once timed. Agreement is
reported against the reference solver.
"""
import argparse
import os

# Compile Metal kernels before solving instead of in the background, so the
# warm-up run leaves the GPU ready.
os.environ.setdefault('PYCGPU_METAL_SYNC_COMPILE', '1')
import time
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pycalphad
from pycalphad import Database, equilibrium, variables as v
from pycalphad.backend import available_backends, available_cpus
from importlib.resources import files
import pycalphad.tests.databases as dbs

parser = argparse.ArgumentParser()
parser.add_argument('nT', type=int, nargs='?', default=100)
parser.add_argument('nX', type=int, nargs='?', default=100)
parser.add_argument('--threads', default=None,
                    help='comma-separated C++ thread counts to time (0 = all cores)')
parser.add_argument('--skip-reference', action='store_true')
args = parser.parse_args()

dbf = Database(str(files(dbs).joinpath('alni_dupin_2001.tdb')))
comps, phases = ['AL', 'NI', 'VA'], sorted(dbf.phases)
conds = {v.N: 1, v.P: 101325, v.T: np.linspace(700, 1900, args.nT),
         v.X('AL'): np.linspace(0.02, 0.98, args.nX)}
n = args.nT * args.nX

runs = []
avail = available_backends()
if not args.skip_reference:
    runs.append(('default', {}))
if avail['cpp'] is None:
    for t in (args.threads.split(',') if args.threads else [None]):
        runs.append(('cpp', {} if t is None else {'threads': int(t)}))
for name in ('cuda', 'metal'):
    if avail[name] is None:
        runs.append((name, {}))

print(f'Al-Ni, {len(phases)} phases, {n} conditions; '
      f'{available_cpus()} cores')
ref = None
for name, opts in runs:
    with pycalphad.backend(name, **opts):
        if name != 'default':
            equilibrium(dbf, comps, phases, conds)  # compile / warm caches
        t0 = time.perf_counter()
        eq = equilibrium(dbf, comps, phases, conds)
        dt = time.perf_counter() - t0
    gm = eq.GM.values
    if ref is None:
        ref = gm
    both = np.isfinite(gm) & np.isfinite(ref)
    dgm = np.abs(gm - ref)[both]
    label = name + (f" threads={opts['threads']}" if 'threads' in opts else '')
    print(f'{label:20s} {dt:8.2f} s  {n / dt:9.0f} cond/s   '
          f'failed {int((~np.isfinite(gm)).sum()):4d}   '
          f'|dGM| p99 {np.percentile(dgm, 99) if dgm.size else 0:.1e}  max {dgm.max() if dgm.size else 0:.1e} J/mol')
