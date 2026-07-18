"""Backend speed benchmark: the AuBi example grid, compile time excluded.

    python benchmark_speed.py                # c++ backend, 260 x 250 grid
    python benchmark_speed.py gpu            # gpu backend
    python benchmark_speed.py c++ 50 50      # smaller grid (nx nt)

Reading the numbers:
  - The FIRST call on a new system compiles its kernel (15-35 s,
    disk-cached afterwards). This script warms the cache with a 1-point
    solve so the timed runs measure the solver, not the compiler. If you
    time your own runs, exclude the first call the same way.
  - Reference machine (dev box, single core): 260x250 grid = 65k
    equilibria runs ~55-60 s reference vs ~7-8 s accelerated (~8x); a
    small 20x20 grid still holds ~6x. If you see far less, run with
    PYCGPU_TIME=1 to print the kernel wall time separately — the gap
    between it and the total is shared Python overhead (grid sampling,
    starting point, result processing), which dominates tiny problems.
"""
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pycalphad
from pycalphad import Database, equilibrium, variables as v

HERE = os.path.dirname(os.path.abspath(__file__))
DBPATH = os.path.join(HERE, '..', '..', 'databases', 'AuBi-07Wan.tdb')

backend = sys.argv[1] if len(sys.argv) > 1 else 'c++'
nx = int(sys.argv[2]) if len(sys.argv) > 2 else 260
nt = int(sys.argv[3]) if len(sys.argv) > 3 else 250

dbf = Database(DBPATH)
comps = ['AU', 'BI', 'VA']
phases = sorted(dbf.phases.keys())
conditions = {v.N: 1, v.P: 101325,
              v.X('BI'): (0.02, 0.981, 0.96 / (nx - 1)),
              v.T: (450, 1000.1, 550.0 / (nt - 1))}
print(f'{nx} x {nt} = {nx * nt} equilibria, backend {backend!r}')

print('warming kernel cache (compile happens once per system)...')
t0 = time.perf_counter()
with pycalphad.backend(backend):
    equilibrium(dbf, comps, phases, {v.N: 1, v.P: 101325, v.T: 700, v.X('BI'): 0.5})
print(f'  warm-up (incl. any compile): {time.perf_counter() - t0:.1f} s')

t0 = time.perf_counter()
ref = equilibrium(dbf, comps, phases, conditions)
t_ref = time.perf_counter() - t0
print(f'reference solver   : {t_ref:.1f} s')

with pycalphad.backend(backend):
    t0 = time.perf_counter()
    acc = equilibrium(dbf, comps, phases, conditions)
    t_acc = time.perf_counter() - t0
print(f'accelerated backend: {t_acc:.1f} s')
print(f'speedup            : {t_ref / t_acc:.1f}x')

gm_r = np.asarray(ref.GM.values, dtype=float).reshape(-1)
gm_a = np.asarray(acc.GM.values, dtype=float).reshape(-1)
both = np.isfinite(gm_r) & np.isfinite(gm_a)
dgm = float(np.abs(gm_a[both] - gm_r[both]).max())
print(f'max|dGM|           : {dgm:.3e} J '
      f'{"(RED FLAG: exact zero = silent fallback)" if dgm == 0.0 else "(eps-class expected)"}')
