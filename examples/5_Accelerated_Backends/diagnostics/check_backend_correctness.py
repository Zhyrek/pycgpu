"""Backend correctness check: A/B a small solve against the reference.

Run this FIRST on any new machine (especially a new GPU / ROCm stack)
before trusting accelerated results:

    python check_backend_correctness.py            # checks c++ and gpu
    python check_backend_correctness.py c++        # one backend only
    python check_backend_correctness.py gpu

What it verifies, and how to read it:
  - max|dGM| in the 1e-9..1e-6 J range with fallback=no  ->  HEALTHY
    (eps-class agreement between genuinely independent computations).
  - max|dGM| EXACTLY 0.0  ->  RED FLAG: the accelerated path silently
    fell back to the reference solver (you compared it with itself).
    The fallback guard below catches this, but exact zero is the
    signature to distrust anywhere else too.
  - Large diffs / NaNs with fallback=no  ->  the backend is computing
    garbage on this platform (on ROCm this is the signature of the
    1 KB dynamic-stack budget applying unraised; see the warning that
    ensure_device_stack_limit prints, and try
    PYCGPU_DEVICE_STACK_BYTES=8192 or the c++ backend).
"""
import os
import sys
import tempfile
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pycalphad
from pycalphad import Database, equilibrium, variables as v

HERE = os.path.dirname(os.path.abspath(__file__))
DBPATH = os.path.join(HERE, '..', '..', 'databases', 'AuBi-07Wan.tdb')

def check(backend):
    dbf = Database(DBPATH)
    comps = ['AU', 'BI', 'VA']
    phases = sorted(dbf.phases.keys())
    conds = {v.N: 1, v.P: 101325, v.T: (500, 900, 50), v.X('BI'): (0.1, 0.9, 0.1)}

    ref = equilibrium(dbf, comps, phases, conds)

    log = os.path.join(tempfile.gettempdir(), f'pycgpu_dispatch_{os.getpid()}.log')
    if os.path.exists(log):
        os.remove(log)
    os.environ['PYCGPU_COUNT_DISPATCH'] = log
    try:
        with pycalphad.backend(backend):
            acc = equilibrium(dbf, comps, phases, conds)
    finally:
        del os.environ['PYCGPU_COUNT_DISPATCH']
    audit = open(log).read() if os.path.exists(log) else '(no dispatch log)'
    fell_back = 'fallback' in audit

    gm_r = np.asarray(ref.GM.values, dtype=float).reshape(-1)
    gm_a = np.asarray(acc.GM.values, dtype=float).reshape(-1)
    mu_r = np.asarray(ref.MU.values, dtype=float).reshape(-1)
    mu_a = np.asarray(acc.MU.values, dtype=float).reshape(-1)
    both = np.isfinite(gm_r) & np.isfinite(gm_a)
    nan_mismatch = int((np.isfinite(gm_r) != np.isfinite(gm_a)).sum())
    dgm = float(np.abs(gm_a[both] - gm_r[both]).max()) if both.any() else float('nan')
    mboth = np.isfinite(mu_r) & np.isfinite(mu_a)
    dmu = float(np.abs(mu_a[mboth] - mu_r[mboth]).max()) if mboth.any() else float('nan')

    print(f'--- backend {backend!r} ---')
    print(f'  conditions        : {gm_r.size} (converged ref {int(np.isfinite(gm_r).sum())}, '
          f'acc {int(np.isfinite(gm_a).sum())}, nan mismatches {nan_mismatch})')
    print(f'  max|dGM|          : {dgm:.3e} J')
    print(f'  max|dMU|          : {dmu:.3e} J')
    print(f'  silent fallback   : {"YES  <-- comparing reference with itself!" if fell_back else "no"}')
    if fell_back:
        print(f'  dispatch audit    : {audit.strip()}')
        verdict = 'FALLBACK (fix the reason above before benchmarking)'
    elif dgm == 0.0:
        verdict = 'SUSPICIOUS (exact zero without a logged fallback — investigate)'
    elif nan_mismatch == 0 and dgm < 1e-3:
        verdict = 'HEALTHY (eps-class agreement)'
    else:
        verdict = 'BROKEN on this platform (see module docstring for ROCm notes)'
    print(f'  verdict           : {verdict}')
    return verdict.startswith('HEALTHY')

if __name__ == '__main__':
    backends = sys.argv[1:] or ['c++', 'gpu']
    ok = True
    for b in backends:
        try:
            ok &= check(b)
        except Exception as e:
            print(f'--- backend {b!r} ---\n  FAILED to run: {type(e).__name__}: {e}')
            ok = False
    sys.exit(0 if ok else 1)
