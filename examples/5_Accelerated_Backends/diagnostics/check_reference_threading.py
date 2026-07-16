"""Is the REFERENCE solver actually using multiple cores on this machine?

    python check_reference_threading.py            # ~1-2 min, default grid
    python check_reference_threading.py 10 24      # smaller grid (nx nt)

Motivation: when comparing backend speedups across machines, a fast
reference leg can look like "the reference is parallelized here". This
measures it directly: the same many-phase reference solve (AlCuFe, where
threaded BLAS would matter most if anywhere) is timed in two child
processes — one with the environment as-is, one with every BLAS/OpenMP
thread-pool variable pinned to 1. Child processes are used because thread
pools are sized when the libraries LOAD; pinning after import is
unreliable without extra dependencies.

Reading the result:
  ratio ~1.0  ->  the reference is effectively SINGLE-THREADED here (its
                  per-point matrices are too small for BLAS threading to
                  engage). Machine-to-machine reference differences are
                  core speed, not core count. Nothing to change.
  ratio < 1   ->  thread pools actively HURT the reference on this machine
                  (spawn/sync overhead on tiny LAPACK calls — measured 2.1x
                  slower unpinned on a 22-core dev box). Pin the pools for
                  your reference runs.
  ratio >>1   ->  threaded BLAS is genuinely accelerating the reference on
                  this machine; compare backends accordingly (e.g. give the
                  c++ backend PYCGPU_CPU_THREADS for a fair fight).
"""
import os
import subprocess
import sys

PIN_VARS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
            'BLIS_NUM_THREADS')

CHILD = '--child'

def child(nx, nt):
    import warnings
    warnings.filterwarnings('ignore')
    import time
    from pycalphad import Database, equilibrium, variables as v
    here = os.path.dirname(os.path.abspath(__file__))
    dbf = Database(os.path.join(here, '..', '..', 'databases', 'Al-Cu-Fe.tdb'))
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = sorted(dbf.phases.keys())
    conditions = {v.N: 1, v.P: 101325,
                  v.X('AL'): (0.05, 0.85, 0.8 / (nx - 1)),
                  v.X('CU'): (0.05, 0.85, 0.8 / (nx - 1)),
                  v.T: (600, 1400.1, 800.0 / (nt - 1))}
    # small warm-up solve so one-time setup (imports, codegen caches on the
    # reference path) stays out of the timing
    equilibrium(dbf, comps, phases, {v.N: 1, v.P: 101325, v.T: 1000,
                                     v.X('AL'): 0.2, v.X('CU'): 0.2})
    t0 = time.perf_counter()
    equilibrium(dbf, comps, phases, conditions)
    print(f'CHILD_SECONDS={time.perf_counter() - t0:.3f}')

def run_case(label, env, nx, nt):
    r = subprocess.run([sys.executable, os.path.abspath(__file__), CHILD,
                        str(nx), str(nt)],
                       env=env, capture_output=True, text=True)
    secs = None
    for line in r.stdout.splitlines():
        if line.startswith('CHILD_SECONDS='):
            secs = float(line.split('=')[1])
    if secs is None:
        print(f'{label}: FAILED\n{r.stderr[-500:]}')
        sys.exit(1)
    print(f'{label}: {secs:.1f} s')
    return secs

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == CHILD:
        child(int(sys.argv[2]), int(sys.argv[3]))
        sys.exit(0)
    nx = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    nt = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    print(f'reference AlCuFe solve, {nx}x{nx}x{nt} = {nx*nx*nt} conditions, '
          f'{os.cpu_count()} cpus visible')
    env_as_is = dict(os.environ)
    env_pinned = dict(os.environ)
    for var in PIN_VARS:
        env_pinned[var] = '1'
    t_free = run_case('threads as-is    ', env_as_is, nx, nt)
    t_pin = run_case('all pools pinned=1', env_pinned, nx, nt)
    ratio = t_pin / t_free
    print(f'ratio (pinned / as-is): {ratio:.2f}')
    if ratio < 0.87:
        print(f'verdict: thread pools actively HURT the reference here '
              f'({1/ratio:.1f}x slower unpinned — spawn/sync overhead on '
              f'tiny per-point LAPACK calls). Pin the pools '
              f'(OPENBLAS_NUM_THREADS=1 etc.) for reference runs.')
    elif ratio < 1.15:
        print('verdict: reference is effectively SINGLE-THREADED here — '
              'machine differences are core speed, not core count.')
    else:
        print(f'verdict: threaded pools speed the reference up ~{ratio:.1f}x '
              'on this machine; account for that when comparing backends '
              '(e.g. set PYCGPU_CPU_THREADS for the c++ backend).')
