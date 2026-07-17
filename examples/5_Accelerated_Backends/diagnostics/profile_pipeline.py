"""Pipeline profiler: where does the time go on THIS machine?

Runs a sample system on an accelerated backend under a wall-clock budget
and reports the stage breakdown — grid energy sampling, starting-point
hull, solver/kernel, input packing, result processing, and the remaining
python "glue". Works for both backends; on the gpu backend the kernel
wall (device time between launch and synchronize) is reported separately
from the host-side stages, which is the number that tells you whether
your card or the host pipeline bounds throughput.

    python profile_pipeline.py binary                # c++ (or gpu if PYCGPU choice)
    python profile_pipeline.py ternary --backend gpu
    python profile_pipeline.py quaternary --backend c++ --budget 600
    python profile_pipeline.py binary --backend c++ --solver-internals

How to read it:
  - "kernel/solve" dominating on gpu  ->  the device bounds you. On
    consumer NVIDIA cards this is usually the 1/32-1/64 FP64 rate, not a
    software problem; datacenter cards (A100/H100/MI2xx/MI3xx) shrink
    this stage by an order of magnitude.
  - host stages dominating on gpu  ->  batch is too small to amortize
    fixed costs, or you found a host-side scaling bug worth reporting.
  - grid dominating  ->  the energy sampling (pdens x phases) is the
    cost; it grows steeply with component count.
  - --solver-internals (c++ and gpu) adds the in-kernel segment shares
    (energy/gradient/Hessian evaluation, constraint-matrix inversion,
    least-squares) from an instrumented run on a bounded grid. It
    recompiles the kernel with cycle counters and prints one line per
    condition internally, so it runs a SMALL fixed grid regardless of
    the budget.

The budget sizes the grid: a small warm probe measures per-condition
cost, then the main run is scaled to fill the budget (within a cap).
First-time kernel compilation (up to a few minutes for many-phase
systems) is excluded — the probe run absorbs it.
"""
import argparse
import collections
import contextlib
import io
import os
import re
import subprocess
import sys
import time
import warnings

warnings.filterwarnings('ignore')
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DB_DIR = os.path.normpath(os.path.join(_HERE, '..', '..', 'databases'))

# (database, components, T-range, [(element, lo, hi)], base X-axis points)
SYSTEMS = {
    'binary': (os.path.join(_DB_DIR, 'AuBi-07Wan.tdb'), ['AU', 'BI', 'VA'],
               (400, 1000), [('BI', 0.005, 0.995)]),
    'ternary': (os.path.join(_DB_DIR, 'Al-Cu-Fe.tdb'), ['AL', 'CU', 'FE', 'VA'],
                (700, 1400), [('CU', 0.02, 0.7), ('FE', 0.02, 0.7)]),
    'quaternary': (None, ['AL', 'CO', 'CR', 'NI', 'VA'],  # resolved below
                   (600, 1900), [('AL', 0.05, 0.30), ('CO', 0.05, 0.30),
                                 ('CR', 0.05, 0.30)]),
}


def _quaternary_db():
    # alcocrni ships with the pycalphad test data (present in both a repo
    # checkout and an installed package).
    import pycalphad.tests
    return os.path.join(os.path.dirname(pycalphad.tests.__file__),
                        'databases', 'alcocrni.tdb')


def _build_conds(system, n_t, n_x):
    from pycalphad import variables as v
    _, _, (t_lo, t_hi), x_axes = SYSTEMS[system]
    conds = {v.N: 1, v.P: 101325, v.T: np.linspace(t_lo, t_hi, n_t)}
    for el, lo, hi in x_axes:
        conds[v.X(el)] = np.linspace(lo, hi, n_x)
    return conds, n_t * n_x ** len(x_axes)


def _profiled_run(dbf, comps, phases, conds, backend):
    """One equilibrium() call with stage timers + kernel-wall capture."""
    import pycalphad
    from pycalphad import equilibrium
    import pycalphad.gpu.point_solver as ps
    import pycalphad.gpu.parallel_calculate as pc
    import pycalphad.gpu.cpu_backend as cb
    import pycalphad.gpu.gpu_equilibrium as ge
    import pycalphad.gpu.gpu_systemspec_array as gs

    timers = collections.OrderedDict()
    saved = []

    def wrap(mod, name, label):
        orig = getattr(mod, name)
        saved.append((mod, name, orig))

        def timed(*a, **k):
            t0 = time.time()
            out = orig(*a, **k)
            timers[label] = timers.get(label, 0.0) + (time.time() - t0)
            return out
        setattr(mod, name, timed)

    wrap(pc, 'parallel_calculate', 'grid energy sampling')
    wrap(ps, 'device_point_hull', 'starting-point hull')
    wrap(cb, 'run_cpu_backend', 'solver (c++ driver)')
    wrap(gs, 'create_system_specifications_array', 'input packing (spec)')
    wrap(ge, '_create_initial_phase_data_struct_array', 'input packing (ipd)')
    wrap(ge, '_process_gpu_results', 'result processing')

    os.environ['PYCGPU_TIME'] = '1'
    buf = io.StringIO()
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(buf):
            eq = equilibrium(dbf, comps, phases, conds, backend=backend)
    finally:
        os.environ.pop('PYCGPU_TIME', None)
        for mod, name, orig in saved:
            setattr(mod, name, orig)
    total = time.time() - t0

    m = re.findall(r'kernel wall: ([\d.]+) s', buf.getvalue())
    if m:  # gpu path: device time between launch and synchronize
        timers['solver (gpu kernel wall)'] = sum(float(x) for x in m)
    gm = np.asarray(eq.GM.values, dtype=float).ravel()
    return total, timers, gm


def _solver_internals(system, backend):
    """In-kernel segment shares from a PYCGPU_PROF-instrumented subprocess.

    Recompiles with cycle counters (separate cache entry) and parses the
    per-condition [PROF] lines. Fixed small grid: the instrumentation
    prints one line per condition.
    """
    n_t, n_x = 4, {1: 64, 2: 8, 3: 4}[len(SYSTEMS[system][3])]
    code = (
        "import warnings; warnings.filterwarnings('ignore')\n"
        "import numpy as np\n"
        "import sys; sys.path.insert(0, %r)\n"
        "from profile_pipeline import SYSTEMS, _build_conds, _quaternary_db\n"
        "from pycalphad import Database, equilibrium\n"
        "db, comps, _, _ = SYSTEMS[%r]\n"
        "db = db or _quaternary_db()\n"
        "dbf = Database(db)\n"
        "conds, _ = _build_conds(%r, %d, %d)\n"
        "equilibrium(dbf, comps, sorted(dbf.phases.keys()), conds, backend=%r)\n"
    ) % (_HERE, system, system, n_t, n_x, backend)
    env = dict(os.environ, PYCGPU_PROF='1', PYTHONHASHSEED='0')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, env=env).stdout
    tot = collections.Counter()
    n = 0
    for line in out.splitlines():
        if not line.startswith('[PROF]'):
            continue
        n += 1
        for k, val in re.findall(r'(\w+)=([\d.\-]+)', line):
            if k in ('iters', 'solve', 'recompute', 'fill', 'lstsq',
                     'hess', 'inv', 'funcs'):
                tot[k] += float(val)
    if not n:
        print('  (no [PROF] output — instrumented compile may have failed)')
        return
    s = max(tot['solve'], 1e-9)
    print(f'  conditions instrumented : {n} (mean {tot["iters"]/n:.1f} iterations)')
    print(f'  energy/grad evaluation  : {100*tot["funcs"]/s:5.1f}% of solve')
    print(f'  Hessian evaluation      : {100*tot["hess"]/s:5.1f}%')
    print(f'  constraint-matrix inv   : {100*tot["inv"]/s:5.1f}%')
    print(f'  least-squares (dgelsd)  : {100*tot["lstsq"]/s:5.1f}%')
    print(f'  matrix fill             : {100*tot["fill"]/s:5.1f}%')
    other = 100 * (1 - (tot['recompute'] + tot['lstsq'] + tot['fill']) / s)
    print(f'  other in-loop           : {other:5.1f}%')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('system', choices=sorted(SYSTEMS))
    ap.add_argument('--backend', default='c++', choices=['c++', 'gpu'])
    ap.add_argument('--budget', type=float, default=60.0,
                    help='target wall seconds for the main run (default 60)')
    ap.add_argument('--threads', type=int, default=None,
                    help="PYCGPU_CPU_THREADS for the c++ backend")
    ap.add_argument('--solver-internals', action='store_true',
                    help='also report in-kernel segment shares (recompiles)')
    args = ap.parse_args()

    if args.threads:
        os.environ['PYCGPU_CPU_THREADS'] = str(args.threads)

    from pycalphad import Database
    db, comps, _, x_axes = SYSTEMS[args.system]
    db = db or _quaternary_db()
    dbf = Database(db)
    phases = sorted(dbf.phases.keys())
    ndim = len(x_axes)

    # Probe 1: absorbs one-time kernel compilation and warms every cache.
    probe_nx = {1: 50, 2: 8, 3: 4}[ndim]
    conds, n_probe = _build_conds(args.system, 4, probe_nx)
    print(f'[probe] {n_probe} conditions (first run may include kernel compilation)...')
    _profiled_run(dbf, comps, phases, conds, args.backend)      # compile + warm caches
    t_probe, _, _ = _profiled_run(dbf, comps, phases, conds, args.backend)
    per_cond = t_probe / n_probe
    print(f'[probe] warm: {t_probe:.2f}s -> {1000*per_cond:.3f} ms/condition')

    # Probe 2: small probes are fixed-overhead dominated and overestimate the
    # per-condition rate badly (20x observed on gpu binaries) — re-measure at
    # ~10% of the budget before sizing the main run.
    n2_target = max(n_probe * 2, int(0.1 * args.budget / per_cond))
    n2_x = max(2, int(round((n2_target / 4) ** (1.0 / ndim))))
    conds, n2 = _build_conds(args.system, 4, n2_x)
    t2, _, _ = _profiled_run(dbf, comps, phases, conds, args.backend)
    per_cond = t2 / n2
    print(f'[probe] at {n2} conditions: {t2:.2f}s -> {1000*per_cond:.3f} ms/condition')

    # Size the main grid to the budget (same shape family, capped).
    n_target = min(int(args.budget / per_cond), 2_000_000)
    n_t = 12
    n_x = max(2, int(round((n_target / n_t) ** (1.0 / ndim))))
    conds, n = _build_conds(args.system, n_t, n_x)
    print(f'[main]  {n} conditions (~{n * per_cond:.0f}s at probe rate, '
          f'budget {args.budget:.0f}s)')

    total, timers, gm = _profiled_run(dbf, comps, phases, conds, args.backend)
    finite = int(np.isfinite(gm).sum())

    print(f'\n=== {args.system} / {args.backend} — {n} conditions, '
          f'{total:.1f}s wall ({1000*total/n:.3f} ms/condition), '
          f'{finite}/{n} converged ===')
    for label, secs in sorted(timers.items(), key=lambda kv: -kv[1]):
        print(f'  {label:28s} {secs:8.2f}s  ({100*secs/total:5.1f}%)')
    glue = max(total - sum(timers.values()), 0.0)
    print(f'  {"python glue / transfers":28s} {glue:8.2f}s  ({100*glue/total:5.1f}%)')

    if args.solver_internals:
        print(f'\n--- in-solver segment shares ({args.backend}, instrumented rerun) ---')
        _solver_internals(args.system, args.backend)


if __name__ == '__main__':
    main()
