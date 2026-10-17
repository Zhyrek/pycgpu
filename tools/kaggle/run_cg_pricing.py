"""cg's batched pricing kernel (pycalphad/cg/cg_price.h) on a GPU, run as a Kaggle script job.

Launched by tools/kaggle/launch.py --script run_cg_pricing.py. It installs the fork at the requested branch and
solves the same tiles (TileSolver, PYCALPHAD_CG_TILE=2) with pricing in Cython, in the host C++ library and in
the CUDA kernel; checks the kernel's results job by job against the host library's; and times them. Results go
to the output directory (/kaggle/working on Kaggle): env.txt, install.log, cg_pricing.json.

Elsewhere (a machine with an NVIDIA or ROCm GPU): python run_cg_pricing.py --out results/ [--repo-dir checkout/]
[--no-install]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

REPO = 'https://github.com/Zhyrek/pycgpu.git'
BRANCH = 'dense-grid'  # launch.py rewrites this line for other branches
KAGGLE_ARGS = []  # command-line options for Kaggle runs (set by launch.py)
ON_KAGGLE = os.path.isdir('/kaggle/working')


def run(cmd, log_path, timeout=None, cwd=None, env=None):
    start = time.time()
    with open(log_path, 'a') as log:
        log.write(f'$ {" ".join(cmd)}\n')
        log.flush()
        try:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, cwd=cwd, env=env).returncode
        except subprocess.TimeoutExpired:
            log.write(f'\n[timed out after {timeout} s]\n')
            rc = 'timeout'
    return rc, round(time.time() - start, 1)


def environment(out):
    lines = [f'python {sys.version}']
    for cmd in (['nvidia-smi'], ['rocm-smi', '--showproductname'], ['lscpu'], ['g++', '--version']):
        if shutil.which(cmd[0]):
            lines.append(subprocess.run(cmd, capture_output=True, text=True).stdout)
    with open(os.path.join(out, 'env.txt'), 'w') as f:
        f.write('\n'.join(lines))
    if shutil.which('nvidia-smi'):
        return subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True,
                              text=True).stdout.split('\n')[0].strip()
    return None


def install(out, repo_dir):
    log = os.path.join(out, 'install.log')
    steps = {}
    if not os.path.isdir(repo_dir):
        steps['clone'] = run(['git', 'clone', '-b', BRANCH, REPO, repo_dir], log)
    try:
        import cupy  # noqa: F401
    except ImportError:
        wheel = 'cupy-rocm-6-0' if shutil.which('rocm-smi') and not shutil.which('nvidia-smi') else 'cupy-cuda12x'
        steps['cupy'] = run([sys.executable, '-m', 'pip', 'install', wheel], log)
    steps['pycalphad'] = run([sys.executable, '-m', 'pip', 'install', '-e', repo_dir], log)
    return steps


BENCH = r'''
import json, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, pycalphad
from importlib.resources import files
import pycalphad.tests.databases as dbs
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases, unpack_species, instantiate_models
import pycalphad.cg.solve as S
pycalphad.set_backend('cg', threads=1)
S.TILE_MODE = 2; S.COLLECT_STATS = True
cost = ['AL', 'MG', 'SI', 'CU', 'ZN', 'FE', 'MN', 'NI', 'CR', 'TI']
minor = {'FE': 0.005, 'MN': 0.005, 'NI': 0.005, 'CR': 0.005, 'TI': 0.001}
def g(n): return list(np.linspace(0.005, 0.06, n))
CASES = {
    'alcrni_40x40': ('alcocrni.tdb', ['AL', 'CR', 'NI', 'VA'],
                     {v.T: 1300.0, v.X('AL'): list(np.linspace(0.02, 0.5, 40)), v.X('CR'): list(np.linspace(0.02, 0.5, 40))}),
    'cost10_8^4': ('COST507.tdb', cost + ['VA'],
                   {v.T: 750.0, **{v.X(e): x for e, x in minor.items()}, v.X('MG'): g(8), v.X('SI'): g(8), v.X('CU'): g(8), v.X('ZN'): g(8)}),
}
import os
DEVICES = os.environ.get('CG_DEVICES', ',host,cuda').split(',')
report = {}
for name in sys.argv[2:]:
    db, comps, c = CASES[name]
    dbf = Database(str(files(dbs) / db)); phases = filter_phases(dbf, unpack_species(dbf, comps))
    models = instantiate_models(dbf, comps, phases)
    conds = {v.N: 1, v.P: 101325, **c}
    out, rep = {}, {}
    for dev in DEVICES:
        S.DEVICE = dev
        t = time.perf_counter(); equilibrium(dbf, comps, phases, conds, model=models); first = time.perf_counter() - t
        cud = [p for (d, *_), p in S._PRICERS.items() if d == 'cuda']
        for p in cud:
            p.seconds = p.kernel_seconds = 0.0; p.jobs = p.batches = 0
        S.SEQUENCE_STATS.clear()
        t = time.perf_counter(); eq = equilibrium(dbf, comps, phases, conds, model=models); wall = time.perf_counter() - t
        st = S.SEQUENCE_STATS; n = sum(x['n'] for x in st)
        GM = eq.GM.values.ravel(); NP = eq.NP.values.reshape(len(GM), -1); Ph = eq.Phase.values.reshape(NP.shape)
        sets = ['+'.join(sorted(p for p, a in zip(Ph[i], NP[i]) if p and a > 1e-6)) for i in range(len(GM))]
        out[dev or 'cython'] = (GM, sets)
        r = {'first_call_s': round(first, 2), 'wall_s': round(wall, 3), 'conditions': n,
             'minimizations_per_condition': round(sum(x['priced'] for x in st) / n, 1),
             'jobs_per_round': [r_['jobs'] for r_ in st[0]['rounds'][1:]]}
        if cud:
            p = cud[0]
            r.update(device_jobs=p.jobs, device_batches=p.batches, device_s=round(p.seconds, 4),
                     kernel_s=round(p.kernel_seconds, 4),
                     kernel_us_per_job=round(1e6 * p.kernel_seconds / max(p.jobs, 1), 3))
        rep[dev or 'cython'] = r
    ref_GM, ref_sets = out['host']
    for k, (G, sets) in out.items():
        rep[k]['max_abs_dGM_vs_host'] = float(np.nanmax(np.abs(G - ref_GM)))
        rep[k]['phase_sets_differ_vs_host'] = int(sum(a != b for a, b in zip(sets, ref_sets)))
    report[name] = rep
    print(name, json.dumps(rep, indent=1), flush=True)

json.dump(report, open(sys.argv[1], 'w'), indent=1)
if 'cuda' not in DEVICES:
    sys.exit(0)
# the kernel against the host library, job by job, on the jobs of one COST507-10 tile
import ctypes
host = [p for (d, *_), p in S._PRICERS.items() if d == 'host'][-1]
cuda = [p for (d, *_), p in S._PRICERS.items() if d == 'cuda'][-1]
diffs = {'tpd': 0.0, 'y': 0.0, 'conv': 0, 'jobs': 0}
def both(jobs, job_phase, job_cond, mu, y0, y, x, G, tpd, conv, evals):
    P = ctypes.c_void_p
    hy, hx, hG, ht = y.copy(), x.copy(), G.copy(), tpd.copy(); hc, he = conv.copy(), evals.copy()
    di, dd, ip, dp = host._desc
    host.function(ctypes.c_int(len(jobs)), *[a.ctypes.data_as(P) for a in (np.ascontiguousarray(jobs), job_phase, job_cond, mu, y0)],
                  ctypes.c_int(y0.shape[1]), *[a.ctypes.data_as(P) for a in (hy, hx, hG, ht, hc, he, di, dd, ip, dp)],
                  ctypes.c_int(mu.shape[1]), ctypes.c_double(cuda.RT), host._sv.ctypes.data_as(P))
    cuda.price(jobs, job_phase, job_cond, mu, y0, y, x, G, tpd, conv, evals)
    fin = np.isfinite(ht[jobs]) & np.isfinite(tpd[jobs])
    diffs['tpd'] = max(diffs['tpd'], float(np.max(np.abs(ht[jobs] - tpd[jobs])[fin], initial=0)))
    diffs['y'] = max(diffs['y'], float(np.nanmax(np.abs(hy[jobs] - y[jobs]), initial=0)))
    diffs['conv'] += int((hc[jobs] != conv[jobs]).sum()); diffs['jobs'] += len(jobs)
orig_attach = cuda.attach
def attach(solver, cores, sv_values):
    ok = cuda.upload(cores, sv_values)
    host._desc = host.descriptions(cores)[1:]; host._sv = np.asarray(sv_values, dtype=float)
    solver.set_pricer_callback(both, ok)
    return int(ok.sum())
cuda.attach = attach
S.DEVICE = 'cuda'
db, comps, c = CASES['cost10_8^4']
dbf = Database(str(files(dbs) / db)); phases = filter_phases(dbf, unpack_species(dbf, comps))
equilibrium(dbf, comps, phases, {v.N: 1, v.P: 101325, **c})
report['kernel_vs_host_jobs'] = diffs
print('kernel vs host, job by job:', diffs, flush=True)
json.dump(report, open(sys.argv[1], 'w'), indent=1)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', default='/kaggle/working' if ON_KAGGLE else 'cg_pricing_results')
    parser.add_argument('--repo-dir', default='/kaggle/tmp/pycgpu' if ON_KAGGLE else 'pycgpu')
    parser.add_argument('--no-install', action='store_true')
    parser.add_argument('--cases', default='alcrni_40x40,cost10_8^4')
    parser.add_argument('--timeout', type=int, default=7200)
    parser.add_argument('--devices', default=',host,cuda', help="pricers to compare ('' = Cython)")
    args = parser.parse_args(KAGGLE_ARGS if ON_KAGGLE else None)
    os.makedirs(args.out, exist_ok=True)
    summary = {'branch': BRANCH, 'gpu': environment(args.out)}
    if not args.no_install:
        summary['install'] = install(args.out, args.repo_dir)
    summary['commit'] = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                                       cwd=args.repo_dir).stdout.strip()
    bench = os.path.join(args.out, 'cg_pricing_bench.py')
    with open(bench, 'w') as f:
        f.write(BENCH)
    result = os.path.join(args.out, 'cg_pricing.json')
    env = dict(os.environ, PYTHONHASHSEED='0', CG_DEVICES=args.devices)
    summary['bench'] = run([sys.executable, bench, result] + args.cases.split(','), os.path.join(args.out, 'bench.log'),
                           timeout=args.timeout, cwd=args.out, env=env)
    if os.path.exists(result):
        with open(result) as f:
            summary['results'] = json.load(f)
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
