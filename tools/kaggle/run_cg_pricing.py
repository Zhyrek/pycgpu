"""cg's batched pricing kernel (pycalphad/cg/cg_price.h) on a GPU, run as a Kaggle script job.

Launched by tools/kaggle/launch.py --script run_cg_pricing.py. It installs the fork at the requested branch and
solves the same tiles (TileSolver, PYCALPHAD_CG_TILE=2) with pricing in Cython, in the host C++ library and in
the CUDA kernel; checks the kernel's results job by job against the host library's; and times them. Results go
to the output directory (/kaggle/working on Kaggle): env.txt, install.log, cg_pricing.json.

With --bench tile (launch.py push ... -- --bench tile): whole tiles in compiled code (PYCALPHAD_CG_TILE=3) by the
host driver (cg_tile_host.h, one core) and by the device driver (cg_tile_device.h's kernels, DEVICE 'cuda'):
phase sets and GM against the host driver's, wall time per tile and per kernel step; into cg_tile.json.
With --bench compile: nvcc's time for the tile's module (and variants of its source), into cg_compile.json;
--bench cache: the module as CuPy builds it in a new process (cg_cache.json). Several, in turn: --bench tile,cache.

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
biggest = {}
def both(jobs, job_phase, job_cond, mu, y0, y, x, G, tpd, conv, evals):
    P = ctypes.c_void_p
    hy, hx, hG, ht = y.copy(), x.copy(), G.copy(), tpd.copy(); hc, he = conv.copy(), evals.copy()
    di, dd, ip, dp = host._desc
    host.function(ctypes.c_int(len(jobs)), *[a.ctypes.data_as(P) for a in (np.ascontiguousarray(jobs), job_phase, job_cond, mu, y0)],
                  ctypes.c_int(y0.shape[1]), *[a.ctypes.data_as(P) for a in (hy, hx, hG, ht, hc, he, di, dd, ip, dp)],
                  ctypes.c_int(mu.shape[1]), ctypes.c_double(cuda.RT), host._sv.ctypes.data_as(P))
    cuda.price(jobs, job_phase, job_cond, mu, y0, y, x, G, tpd, conv, evals)
    if len(jobs) > len(biggest.get('jobs', ())):
        biggest.update(jobs=np.array(jobs), jp=job_phase[jobs].copy(), jc=job_cond[jobs].copy(), mu=mu.copy(),
                       y0=y0[jobs].copy())
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

# divergence: the largest batch replayed on the device in condition order (warps mix phases, as a kernel running
# a whole condition per thread would) and sorted by phase (each warp runs one phase's code); several block sizes
import cupy as cp
b = biggest; n = len(b['jobs'])
C, ys = b['mu'].shape[1], b['y0'].shape[1]
di, dd, ip, dp = cuda.d_desc
def replay(order, block, reps=3):
    jp = cp.asarray(b['jp'][order]); jc = cp.asarray(b['jc'][order]); d_mu = cp.asarray(b['mu']); d_y0 = cp.asarray(b['y0'][order])
    d_y = cp.zeros((n, ys)); d_x = cp.zeros((n, C)); d_G = cp.zeros(n); d_t = cp.zeros(n)
    d_c = cp.zeros(n, dtype=cp.int32); d_e = cp.zeros(2 * n, dtype=cp.int32); ident = cp.arange(n, dtype=cp.int32)
    best = None
    for _ in range(reps):
        s_, e_ = cp.cuda.Event(), cp.cuda.Event(); s_.record()
        cuda.kernel(((n + block - 1) // block,), (block,), (np.int32(n), ident, jp, jc, d_mu, d_y0, np.int32(ys), d_y, d_x,
                    d_G, d_t, d_c, d_e, di, dd, ip, dp, np.int32(C), np.float64(cuda.RT), cuda.d_sv))
        e_.record(); e_.synchronize()
        ms = cp.cuda.get_elapsed_time(s_, e_); best = ms if best is None else min(best, ms)
    return best / 1e3, d_e.get()
div = {'jobs': n, 'size_classes': [list(c) for c in cuda.classes]}
by_phase = np.argsort(b['jp'], kind='stable')
# the class kernels (as CudaPricer prices): sorted by class and phase, one launch per class; kernel time only
yy = np.zeros((n, ys)); xx = np.zeros((n, C)); GG = np.zeros(n); tt = np.zeros(n); cc = np.zeros(n, dtype=np.int32); ee = np.zeros(2 * n, dtype=np.int32)
best = None
for _ in range(3):
    k0 = cuda.kernel_seconds
    cuda.price(np.arange(n, dtype=np.int32), b['jp'], np.arange(n, dtype=np.int32) * 0 + b['jc'], b['mu'], b['y0'], yy, xx, GG, tt, cc, ee)
    sec = cuda.kernel_seconds - k0; best = sec if best is None else min(best, sec)
div['class_kernels_sorted_s'] = round(best, 4)
for block in (32, 64, 128, 256):
    for lab, order in (('condition_order', np.arange(n)), ('sorted_by_phase', by_phase)):
        try:
            sec, ev = replay(order, block)
            div[f'{lab}_block{block}_s'] = round(sec, 4)
        except Exception as e:
            div[f'{lab}_block{block}_s'] = repr(e)[:200]
div['evals_per_job'] = [float(ev[0::2].mean()), float(ev[1::2].mean())]
# the same batch on one host core (the library)
P = ctypes.c_void_p
hy = np.zeros((n, ys)); hx = np.zeros((n, C)); hG = np.zeros(n); ht = np.zeros(n); hc = np.zeros(n, dtype=np.int32); he = np.zeros(2 * n, dtype=np.int32)
hdi, hdd, hip, hdp = host._desc; ident = np.arange(n, dtype=np.int32)
t = time.perf_counter()
host.function(ctypes.c_int(n), *[np.ascontiguousarray(a).ctypes.data_as(P) for a in (ident, b['jp'], b['jc'], b['mu'], b['y0'])], ctypes.c_int(ys),
              *[a.ctypes.data_as(P) for a in (hy, hx, hG, ht, hc, he, hdi, hdd, hip, hdp)], ctypes.c_int(C), ctypes.c_double(cuda.RT), host._sv.ctypes.data_as(P))
div['host_one_core_s'] = round(time.perf_counter() - t, 4)
report['divergence'] = div
print('divergence:', div, flush=True)
json.dump(report, open(sys.argv[1], 'w'), indent=1)
'''

TILE_BENCH = r'''
import json, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, pycalphad
from importlib.resources import files
import pycalphad.tests.databases as dbs
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases, unpack_species, instantiate_models
import pycalphad.cg.solve as S
from pycalphad.cg import device as Dv
pycalphad.set_backend('cg', threads=1)
S.TILE_MODE = 3
cost = ['AL', 'MG', 'SI', 'CU', 'ZN', 'FE', 'MN', 'NI', 'CR', 'TI']
minor = {'FE': 0.005, 'MN': 0.005, 'NI': 0.005, 'CR': 0.005, 'TI': 0.001}
def g(n): return list(np.linspace(0.005, 0.06, n))
CASES = {
    'alcrni_40x40': ('alcocrni.tdb', ['AL', 'CR', 'NI', 'VA'],
                     {v.T: 1300.0, v.X('AL'): list(np.linspace(0.02, 0.5, 40)), v.X('CR'): list(np.linspace(0.02, 0.5, 40))}),
    'cost10_8^4': ('COST507.tdb', cost + ['VA'],
                   {v.T: 750.0, **{v.X(e): x for e, x in minor.items()}, v.X('MG'): g(8), v.X('SI'): g(8), v.X('CU'): g(8), v.X('ZN'): g(8)}),
}
TILES = []   # per solve_tile call: where, conditions, seconds, counters, step times
def timed(cls, where):
    orig = cls.solve_tile
    def solve_tile(self, *a, **k):
        t = time.perf_counter(); out = orig(self, *a, **k); sec = time.perf_counter() - t
        if out is not None:
            TILES.append({'where': where, 'conditions': len(out[0]), 'seconds': round(sec, 4),
                          'counters': dict(self.tile_counters),
                          'steps': {s_: round(x, 4) for s_, x in (getattr(self, 'tile_times', None) or {}).items()}})
        return out
    cls.solve_tile = solve_tile
timed(Dv.HostPricer, 'host'); timed(Dv.CudaPricer, 'cuda')
report = {}
for name in sys.argv[2:]:
    db, comps, c = CASES[name]
    dbf = Database(str(files(dbs) / db)); phases = filter_phases(dbf, unpack_species(dbf, comps))
    models = instantiate_models(dbf, comps, phases)
    conds = {v.N: 1, v.P: 101325, **c}
    out, rep = {}, {}
    for dev in ('', 'cuda'):
        S.DEVICE = dev
        t = time.perf_counter(); equilibrium(dbf, comps, phases, conds, model=models); first = time.perf_counter() - t
        walls = []
        for rep_ in range(2):
            TILES.clear()
            t = time.perf_counter(); eq = equilibrium(dbf, comps, phases, conds, model=models)
            walls.append(round(time.perf_counter() - t, 3))
        GM = eq.GM.values.ravel(); NP = eq.NP.values.reshape(len(GM), -1); Ph = eq.Phase.values.reshape(NP.shape)
        sets = ['+'.join(sorted(p for p, a in zip(Ph[i], NP[i]) if p and a > 1e-6)) for i in range(len(GM))]
        out[dev or 'host'] = (GM, sets, eq.MU.values.reshape(len(GM), -1))
        r = {'first_call_s': round(first, 2), 'wall_s': walls, 'tiles': list(TILES)}
        if dev == 'cuda':
            import cupy as cp
            r['device_pool_bytes'] = int(cp.get_default_memory_pool().total_bytes())
        rep[dev or 'host'] = r
    ref_GM, ref_sets, ref_MU = out['host']
    G, sets, MU = out['cuda']
    rep['cuda']['max_abs_dGM_vs_host'] = float(np.nanmax(np.abs(G - ref_GM)))
    dG = np.nan_to_num(np.abs(G - ref_GM))
    rep['cuda']['largest_dGM_vs_host'] = [   # (condition, phases, GM, |dGM|, max |dMU|)
        [int(i), ref_sets[i], float(ref_GM[i]), float(dG[i]), float(np.nanmax(np.abs(MU[i] - ref_MU[i])))]
        for i in np.argsort(-dG)[:8]]
    rep['cuda']['phase_sets_differ_vs_host'] = int(sum(a != b for a, b in zip(sets, ref_sets)))
    rep['cuda']['nan'] = [int(np.isnan(G).sum()), int(np.isnan(ref_GM).sum())]
    report[name] = rep
    print(name, json.dumps(rep, indent=1), flush=True)
    json.dump(report, open(sys.argv[1], 'w'), indent=1)
'''

COMPILE_BENCH = r'''
import json, os, re, subprocess, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, pycalphad
from importlib.resources import files
import pycalphad.tests.databases as dbs
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases, unpack_species, instantiate_models
import pycalphad.cg.solve as S
import pycalphad.gpu.kernel_manager as KM
import cupy as cp
pycalphad.set_backend('cg', threads=1)
S.TILE_MODE = 3; S.DEVICE = 'cuda'
cost = ['AL', 'MG', 'SI', 'CU', 'ZN', 'FE', 'MN', 'NI', 'CR', 'TI']
minor = {'FE': 0.005, 'MN': 0.005, 'NI': 0.005, 'CR': 0.005, 'TI': 0.001}
CASES = {
    'alcrni_40x40': ('alcocrni.tdb', ['AL', 'CR', 'NI', 'VA'], {v.T: 1300.0, v.X('AL'): [0.1, 0.2], v.X('CR'): [0.1, 0.2]}),
    'cost10_8^4': ('COST507.tdb', cost + ['VA'],
                   {v.T: 750.0, **{v.X(e): x for e, x in minor.items()}, v.X('MG'): [0.01, 0.02], v.X('SI'): [0.01, 0.02],
                    v.X('CU'): [0.01, 0.02], v.X('ZN'): [0.01, 0.02]}),
}
class Captured(Exception):
    pass
def capture(code, options, verbose=False):
    raise Captured(code, options)
cupy_module = KM.cuda_raw_module
KM.cuda_raw_module = capture
price_h = open(os.path.join(os.path.dirname(S.__file__), 'cg_price.h')).read()
NOINLINE = ('cg_price', 'cg_sens', 'cg_polish_core', 'cg_gen_jobs', 'cg_add_point')
def variant(src, name):
    """the tile module's source (as_is) with: the generated model functions inlined into cg_eval (inline), also
    the solver's big functions out of line (all), without the model code (no_model), or the model code alone
    (model_only)"""
    if name == 'inline':
        src = re.sub(r'(?m)^__device__ __noinline__ (\w+) (pycgpu_model_\d+_)', r'__device__ \1 \2', src)
    if name == 'all':
        for f in NOINLINE:
            src = re.sub(r'CG_DEVICE static (\w+) %s\(' % f, r'CG_DEVICE __noinline__ static \1 %s(' % f, src)
    if name == 'no_model':
        a = src.index('#endif\n', src.index('#include <math.h>')) + 7
        src = src[:a] + src[src.index('#define CG_DEVICE __device__'):]
        src = re.sub(r'(?m)^    case \d+: .*$', '', src)
    if name == 'model_only':
        src = src[:src.index(price_h[:300])]
        src += ('extern "C" __global__ void cg_eval_kernel(int n, const int* model, double* eg, double* H, const double* x) {\n'
                '    int i = blockIdx.x * blockDim.x + threadIdx.x;\n'
                '    if (i < n) cg_eval(model[i], i & 1, eg + i * 64, H + i * 4096, x + i * 64);\n}\n')
    return src
cc = cp.cuda.Device().compute_capability
nvcc = cp.cuda.get_nvcc_path() if hasattr(cp.cuda, 'get_nvcc_path') else 'nvcc'
report = {'nvcc': subprocess.run([nvcc, '--version'], capture_output=True, text=True).stdout.strip().splitlines()[-1],
          'arch': f'sm_{cc}', 'cupy': cp.__version__, 'cupy_cache_dir': os.environ.get('CUPY_CACHE_DIR', '~/.cupy/kernel_cache')}
work = os.path.abspath('compile_work'); os.makedirs(work, exist_ok=True)
for name in sys.argv[2:]:
    db, comps, c = CASES[name]
    dbf = Database(str(files(dbs) / db)); phases = filter_phases(dbf, unpack_species(dbf, comps))
    models = instantiate_models(dbf, comps, phases)
    try:
        equilibrium(dbf, comps, phases, {v.N: 1, v.P: 101325, **c}, model=models)
        raise RuntimeError('the tile module was not compiled')
    except Captured as e:
        src, options = e.args
    rep = {'source_bytes': len(src), 'options': list(options)}
    for var in os.environ.get('CG_COMPILE_VARIANTS', 'as_is,all,no_model,model_only,inline').split(','):
        if var == 'cupy':   # (the module as CudaPricer builds it: from CuPy's kernel cache if it holds it)
            t = time.perf_counter()
            cupy_module(src, options).get_function('cg_dev_lp')
            rep[var] = {'seconds': round(time.perf_counter() - t, 2)}
            print(name, var, rep[var], flush=True)
            continue
        vs = variant(src, var)
        path = os.path.join(work, f'{name}_{var}.cu')
        open(path, 'w').write(vs)
        cmd = [nvcc, '--cubin', f'-arch=sm_{cc}', *options, '--time', path + '.time.csv', '-o', path + '.cubin', path]
        t = time.perf_counter()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1500)
            out = {'seconds': round(time.perf_counter() - t, 1), 'rc': r.returncode, 'stderr': r.stderr[-1500:]}
        except subprocess.TimeoutExpired:
            out = {'seconds': 'timeout 1500'}
        if os.path.exists(path + '.time.csv'):
            out['phases'] = open(path + '.time.csv').read()[-3000:]
        if os.path.exists(path + '.cubin'):
            out['cubin_bytes'] = os.path.getsize(path + '.cubin')
        rep[var] = out
        print(name, var, json.dumps(out)[:600], flush=True)
        report[name] = rep
        json.dump(report, open(sys.argv[1], 'w'), indent=1)
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
    parser.add_argument('--bench', default='pricing',
                        help="comma-separated, run in turn: 'pricing' (the pricing kernel in the Cython tile), 'tile' "
                             "(whole tiles on the device), 'compile' (nvcc's time for the tile's module and variants "
                             "of its source), 'cache' (the tile's module as CuPy builds it, in a new process: from its "
                             "kernel cache after 'tile')")
    parser.add_argument('--no-fmad', action='store_true', help='kernels without fused multiply-adds')
    args = parser.parse_args(KAGGLE_ARGS if ON_KAGGLE else None)
    os.makedirs(args.out, exist_ok=True)
    summary = {'branch': BRANCH, 'gpu': environment(args.out)}
    if not args.no_install:
        summary['install'] = install(args.out, args.repo_dir)
    summary['commit'] = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                                       cwd=args.repo_dir).stdout.strip()
    summary['fmad'] = not args.no_fmad
    summary['bench'], summary['results'] = {}, {}
    for name in args.bench.split(','):
        bench = os.path.join(args.out, f'cg_{name}_bench.py')
        with open(bench, 'w') as f:
            f.write({'pricing': BENCH, 'tile': TILE_BENCH, 'compile': COMPILE_BENCH, 'cache': COMPILE_BENCH}[name])
        result = os.path.join(args.out, f'cg_{name}.json')
        env = dict(os.environ, PYTHONHASHSEED='0', CG_DEVICES=args.devices,
                   PYCALPHAD_CG_FMAD='0' if args.no_fmad else '1')
        if name == 'cache':
            env['CG_COMPILE_VARIANTS'] = 'cupy'
        summary['bench'][name] = run([sys.executable, bench, result] + args.cases.split(','),
                                     os.path.join(args.out, 'bench.log'), timeout=args.timeout, cwd=args.out, env=env)
        if os.path.exists(result):
            with open(result) as f:
                summary['results'][name] = json.load(f)
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
