"""GPU checks for the compiled backends, run as a Kaggle script job.

Launched by tools/kaggle/launch.py, which uploads this file and picks the GPU.
It installs the fork at the requested branch, runs the test suite on the CUDA
backends, compares CUDA with the C++ backend on equilibrium grids, and times
both. Everything it reports is written to the output directory
(/kaggle/working on Kaggle): env.txt, install.log, pytest_<backend>.txt,
compare.json and summary.json.

Outside Kaggle (e.g. on a rented machine with an NVIDIA or ROCm GPU) run it
directly: python run_gpu_checks.py --out results/ [--repo-dir checkout/]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

REPO = 'https://github.com/Zhyrek/pycgpu.git'
BRANCH = 'gpu-port-sync'  # launch.py rewrites this line for other branches
KAGGLE_ARGS = []  # command-line options for Kaggle runs (set by launch.py)
ON_KAGGLE = os.path.isdir('/kaggle/working')


def run(cmd, log_path, timeout=None, cwd=None):
    """Run `cmd`, appending its output to `log_path`; return (exit code, seconds)."""
    start = time.time()
    with open(log_path, 'a') as log:
        log.write(f'$ {" ".join(cmd)}\n')
        log.flush()
        try:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, cwd=cwd).returncode
        except subprocess.TimeoutExpired:
            log.write(f'\n[timed out after {timeout} s]\n')
            rc = 'timeout'
    return rc, round(time.time() - start, 1)


def environment(out):
    lines = [f'python {sys.version}']
    for cmd in (['nvidia-smi'], ['nvcc', '--version'], ['rocm-smi', '--showproductname'], ['g++', '--version'],
                [sys.executable, '-c', 'import cupy; cupy.show_config()']):
        if shutil.which(cmd[0]):
            lines.append(subprocess.run(cmd, capture_output=True, text=True).stdout)
    with open(os.path.join(out, 'env.txt'), 'w') as f:
        f.write('\n'.join(lines))


def install(out, repo_dir):
    log = os.path.join(out, 'install.log')
    steps = {}
    if not os.path.isdir(repo_dir):
        steps['clone'] = run(['git', 'clone', '--depth', '1', '-b', BRANCH, REPO, repo_dir], log)
    try:
        import cupy  # noqa: F401  (Kaggle images may already ship one)
    except ImportError:
        wheel = 'cupy-rocm-6-0' if shutil.which('rocm-smi') and not shutil.which('nvidia-smi') else 'cupy-cuda12x'
        steps['cupy'] = run([sys.executable, '-m', 'pip', 'install', wheel], log)
    steps['pycalphad'] = run([sys.executable, '-m', 'pip', 'install', '-e', repo_dir, 'pytest'], log)
    return steps


def test_suite(out, repo_dir, backend, timeout):
    log = os.path.join(out, f'pytest_{backend}.txt')
    rc, seconds = run([sys.executable, '-m', 'pytest', 'pycalphad/tests', f'--backend={backend}', '-q', '-rfEs',
                       '-p', 'no:cacheprovider', '--durations=15'], log, timeout=timeout, cwd=repo_dir)
    with open(log) as f:
        tail = [line.strip() for line in f.read().splitlines() if line.strip()][-1:]
    return {'exit': rc, 'seconds': seconds, 'result': tail[0] if tail else ''}


COMPARE = r'''
import json, sys, time, numpy as np, pycalphad
from importlib.resources import files
import pycalphad.tests.databases as dbs
from pycalphad import Database, equilibrium, variables as v
cases = {
    'Al-Ni': ('alni_dupin_2001.tdb', ['AL', 'NI', 'VA'], {v.T: (1000, 1700, STEP_T), v.X('AL'): (0.01, 0.99, STEP_X)}),
    'Al-Cr-Ni': ('alcocrni.tdb', ['AL', 'CR', 'NI', 'VA'], {v.T: 1300, v.X('AL'): (0.01, 0.6, STEP_X3), v.X('CR'): (0.01, 0.6, STEP_X3)}),
}
if LARGE:  # throughput: GPUs only pay off on large batches
    cases['Al-Ni large'] = ('alni_dupin_2001.tdb', ['AL', 'NI', 'VA'], {v.T: (1000, 1700, 5), v.X('AL'): (0.0025, 0.9975, 0.0025)})
results = {}
for name, (tdb, comps, conds) in cases.items():
    dbf = Database(str(files(dbs) / tdb))
    phases = list(dbf.phases.keys())
    conds = {v.N: 1, v.P: 101325, **conds}
    entry = {}
    outputs = {}
    for backend in ('cpp', 'cuda', 'cuda-fast'):
        try:
            t = time.time(); eq = equilibrium(dbf, comps, phases, conds, backend=backend); first = time.time() - t
            t = time.time(); eq = equilibrium(dbf, comps, phases, conds, backend=backend); second = time.time() - t
            outputs[backend] = eq
            entry[backend] = {'first_call_s': round(first, 2), 'warm_call_s': round(second, 2),
                              'conditions': int(eq.GM.size), 'nan_GM': int(np.isnan(eq.GM.values).sum())}
        except Exception as e:
            import traceback
            text = ''.join(traceback.format_exception(e))
            # (keep both ends: the root cause comes first in a chained traceback)
            entry[backend] = {'error': text if len(text) <= 12000 else text[:6000] + '\n[...]\n' + text[-6000:]}
    ref = outputs.get('cpp')
    for backend in ('cuda', 'cuda-fast'):
        eq = outputs.get(backend)
        if ref is None or eq is None:
            continue
        gm, gm_ref = eq.GM.values, ref.GM.values
        both = ~np.isnan(gm) & ~np.isnan(gm_ref)
        phases_cmp = np.sort(np.where(eq.Phase.values == '', '~', eq.Phase.values), axis=-1)
        phases_ref = np.sort(np.where(ref.Phase.values == '', '~', ref.Phase.values), axis=-1)
        same = (phases_cmp == phases_ref).all(axis=-1)
        entry[backend].update({
            'max_abs_dGM_vs_cpp': float(np.abs(gm - gm_ref)[both].max()) if both.any() else None,
            'conditions_with_different_phase_set': int((~same).sum()),
            'nan_mismatch_vs_cpp': int((np.isnan(gm) != np.isnan(gm_ref)).sum()),
        })
    results[name] = entry
json.dump(results, open(sys.argv[1], 'w'), indent=1)
'''


def compare(out, quick):
    script = os.path.join(out, 'compare_script.py')
    steps = ('10', '0.01', '0.02') if not quick else ('100', '0.1', '0.1')
    src = (COMPARE.replace('STEP_T', steps[0]).replace('STEP_X3', steps[2]).replace('STEP_X', steps[1])
           .replace('LARGE', str(not quick)))
    with open(script, 'w') as f:
        f.write(src)
    path = os.path.join(out, 'compare.json')
    rc, seconds = run([sys.executable, script, path], os.path.join(out, 'compare.log'), timeout=3600)
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = None
    return {'exit': rc, 'seconds': seconds, 'results': data}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--out', default='/kaggle/working' if ON_KAGGLE else 'gpu_check_results')
    parser.add_argument('--repo-dir', default='/kaggle/tmp/pycgpu' if ON_KAGGLE else 'pycgpu')
    parser.add_argument('--suite-timeout', type=int, default=5400, help='seconds per test-suite run')
    parser.add_argument('--quick', action='store_true', help='small comparison grids')
    parser.add_argument('--skip-suite', action='store_true')
    args = parser.parse_args(KAGGLE_ARGS if ON_KAGGLE else None)
    os.makedirs(args.out, exist_ok=True)
    summary = {'branch': BRANCH}
    environment(args.out)
    summary['install'] = install(args.out, args.repo_dir)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=args.repo_dir)
    summary['commit'] = commit.stdout.strip()
    summary['compare'] = compare(args.out, args.quick)
    if not args.skip_suite:
        summary['suite'] = {backend: test_suite(args.out, args.repo_dir, backend, args.suite_timeout)
                            for backend in ('cuda', 'cpp')}
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
