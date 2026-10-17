"""Run tools/kaggle/run_gpu_checks.py on a Kaggle GPU and fetch its results.

Needs the Kaggle CLI (pip install kaggle) and an API token from
kaggle.com -> Settings -> API, saved as ~/.kaggle/access_token (older
tokens: ~/.kaggle/kaggle.json), on an account with a verified phone number
(required for GPUs and internet access). Runs count against the account's
free weekly GPU quota; nothing is billed.

    python tools/kaggle/launch.py push --gpu t4 [--branch NAME] [--timeout 10800] [--quick] [--skip-suite]
        [--skip-compare] [--suites cuda,cpp] [--suite-timeout 5400]
    python tools/kaggle/launch.py status --gpu t4
    python tools/kaggle/launch.py fetch --gpu t4 [--dest DIR]
    python tools/kaggle/launch.py quota

Each GPU type gets its own private job (<user>/pycgpu-gpu-checks-<gpu>), so
a T4 and a P100 run can be in flight together. --script runs another job
script from this directory instead (its own job, <user>/pycgpu-<script>-<gpu>;
pass it to status and fetch too), e.g. run_cg_pricing.py, cg's batched
pricing kernel:

    python tools/kaggle/launch.py push --gpu p100 --script run_cg_pricing.py --branch dense-grid
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ACCELERATORS = {'t4': 'NvidiaTeslaT4', 'p100': 'NvidiaTeslaP100'}


def kaggle(*args, capture=False):
    exe = shutil.which('kaggle')
    if exe is None:
        sys.exit("the Kaggle CLI is not installed (pip install kaggle)")
    if capture:
        return subprocess.run([exe, *args], capture_output=True, text=True, check=True).stdout
    subprocess.run([exe, *args], check=True)


def username():
    """The account's username: from a legacy kaggle.json, else as the CLI
    reports it (`kaggle config view` prints settings, never the token)."""
    legacy = os.path.expanduser(os.path.join(os.environ.get('KAGGLE_CONFIG_DIR', '~/.kaggle'), 'kaggle.json'))
    if os.path.exists(legacy):
        with open(legacy) as f:
            return json.load(f)['username']
    match = re.search(r'^- username: (\S+)', kaggle('config', 'view', capture=True), flags=re.M)
    if match is None or match.group(1) == 'None':
        sys.exit("the Kaggle CLI is not authenticated (kaggle.com -> Settings -> API -> Generate New Token, "
                 "saved as ~/.kaggle/access_token)")
    return match.group(1)


DEFAULT_SCRIPT = 'run_gpu_checks.py'


def job_name(gpu, script=DEFAULT_SCRIPT):
    stem = 'gpu-checks' if script == DEFAULT_SCRIPT else os.path.splitext(script)[0].replace('run_', '').replace('_', '-')
    return f"pycgpu-{stem}-{gpu}"


def kernel_id(gpu, script=DEFAULT_SCRIPT):
    return f"{username()}/{job_name(gpu, script)}"


def push(args):
    with tempfile.TemporaryDirectory() as folder:
        with open(os.path.join(HERE, args.script)) as f:
            source = f.read()
        source, n = re.subn(r"^BRANCH = .*$", f"BRANCH = {args.branch!r}", source, count=1, flags=re.M)
        assert n == 1
        if args.script == DEFAULT_SCRIPT:
            job_args = (['--quick'] if args.quick else []) + (['--skip-suite'] if args.skip_suite else []) + \
                       (['--skip-compare'] if args.skip_compare else []) + \
                       ['--suites', args.suites, '--suite-timeout', str(args.suite_timeout)]
        else:
            job_args = list(args.job_args)
        source, n = re.subn(r"^KAGGLE_ARGS = .*$", f"KAGGLE_ARGS = {job_args!r}", source, count=1, flags=re.M)
        assert n == 1
        with open(os.path.join(folder, args.script), 'w') as f:
            f.write(source)
        meta = {
            'id': kernel_id(args.gpu, args.script),
            'title': job_name(args.gpu, args.script),
            'code_file': args.script,
            'language': 'python',
            'kernel_type': 'script',
            'is_private': True,
            'enable_gpu': True,
            'enable_internet': True,
        }
        with open(os.path.join(folder, 'kernel-metadata.json'), 'w') as f:
            json.dump(meta, f, indent=1)
        kaggle('kernels', 'push', '-p', folder, '--accelerator', ACCELERATORS[args.gpu],
               '--timeout', str(args.timeout))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('push', help='start a run')
    p.add_argument('--gpu', choices=ACCELERATORS, required=True)
    p.add_argument('--branch', default='gpu-port-sync')
    p.add_argument('--timeout', type=int, default=3 * 3600, help='maximum run time in seconds')
    p.add_argument('--quick', action='store_true', help='small comparison grids')
    p.add_argument('--skip-suite', action='store_true', help='skip the test suites')
    p.add_argument('--skip-compare', action='store_true', help='skip the CUDA vs C++ comparison grids')
    p.add_argument('--suites', default='cuda,cpp', help='comma-separated backends to run the test suite on')
    p.add_argument('--suite-timeout', type=int, default=5400, help='seconds per test-suite run')
    p.add_argument('--script', default=DEFAULT_SCRIPT, help='job script in tools/kaggle (default %(default)s)')
    p.add_argument('job_args', nargs='*', help='options for a --script job (after --)')
    for name, help_text in (('status', 'show the run status'), ('fetch', 'download the run outputs')):
        p = sub.add_parser(name, help=help_text)
        p.add_argument('--gpu', choices=ACCELERATORS, required=True)
        p.add_argument('--script', default=DEFAULT_SCRIPT)
        if name == 'fetch':
            p.add_argument('--dest', default=None, help='default: kaggle_results/<gpu>')
    sub.add_parser('quota', help="show the account's remaining weekly GPU quota")
    args = parser.parse_args()
    if args.command == 'push':
        push(args)
    elif args.command == 'status':
        kaggle('kernels', 'status', kernel_id(args.gpu, args.script))
    elif args.command == 'fetch':
        dest = args.dest or os.path.join('kaggle_results', args.gpu if args.script == DEFAULT_SCRIPT
                                         else job_name(args.gpu, args.script))
        os.makedirs(dest, exist_ok=True)
        kaggle('kernels', 'output', kernel_id(args.gpu, args.script), '-p', dest)
        summary = os.path.join(dest, 'summary.json')
        if os.path.exists(summary):
            with open(summary) as f:
                used = json.load(f).get('gpu')
            wanted = {'t4': 'T4', 'p100': 'P100'}[args.gpu]
            print(f"GPU used: {used}" + ('' if used and wanted in used else f"  (WARNING: requested {wanted})"))
    else:
        kaggle('quota')


if __name__ == '__main__':
    main()
