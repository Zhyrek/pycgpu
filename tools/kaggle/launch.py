"""Run tools/kaggle/run_gpu_checks.py on a Kaggle GPU and fetch its results.

Needs the Kaggle CLI (pip install kaggle) and an API token in
~/.kaggle/kaggle.json (kaggle.com -> Settings -> API -> Create New Token),
on an account with a verified phone number (required for GPUs and internet
access). Runs count against the account's free weekly GPU quota; nothing is
billed.

    python tools/kaggle/launch.py push --gpu t4 [--branch NAME] [--timeout 10800]
    python tools/kaggle/launch.py status --gpu t4
    python tools/kaggle/launch.py fetch --gpu t4 [--dest DIR]
    python tools/kaggle/launch.py quota

Each GPU type gets its own private job (<user>/pycgpu-gpu-checks-<gpu>), so
a T4 and a P100 run can be in flight together.
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
    token = os.path.expanduser(os.path.join(os.environ.get('KAGGLE_CONFIG_DIR', '~/.kaggle'), 'kaggle.json'))
    if not os.path.exists(token):
        sys.exit(f"no Kaggle API token at {token} (kaggle.com -> Settings -> API -> Create New Token)")
    with open(token) as f:
        return json.load(f)['username']


def kernel_id(gpu):
    return f"{username()}/pycgpu-gpu-checks-{gpu}"


def push(args):
    with tempfile.TemporaryDirectory() as folder:
        with open(os.path.join(HERE, 'run_gpu_checks.py')) as f:
            source = f.read()
        source, n = re.subn(r"^BRANCH = .*$", f"BRANCH = {args.branch!r}", source, count=1, flags=re.M)
        assert n == 1
        with open(os.path.join(folder, 'run_gpu_checks.py'), 'w') as f:
            f.write(source)
        meta = {
            'id': kernel_id(args.gpu),
            'title': f'pycgpu-gpu-checks-{args.gpu}',
            'code_file': 'run_gpu_checks.py',
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
    for name, help_text in (('status', 'show the run status'), ('fetch', 'download the run outputs')):
        p = sub.add_parser(name, help=help_text)
        p.add_argument('--gpu', choices=ACCELERATORS, required=True)
        if name == 'fetch':
            p.add_argument('--dest', default=None, help='default: kaggle_results/<gpu>')
    sub.add_parser('quota', help="show the account's remaining weekly GPU quota")
    args = parser.parse_args()
    if args.command == 'push':
        push(args)
    elif args.command == 'status':
        kaggle('kernels', 'status', kernel_id(args.gpu))
    elif args.command == 'fetch':
        dest = args.dest or os.path.join('kaggle_results', args.gpu)
        os.makedirs(dest, exist_ok=True)
        kaggle('kernels', 'output', kernel_id(args.gpu), '-p', dest)
    else:
        kaggle('quota')


if __name__ == '__main__':
    main()
