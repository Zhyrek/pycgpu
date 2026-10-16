"""python -m pycalphad.densegrid create SPEC.json RUN | run RUN [--workers N] | status RUN"""
import argparse
import json
import sys

from . import create, run, status
from .worker import DEFAULT_LEASE


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m pycalphad.densegrid', description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('create', help='lay out a run directory from a grid spec (JSON)')
    c.add_argument('spec'); c.add_argument('run')
    c.add_argument('--overwrite', action='store_true', help='replace an existing run directory')
    r = sub.add_parser('run', help='solve pending tiles (start on each machine of a cluster; rerun to resume)')
    r.add_argument('run')
    r.add_argument('--workers', type=int, default=None,
                   help='worker processes on this machine (default: cores / solver threads; 0: in this process)')
    r.add_argument('--lease', type=float, default=DEFAULT_LEASE,
                   help='seconds without a heartbeat before a claimed tile is taken over (default %(default)s)')
    r.add_argument('--max-tiles', type=int, default=None, help='stop each worker after this many tiles')
    r.add_argument('--retry-failed', action='store_true', help='solve failed tiles again')
    r.add_argument('--node', type=int, default=None,
                   help="this machine's index among the machines working on the run (default: SLURM_NODEID, or 0)")
    r.add_argument('--nodes', type=int, default=None,
                   help='number of machines working on the run (default: SLURM_JOB_NUM_NODES, or 1)')
    r.add_argument('--no-wait', action='store_true',
                   help='exit when no tile is free to claim, instead of watching tiles other workers hold')
    s = sub.add_parser('status', help='progress of a run')
    s.add_argument('run')
    a = p.parse_args(argv)
    if a.command == 'create':
        rr = create(a.spec, a.run, overwrite=a.overwrite)
        print(f"{rr.path}: grid {tuple(rr.meta['shape'])} = {int(__import__('math').prod(rr.meta['shape']))} points, "
              f"{rr.n_tiles} tiles of {tuple(rr.meta['tile_shape'])}")
    elif a.command == 'run':
        st = run(a.run, workers=a.workers, lease=a.lease, max_tiles=a.max_tiles, retry_failed=a.retry_failed,
                 wait=not a.no_wait, node=a.node, nodes=a.nodes)
        print(json.dumps(st, indent=1))
        unfinished = st['pending'] or st['running']
        if st['failed'] or (unfinished and a.max_tiles is None and not a.no_wait):
            return 1
    else:
        print(json.dumps(status(a.run), indent=1))


if __name__ == '__main__':
    sys.exit(main())
