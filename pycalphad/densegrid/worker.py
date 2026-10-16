"""Worker processes: each builds the system once, then claims tiles, solves them and writes their results.

Workers coordinate only through the run directory (atomic claim files), so the same run can be worked on by
processes on several machines sharing a filesystem: start `python -m pycalphad.densegrid run RUN` on each. A tile
is always solved from scratch (fresh seeding), so its results depend on the tile layout but not on which worker,
machine or order it was solved in.
"""
import logging
import multiprocessing as mp
import os
import random
import socket
import threading
import time
import traceback

import numpy as np

from .store import Run, system_layout, TRAILING, STATUS_SOLVED, STATUS_UNSOLVED, STATUS_INVALID

DEFAULT_LEASE = 300.0   # seconds without a heartbeat after which another worker takes a claimed tile over
IDLE_POLL = 2.0         # seconds between checks of the tiles other workers hold, once none is free (doubling
MAX_IDLE_POLL = 60.0    #   while idle, up to this)
# libraries that start their own thread pools: one thread each per worker process unless the user says otherwise
_THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
               'NUMEXPR_NUM_THREADS')

log = logging.getLogger('pycalphad.densegrid')


class TileSolver:
    """Solves the conditions of one tile with pycalphad.equilibrium on models and phase records built once.
    (The interface a different engine for tiles would provide: solve(slices) -> (arrays, info).)"""

    def __init__(self, run):
        from pycalphad import Database
        from pycalphad.property_framework import as_property
        self.run = run
        spec = run.spec
        self.dbf = Database(os.path.join(run.path, 'database.tdb'))
        layout, self.models, self.prf = system_layout(spec, self.dbf, phases=run.meta['phases'])
        for key in ('elements', 'phases', 'vertex', 'internal_dof'):
            if layout[key] != run.meta[key]:
                raise RuntimeError(f'{key} differs from the run metadata: {layout[key]} vs {run.meta[key]}')
        self.phases = layout['phases']
        self.phase_index = {name: i for i, name in enumerate(self.phases)}
        self.sizes = {'component': len(layout['elements']), 'vertex': layout['vertex'],
                      'internal_dof': layout['internal_dof']}
        self.fixed = {as_property(k): float(val) for k, val in spec.fixed.items()}
        self.axes = [as_property(a) for a in spec.axis_names]
        self.axis_dims = spec.dims
        from pycalphad import variables as v
        self._global_x = lambda c: isinstance(c, v.MoleFraction) and c.phase_name is None

    def warm_up(self):
        """Solve one valid point (compiles what the backend compiles on first use, so tile timings are steady
        state). Returns the seconds taken."""
        from pycalphad import equilibrium
        t0 = time.perf_counter()
        for tile in range(self.run.n_tiles):
            slices = self.run.tile_slices(tile)
            invalid = self.invalid_mask(slices)
            if not invalid.all():
                first = np.unravel_index(int(np.argmin(invalid)), invalid.shape)
                one = tuple(slice(sl.start + i, sl.start + i + 1) for sl, i in zip(slices, first))
                with self._backend():
                    equilibrium(self.dbf, self.run.spec.components, self.phases, self.conditions(one),
                                model=self.models, phase_records=self.prf, to_xarray=False)
                break
        return time.perf_counter() - t0

    def _backend(self):
        """The spec's backend for the calls in a with block (the caller's global backend is left alone)."""
        from pycalphad.backend import backend
        return backend(self.run.spec.backend, threads=self.run.spec.threads)

    def conditions(self, slices):
        conds = dict(self.fixed)
        for axis, name, sl in zip(self.axes, self.run.spec.axis_names, slices):
            conds[axis] = [float(x) for x in self.run.spec.values(name)[sl]]
        return conds

    def invalid_mask(self, slices):
        """Points whose (global) mole-fraction conditions sum to more than one."""
        shape = tuple(sl.stop - sl.start for sl in slices)
        total = np.full(shape, sum(val for c, val in self.fixed.items() if self._global_x(c)))
        for k, (axis, name, sl) in enumerate(zip(self.axes, self.run.spec.axis_names, slices)):
            if self._global_x(axis):
                vals = self.run.spec.values(name)[sl]
                total = total + vals.reshape([-1 if i == k else 1 for i in range(len(shape))])
        return total > 1

    def solve(self, slices):
        from pycalphad import equilibrium
        conds = self.conditions(slices)
        t0, c0 = time.perf_counter(), time.process_time()
        with self._backend():
            eq = equilibrium(self.dbf, self.run.spec.components, self.phases, conds, model=self.models,
                             phase_records=self.prf, to_xarray=False)
        seconds, cpu_seconds = time.perf_counter() - t0, time.process_time() - c0
        components = [str(c) for c in np.asarray(eq.coords['component'])]
        if components != self.run.meta['elements']:
            raise RuntimeError(f'component order {components} differs from {self.run.meta["elements"]}')
        out = {}
        for name in self.run.spec.outputs:
            dims, values = eq.data_vars[name]
            out[name] = self._arrange(name, list(dims), np.asarray(values))
        GM = out['GM'] if 'GM' in out else self._arrange('GM', *self._var(eq, 'GM'))
        invalid = self.invalid_mask(slices)
        status = np.where(np.isfinite(GM), STATUS_SOLVED, np.where(invalid, STATUS_INVALID, STATUS_UNSOLVED))
        out['status'] = status.astype(np.int8)
        info = {'points': int((status != STATUS_INVALID).sum()), 'unsolved': int((status == STATUS_UNSOLVED).sum()),
                'invalid': int((status == STATUS_INVALID).sum()), 'seconds': seconds, 'cpu_seconds': cpu_seconds}
        return out, info

    @staticmethod
    def _var(eq, name):
        dims, values = eq.data_vars[name]
        return list(dims), np.asarray(values)

    def _arrange(self, name, dims, values):
        """An output in the run's layout: the tile's axes in spec order, then the trailing dimensions at the run's
        sizes; other (single-valued) condition dimensions dropped; phase names as indices."""
        trailing = TRAILING[name]
        index, kept = [], []
        for i, d in enumerate(dims):
            if d in self.axis_dims or d in trailing:
                index.append(slice(None)); kept.append(d)
            elif values.shape[i] == 1:
                index.append(0)
            else:
                raise RuntimeError(f'{name}: unexpected dimension {d} of size {values.shape[i]}')
        values = values[tuple(index)]
        values = np.transpose(values, [kept.index(d) for d in list(self.axis_dims) + list(trailing)])
        if name == 'Phase':
            lookup = np.vectorize(lambda p: self.phase_index.get(str(p), -1), otypes=[np.int16])
            values = lookup(values) if values.size else values.astype(np.int16)
        else:
            values = values.astype(np.float64)
        # trailing sizes: pad with empty entries, or drop entries that are all empty
        nax = len(self.axis_dims)
        for k, t in enumerate(trailing):
            axis, have, want = nax + k, values.shape[nax + k], self.sizes[t]
            if have < want:
                pad = [(0, 0)] * values.ndim; pad[axis] = (0, want - have)
                values = np.pad(values, pad, constant_values=-1 if name == 'Phase' else np.nan)
            elif have > want:
                extra = np.take(values, range(want, have), axis=axis)
                if not (np.all(extra == -1) if name == 'Phase' else np.all(np.isnan(extra))):
                    raise RuntimeError(f'{name}: more {t} entries than the run layout holds ({have} > {want})')
                values = np.take(values, range(want), axis=axis)
        return values


class _Heartbeat:
    """Keeps the claim of the tile being worked on fresh."""

    def __init__(self, run, interval):
        self.run, self.interval = run, interval
        self.tile = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.wait(self.interval):
            tile = self.tile
            if tile is not None:
                self.run.touch(tile)

    def stop(self):
        self._stop.set()


def worker_main(run_path, rank=0, nranks=1, lease=DEFAULT_LEASE, max_tiles=None, wait=True):
    """Claim and solve tiles: first this worker's share (every nranks-th tile from rank: no contention for claims),
    then any tile nobody has finished or holds, until every tile is finished (or this worker has done
    `max_tiles`). With `wait`, keep watching the tiles other workers hold, to take them over if those workers stop.
    Tiles are found from directory listings, not one file check per tile, so the cost on a shared filesystem stays
    small with many tiles and workers."""
    run = Run(run_path)
    host = socket.gethostname()
    handler = logging.FileHandler(os.path.join(run.path, 'logs', f'{host}-{os.getpid()}.log'))
    handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    log.addHandler(handler); log.setLevel(logging.INFO)
    t0 = time.perf_counter()
    solver = TileSolver(run)
    warm = solver.warm_up()
    root = run.open_zarr('r+')
    log.info('worker %d of %d ready in %.1f s (first solve %.1f s; %s, %d tiles)', rank, nranks,
             time.perf_counter() - t0, warm, run.spec.backend, run.n_tiles)
    beat = _Heartbeat(run, max(1.0, min(30.0, lease / 4)) if lease else 30.0)
    rng = random.Random(f'{host}-{rank}-{os.getpid()}')
    count = 0

    def solve(tile):
        beat.tile = tile
        slices = run.tile_slices(tile)
        try:
            arrays, info = solver.solve(slices)
            for name, values in arrays.items():
                root[name][slices + (slice(None),) * len(TRAILING[name])] = values
            info.update(host=host, pid=os.getpid(), finished=time.time(), hash_seed=os.environ.get('PYTHONHASHSEED'))
            run.mark_done(tile, info)
            log.info('tile %d: %d points in %.2f s, %.2f s CPU (%.3f CPU-ms/point), %d unsolved', tile,
                     info['points'], info['seconds'], info['cpu_seconds'],
                     1e3 * info['cpu_seconds'] / max(info['points'], 1), info['unsolved'])
        except Exception:
            run.mark_failed(tile, {'host': host, 'pid': os.getpid(), 'traceback': traceback.format_exc()})
            log.error('tile %d failed:\n%s', tile, traceback.format_exc())
        finally:
            beat.tile = None

    try:
        finished = run.finished_ids()
        for tile in range(rank % max(nranks, 1), run.n_tiles, max(nranks, 1)):
            if tile in finished or not run.claim(tile, lease):
                continue
            solve(tile)
            count += 1
            if max_tiles is not None and count >= max_tiles:
                return count
        poll = IDLE_POLL
        while True:
            unfinished = set(range(run.n_tiles)) - run.finished_ids()
            if not unfinished:
                return count
            held = run.claimed_ids()
            # free tiles first; then held ones (claim() takes over a claim only once it is stale)
            candidates = sorted(unfinished - held)
            rng.shuffle(candidates)
            candidates += sorted(unfinished & held)
            for tile in candidates:
                if run.claim(tile, lease):
                    solve(tile)
                    count += 1
                    if max_tiles is not None and count >= max_tiles:
                        return count
                    poll = IDLE_POLL
                    break
            else:
                if not wait:
                    return count
                # every unfinished tile is held by a live worker: check again later, less often while idle
                time.sleep(poll)
                poll = min(2 * poll, MAX_IDLE_POLL)
    finally:
        beat.stop()
        log.removeHandler(handler); handler.close()


def _cluster_position():
    """(node index, number of nodes) from the batch system's environment, if any (SLURM), else (0, 1)."""
    env = os.environ
    nodes = env.get('SLURM_JOB_NUM_NODES') or env.get('SLURM_NNODES')
    if nodes and env.get('SLURM_NODEID') is not None:
        return int(env['SLURM_NODEID']), int(nodes)
    return 0, 1


def run(run_path, workers=None, lease=DEFAULT_LEASE, max_tiles=None, retry_failed=False, wait=True,
        progress=print, progress_interval=30.0, node=None, nodes=None):
    """Work on a run with `workers` processes on this machine (default: one per core, divided by the spec's solver
    threads; 0: in this process). On a cluster, start this on every machine; `node` (0 .. nodes - 1) and `nodes`
    give each machine's workers their own share of the tiles first (from SLURM's environment when not given; any
    values are correct, they only spread the first claims). Returns the run's status afterwards."""
    r = Run(run_path)
    env_node, env_nodes = _cluster_position()
    node = env_node if node is None else int(node)
    nodes = env_nodes if nodes is None else int(nodes)
    if retry_failed:
        r.clear_failed()
    if workers is None:
        workers = max(1, (os.cpu_count() or 1) // max(1, r.spec.threads))
    if workers == 0:
        worker_main(r.path, node, nodes, lease, max_tiles, wait)
        return r.status()
    for k in _THREAD_ENV:
        os.environ.setdefault(k, '1')
    # pycalphad's symbolic terms are summed in hash order: one hash seed for every worker (on every machine, as
    # each machine's launcher sets the same default) keeps tiles bit-for-bit independent of the process solving them
    os.environ.setdefault('PYTHONHASHSEED', '0')
    ctx = mp.get_context('spawn')
    procs = [ctx.Process(target=worker_main, args=(r.path, node * workers + i, nodes * workers, lease, max_tiles,
                                                   wait), daemon=False)
             for i in range(workers)]
    for p in procs:
        p.start()
    t0 = time.time()
    while any(p.is_alive() for p in procs):
        for p in procs:
            p.join(timeout=progress_interval / max(len(procs), 1))
        if progress is not None:
            s = r.status()
            rate = f", {s['ms_per_point']:.3f} worker-ms/point" if s['ms_per_point'] else ''
            progress(f"[{time.time() - t0:7.0f} s] tiles done {s['done']}/{s['tiles']}, running {s['running']}, "
                     f"failed {s['failed']}; {s['points_done']} points{rate}")
    status = r.status()
    bad = [p.exitcode for p in procs if p.exitcode]
    if bad:
        # (their tiles are taken over by the others once the claims go stale; what is left shows in the status)
        status['worker_exit_codes'] = bad
        if progress is not None:
            progress(f'{len(bad)} worker process(es) exited with errors {bad}; see {r.path}/logs')
    return status
