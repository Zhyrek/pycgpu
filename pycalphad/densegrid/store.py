"""The run directory: the spec, a copy of the database, the results (a Zarr store with one chunk per tile) and the
tile bookkeeping.

    RUN/spec.json             the GridSpec
    RUN/meta.json             element and phase order, array sizes, versions (written once, at creation)
    RUN/database.tdb          the database every worker reads
    RUN/grid.zarr/            results: one array per output over (axes..., trailing dims), chunked by tile
    RUN/tiles/claims/<id>     a worker is on the tile (host, pid, start time; touched while it works)
    RUN/tiles/done/<id>       the tile's results are written (JSON: timing and counts)
    RUN/tiles/failed/<id>     the tile raised (traceback); not retried unless asked
    RUN/logs/                 one log per worker process

Arrays: GM (axes), MU (axes, component), NP and Phase (axes, vertex), X (axes, vertex, component), Y (axes, vertex,
internal_dof), status (axes). Phase holds indices into meta["phases"] (-1: no phase). status: -1 not computed yet,
0 solved, 1 not solved (NaN), 2 not a valid composition (mole fractions summing to more than one).
"""
import json
import math
import os
import shutil
import socket
import subprocess
import time

import numpy as np

from .spec import GridSpec

STATUS_PENDING, STATUS_SOLVED, STATUS_UNSOLVED, STATUS_INVALID = -1, 0, 1, 2
TRAILING = {'GM': (), 'MU': ('component',), 'NP': ('vertex',), 'Phase': ('vertex',),
            'X': ('vertex', 'component'), 'Y': ('vertex', 'internal_dof'), 'status': ()}


def _git_commit():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return subprocess.run(['git', '-C', here, 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:
        return None


def system_layout(spec, dbf, phases=None):
    """Element order, phases and array sizes for a spec (as the solver will see them), with the models and phase
    records."""
    from pycalphad.core.utils import filter_phases, unpack_species, instantiate_models
    from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
    from pycalphad import variables as v
    phases = phases or spec.phases or filter_phases(dbf, unpack_species(dbf, spec.components))
    phases = sorted(phases)
    models = instantiate_models(dbf, spec.components, phases)
    prf = PhaseRecordFactory(dbf, spec.components, {v.N, v.P, v.T}, models)
    elements = [str(e) for e in prf.nonvacant_elements]
    return {'elements': elements, 'phases': phases, 'vertex': len(elements) + 1,
            'internal_dof': max(len(m.site_fractions) for m in models.values())}, models, prf


class Run:
    """A run directory (see the module docstring)."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.spec = GridSpec.load(os.path.join(self.path, 'spec.json'))
        with open(os.path.join(self.path, 'meta.json')) as f:
            self.meta = json.load(f)
        self.zarr_path = os.path.join(self.path, 'grid.zarr')
        self.tiles_path = os.path.join(self.path, 'tiles')

    # ---- creation
    @classmethod
    def create(cls, spec, path, overwrite=False):
        """Lay out a run directory for `spec` (a GridSpec, or the path of its JSON)."""
        import zarr
        from pycalphad import Database, __version__
        if not isinstance(spec, GridSpec):
            spec = GridSpec.load(spec)
        path = os.path.abspath(path)
        if os.path.exists(path):
            if not overwrite:
                raise FileExistsError(f'{path} exists (pass overwrite=True to replace it)')
            shutil.rmtree(path)
        os.makedirs(path)
        shutil.copyfile(spec.database, os.path.join(path, 'database.tdb'))
        layout, _, _ = system_layout(spec, Database(os.path.join(path, 'database.tdb')))
        n_tiles = math.prod(math.ceil(n / t) for n, t in zip(spec.shape, spec.tile_shape))
        meta = dict(layout, shape=list(spec.shape), tile_shape=list(spec.tile_shape), n_tiles=n_tiles,
                    dims=spec.dims, pycalphad_version=__version__, git_commit=_git_commit(),
                    cg_environment={k: val for k, val in os.environ.items() if k.startswith('PYCALPHAD_CG_')},
                    created=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        with open(os.path.join(path, 'spec.json'), 'w') as f:
            f.write(spec.to_json())
        with open(os.path.join(path, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=1)
        for d in ('claims', 'done', 'failed'):
            os.makedirs(os.path.join(path, 'tiles', d))
        os.makedirs(os.path.join(path, 'logs'))

        root = zarr.open_group(os.path.join(path, 'grid.zarr'), mode='w')
        root.attrs.update({'phases': layout['phases'], 'elements': layout['elements'], 'dims': spec.dims})
        sizes = {'component': len(layout['elements']), 'vertex': layout['vertex'],
                 'internal_dof': layout['internal_dof']}
        for axis, dim in zip(spec.axis_names, spec.dims):
            vals = spec.values(axis)
            a = root.create_array(dim, shape=vals.shape, dtype='float64', chunks=vals.shape, dimension_names=[dim])
            a[:] = vals
        for name in list(spec.outputs) + ['status']:
            trailing = TRAILING[name]
            shape = tuple(spec.shape) + tuple(sizes[t] for t in trailing)
            chunks = tuple(spec.tile_shape) + tuple(sizes[t] for t in trailing)
            if name == 'Phase':
                dtype, fill = 'int16', -1
            elif name == 'status':
                dtype, fill = 'int8', STATUS_PENDING
            else:
                dtype, fill = 'float64', np.nan
            root.create_array(name, shape=shape, dtype=dtype, chunks=chunks, fill_value=fill,
                              dimension_names=list(spec.dims) + list(trailing))
        return cls(path)

    # ---- tiles
    @property
    def tile_grid(self):
        return tuple(math.ceil(n / t) for n, t in zip(self.meta['shape'], self.meta['tile_shape']))

    @property
    def n_tiles(self):
        return self.meta['n_tiles']

    def tile_slices(self, tile_id):
        idx = np.unravel_index(tile_id, self.tile_grid)
        return tuple(slice(i * t, min((i + 1) * t, n)) for i, t, n in zip(idx, self.meta['tile_shape'],
                                                                          self.meta['shape']))

    def _marker(self, kind, tile_id):
        return os.path.join(self.tiles_path, kind, str(tile_id))

    def is_finished(self, tile_id):
        return os.path.exists(self._marker('done', tile_id)) or os.path.exists(self._marker('failed', tile_id))

    def claim(self, tile_id, lease):
        """Claim a tile for this process: True if it is now ours. A claim untouched for more than `lease` seconds
        (its worker died) is taken over; only one worker can win the rename that removes it."""
        path = self._marker('claims', tile_id)
        for attempt in range(2):
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                if attempt or lease is None:
                    return False
                try:
                    age = time.time() - os.stat(path).st_mtime
                except FileNotFoundError:
                    continue
                if age <= lease:
                    return False
                try:
                    os.rename(path, f'{path}.stale.{socket.gethostname()}.{os.getpid()}.{time.time():.0f}')
                except FileNotFoundError:
                    return False
                continue
            with os.fdopen(fd, 'w') as f:
                json.dump({'host': socket.gethostname(), 'pid': os.getpid(), 'start': time.time()}, f)
            if self.is_finished(tile_id):   # finished between our check and the claim
                self.release(tile_id)
                return False
            return True
        return False

    def touch(self, tile_id):
        try:
            os.utime(self._marker('claims', tile_id))
        except FileNotFoundError:
            pass

    def release(self, tile_id):
        try:
            os.remove(self._marker('claims', tile_id))
        except FileNotFoundError:
            pass

    def _write_marker(self, kind, tile_id, info):
        path = self._marker(kind, tile_id)
        tmp = f'{path}.tmp.{os.getpid()}'
        with open(tmp, 'w') as f:
            json.dump(info, f)
        os.replace(tmp, path)

    def mark_done(self, tile_id, info):
        self._write_marker('done', tile_id, info)
        self.release(tile_id)

    def mark_failed(self, tile_id, info):
        self._write_marker('failed', tile_id, info)
        self.release(tile_id)

    def clear_failed(self):
        d = os.path.join(self.tiles_path, 'failed')
        for name in os.listdir(d):
            os.remove(os.path.join(d, name))

    def _ids(self, kind):
        return {int(n) for n in os.listdir(os.path.join(self.tiles_path, kind)) if n.isdigit()}

    def finished_ids(self):
        """Tiles done or failed (two directory listings)."""
        return self._ids('done') | self._ids('failed')

    def claimed_ids(self):
        return self._ids('claims')

    def status(self):
        """Counts of tiles by state, and totals from the done markers."""
        done, failed, claims = self._ids('done'), self._ids('failed'), self._ids('claims')
        infos = []
        for t in done:
            try:
                with open(self._marker('done', t)) as f:
                    infos.append(json.load(f))
            except (OSError, ValueError):
                pass
        points = sum(i.get('points', 0) for i in infos)
        seconds = sum(i.get('seconds', 0.0) for i in infos)
        cpu = sum(i.get('cpu_seconds', 0.0) for i in infos)
        return {'tiles': self.n_tiles, 'done': len(done), 'failed': len(failed),
                'running': len(claims - done - failed),
                'pending': self.n_tiles - len(done | failed | claims),
                'points_done': points, 'unsolved_points': sum(i.get('unsolved', 0) for i in infos),
                'worker_seconds': seconds, 'worker_cpu_seconds': cpu,
                'ms_per_point': 1e3 * seconds / points if points else None,
                'cpu_ms_per_point': 1e3 * cpu / points if points else None}

    # ---- results
    def open_zarr(self, mode='r'):
        import zarr
        return zarr.open_group(self.zarr_path, mode=mode)


def open_dataset(path):
    """The results of a run as a lazy xarray Dataset (dask-backed where dask is installed). Phase holds indices into
    ds.attrs['phases']; status says which points are computed (see the module docstring)."""
    import xarray as xr
    run = Run(path)
    ds = xr.open_zarr(run.zarr_path, consolidated=False)
    coords = {'component': run.meta['elements'], 'vertex': np.arange(run.meta['vertex'])}
    if 'internal_dof' in ds.dims:
        coords['internal_dof'] = np.arange(run.meta['internal_dof'])
    ds = ds.assign_coords({k: val for k, val in coords.items() if k in ds.dims})
    ds.attrs.update({'phases': run.meta['phases'], 'spec': run.spec.to_json()})
    return ds
