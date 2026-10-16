"""Tiled dense grids (pycalphad.densegrid)."""
import json
import os
import time
from importlib.resources import files

import numpy as np
import pytest

pytest.importorskip('zarr')
pytest.importorskip('pycalphad.cg._core')

import pycalphad.tests.databases as dbs
from pycalphad import Database, equilibrium, variables as v
from pycalphad.backend import backend
from pycalphad.densegrid import GridSpec, Run, create, run, open_dataset
from pycalphad.densegrid.spec import TARGET_TILE_POINTS

ALNI = str(files(dbs) / 'alni_dupin_2001.tdb')


def alni_spec(**kw):
    args = dict(database=ALNI, components=['AL', 'NI', 'VA'],
                axes={'T': [900, 1300, 1700], 'X(AL)': {'start': 0.05, 'stop': 0.95, 'num': 19}},
                tile={'T': 2, 'X(AL)': 8}, outputs=['GM', 'MU', 'NP', 'Phase', 'X', 'Y'])
    args.update(kw)
    return GridSpec(**args)


def direct(r, tile):
    """The tile solved by one equilibrium() call."""
    sl = r.tile_slices(tile)
    with backend('cg', threads=1):
        return equilibrium(Database(ALNI), ['AL', 'NI', 'VA'], r.meta['phases'],
                           {v.N: 1, v.P: 101325, v.T: list(r.spec.values('T')[sl[0]]),
                            v.X('AL'): list(r.spec.values('X(AL)')[sl[1]])})


def test_default_tile_sizes():
    "One value of a state variable per tile; the other axes grown within TARGET_TILE_POINTS."
    spec = GridSpec(database=ALNI, components=['AL', 'NI', 'VA'],
                    axes={'T': [800, 900], 'X(AL)': {'start': 0.0, 'stop': 1.0, 'num': 1001}})
    assert spec.tile == {'T': 1, 'X(AL)': 1001}
    spec = GridSpec(database=ALNI, components=['AL', 'CR', 'NI', 'VA'],
                    axes={'T': [800, 900], 'X(AL)': {'start': 0.0, 'stop': 1.0, 'step': 0.001},
                          'X(CR)': {'start': 0.0, 'stop': 1.0, 'step': 0.001}})
    assert spec.tile['T'] == 1 and spec.tile['X(AL)'] * spec.tile['X(CR)'] <= TARGET_TILE_POINTS
    assert abs(spec.tile['X(AL)'] - spec.tile['X(CR)']) <= 1
    assert spec.fixed == {'N': 1.0, 'P': 101325.0}


def test_spec_rejects_bad_input():
    with pytest.raises(ValueError):
        alni_spec(fixed={'T': 1000})          # T both on an axis and fixed
    with pytest.raises(ValueError):
        alni_spec(outputs=['GM', 'CPM'])      # not an output of the driver
    with pytest.raises(ValueError):
        alni_spec(tile={'X(NI)': 4})          # tile size for a condition that is not an axis


def test_spec_json_round_trip():
    spec = alni_spec()
    again = GridSpec.from_json(spec.to_json())
    assert again.to_json() == spec.to_json()
    np.testing.assert_array_equal(again.values('X(AL)'), spec.values('X(AL)'))


def test_tiles_match_direct_equilibrium(tmp_path):
    "Each tile holds exactly what one equilibrium() call over its conditions returns, in the spec's axis order."
    r = create(alni_spec(), tmp_path / 'run')
    assert r.n_tiles == 2 * 3
    status = run(r.path, workers=0, progress=None)
    assert status['done'] == r.n_tiles and status['failed'] == 0 and status['points_done'] == 3 * 19
    ds = open_dataset(r.path)
    assert ds.GM.dims == ('T', 'X_AL') and ds.X.dims == ('T', 'X_AL', 'vertex', 'component')
    assert list(ds.component.values) == ['AL', 'NI']
    assert (ds.status.values == 0).all()
    for tile in range(r.n_tiles):
        sl = r.tile_slices(tile)
        eq = direct(r, tile)
        np.testing.assert_array_equal(ds.GM.values[sl], eq.GM.values.squeeze(axis=(0, 1)))
        np.testing.assert_array_equal(ds.MU.values[sl], eq.MU.values.squeeze(axis=(0, 1)))
        names = np.array(ds.attrs['phases'] + [''])[ds.Phase.values[sl]]
        np.testing.assert_array_equal(names, eq.Phase.values.squeeze(axis=(0, 1)))
        D = eq.Y.shape[-1]
        np.testing.assert_array_equal(ds.Y.values[sl][..., :D], eq.Y.values.squeeze(axis=(0, 1)))


def test_run_resumes_where_it_stopped(tmp_path):
    "A run stopped after some tiles continues with the others; the results are those of an uninterrupted run."
    r = create(alni_spec(), tmp_path / 'a')
    status = run(r.path, workers=0, max_tiles=2, progress=None)
    assert status['done'] == 2 and status['pending'] == r.n_tiles - 2
    ds = open_dataset(r.path)
    assert (ds.status.values == -1).sum() > 0 and np.isnan(ds.GM.values[ds.status.values == -1]).all()
    status = run(r.path, workers=0, progress=None)
    assert status['done'] == r.n_tiles and status['pending'] == 0
    full = create(alni_spec(), tmp_path / 'b')
    run(full.path, workers=0, progress=None)
    for k in ('GM', 'MU', 'NP', 'Phase', 'X', 'status'):
        np.testing.assert_array_equal(open_dataset(r.path)[k].values, open_dataset(full.path)[k].values)


def test_stale_claims_are_taken_over(tmp_path):
    "A claim nobody refreshed within the lease (its worker died) is taken over; a fresh one is left alone."
    r = create(alni_spec(), tmp_path / 'run')
    for tile, age in ((0, 3600.0), (1, 0.0)):
        path = os.path.join(r.tiles_path, 'claims', str(tile))
        with open(path, 'w') as f:
            json.dump({'host': 'elsewhere', 'pid': 1, 'start': 0}, f)
        os.utime(path, (time.time() - age, time.time() - age))
    status = run(r.path, workers=0, lease=60.0, wait=False, progress=None)
    assert Run(r.path).is_finished(0) and not Run(r.path).is_finished(1)
    assert status['done'] == r.n_tiles - 1 and status['running'] == 1


def test_invalid_compositions_are_marked(tmp_path):
    "Points whose mole fractions sum to more than one are left empty with status 2."
    spec = GridSpec(database=str(files(dbs) / 'alcocrni.tdb'), components=['AL', 'CR', 'NI', 'VA'],
                    axes={'X(AL)': [0.1, 0.6], 'X(CR)': [0.1, 0.6]}, fixed={'T': 1300}, outputs=['GM', 'Phase'])
    r = create(spec, tmp_path / 'run')
    status = run(r.path, workers=0, progress=None)
    ds = open_dataset(r.path)
    np.testing.assert_array_equal(ds.status.values, [[0, 0], [0, 2]])
    assert np.isnan(ds.GM.values[1, 1]) and (ds.Phase.values[1, 1] == -1).all()
    assert status['points_done'] == 3


def test_worker_processes(tmp_path):
    "Worker processes give the same results as solving in this process (up to hash-order rounding)."
    a = create(alni_spec(), tmp_path / 'a')
    st = run(a.path, workers=2, progress=None)
    assert st['done'] == a.n_tiles and 'worker_exit_codes' not in st
    b = create(alni_spec(), tmp_path / 'b')
    run(b.path, workers=0, progress=None)
    da, db = open_dataset(a.path), open_dataset(b.path)
    np.testing.assert_allclose(da.GM.values, db.GM.values, atol=1e-6)
    np.testing.assert_array_equal(np.sort(np.where(da.NP.values > 1e-9, da.Phase.values, -1), -1),
                                  np.sort(np.where(db.NP.values > 1e-9, db.Phase.values, -1), -1))
    seeds = set()
    for name in os.listdir(os.path.join(a.tiles_path, 'done')):
        with open(os.path.join(a.tiles_path, 'done', name)) as f:
            seeds.add(json.load(f)['hash_seed'])
    assert seeds == {'0'}


def test_machines_start_with_their_own_share(tmp_path):
    "With node/nodes given, a machine's first tiles are its own share (tile number modulo the number of workers)."
    r = create(alni_spec(), tmp_path / 'run')
    run(r.path, workers=0, node=1, nodes=3, max_tiles=2, progress=None)
    assert Run(r.path).finished_ids() == {1, 4}
    status = run(r.path, workers=0, node=2, nodes=3, progress=None)   # its share, then whatever is left
    assert status['done'] == r.n_tiles
