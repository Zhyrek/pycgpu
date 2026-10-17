"""Column-generation backend internals."""
import numpy as np
import pytest

solve = pytest.importorskip('pycalphad.cg.solve')


def test_consolidate_keeps_mass_balance_and_energy():
    "Merging two points of one phase within COMP_DIFFERENCE_TOL keeps the overall composition and GM."
    # two FCC points on one tangent plane (G = mu . X) a little apart, and a second phase
    mu = np.array([-30000.0, -60000.0])
    XA, XB, XC = np.array([0.90, 0.10]), np.array([0.90004, 0.09996]), np.array([0.3, 0.7])
    result = ([0, 0, 1], [np.array([0.9, 0.1]), np.array([0.90004, 0.09996]), np.array([0.3, 0.7])],
              [XA, XB, XC], [0.4, 0.3, 0.3], [mu @ XA, mu @ XB, mu @ XC])
    z = 0.4 * XA + 0.3 * XB + 0.3 * XC
    gm = (0.4 * mu @ XA + 0.3 * mu @ XB + 0.3 * mu @ XC)
    phs, Y, X, n, G = solve._consolidate(result)
    assert phs == [0, 1]
    np.testing.assert_allclose(sum(a * x for a, x in zip(n, X)), z, atol=1e-15)
    assert abs(sum(a * g for a, g in zip(n, G)) - gm) < 1e-9


def test_tile_stages_start_from_earlier_stages():
    "Stages by the coarsest lattice holding a condition; each condition starts from the nearest one solved before."
    import itertools
    idx = list(itertools.product(range(9), range(9)))
    stage, init = solve._tile_stages(idx, 2)
    assert stage[0] == 0 and (stage[1:] >= 1).all()
    assert np.bincount(stage).tolist() == [1, 8, 16, 56]
    assert (stage[init[1:]] < stage[1:]).all()
    assert idx[init[idx.index((2, 3))]] == (2, 2)
    stage, init = solve._tile_stages(idx, 0)
    assert stage[0] == 0 and (stage[1:] == 1).all() and (init == 0).all()


@pytest.mark.parametrize('levels', [0, 3])
def test_lockstep_tile_matches_sequential(monkeypatch, levels):
    "Lockstep rounds over a shared point store give the sequential results (phase sets and GM within tolerance)."
    from pycalphad import Database, equilibrium, variables as v
    from pycalphad.backend import backend
    from importlib.resources import files
    import pycalphad.tests.databases as dbs
    dbf = Database(str(files(dbs) / 'alcocrni.tdb'))
    comps = ['AL', 'CR', 'NI', 'VA']
    phases = list(dbf.phases)
    conds = {v.N: 1, v.P: 101325, v.T: 1300, v.X('AL'): list(np.linspace(0.02, 0.5, 40)),
             v.X('CR'): list(np.linspace(0.02, 0.5, 40))}
    out = {}
    with backend('cg', threads=1):
        for mode in (False, True):
            monkeypatch.setattr(solve, 'TILE_MODE', mode)
            monkeypatch.setattr(solve, 'TILE_LEVELS', levels)
            eq = equilibrium(dbf, comps, phases, conds)
            NP = eq.NP.values.reshape(-1, eq.NP.shape[-1]); Ph = eq.Phase.values.reshape(NP.shape)
            sets = ['+'.join(sorted(p for p, n in zip(Ph[i], NP[i]) if p and n > 1e-6)) for i in range(len(NP))]
            out[mode] = (eq.GM.values.ravel(), sets)
    np.testing.assert_allclose(out[True][0], out[False][0], atol=1e-3)
    assert out[True][1] == out[False][1]


def test_batched_pricing_is_bit_identical_to_lockstep(monkeypatch):
    "Pricing as batches of independent jobs per sub-round changes nothing (same arithmetic in the same order)."
    from pycalphad import Database, equilibrium, variables as v
    from pycalphad.backend import backend
    from importlib.resources import files
    import pycalphad.tests.databases as dbs
    dbf = Database(str(files(dbs) / 'alcocrni.tdb'))
    conds = {v.N: 1, v.P: 101325, v.T: 1300, v.X('AL'): list(np.linspace(0.02, 0.5, 40)),
             v.X('CR'): list(np.linspace(0.02, 0.5, 40))}
    out = {}
    monkeypatch.setattr(solve, 'DEVICE', '')   # (the compiled pricer agrees to rounding, not bit for bit)
    with backend('cg', threads=1):
        for mode in (1, 2):
            monkeypatch.setattr(solve, 'TILE_MODE', mode)
            out[mode] = equilibrium(dbf, ['AL', 'CR', 'NI', 'VA'], list(dbf.phases), conds)
    for k in ('GM', 'MU', 'NP', 'X', 'Y'):
        np.testing.assert_array_equal(out[1][k].values, out[2][k].values)
    np.testing.assert_array_equal(out[1].Phase.values, out[2].Phase.values)


def test_compiled_pricer_matches_cython():
    "cg_price.h compiled for the host prices like CPhase.price (to rounding) from random starts and potentials."
    import shutil
    if not (shutil.which('clang++') or shutil.which('g++')):
        pytest.skip('no C++ compiler')
    import ctypes
    from pycalphad import Database, Workspace, variables as v
    from importlib.resources import files
    import pycalphad.tests.databases as dbs
    from pycalphad.cg.device import HostPricer, describe
    dbf = Database(str(files(dbs) / 'alni_dupin_2001.tdb'))
    comps = ['AL', 'NI', 'VA']
    wks = Workspace(dbf, comps, list(dbf.phases), {v.N: 1, v.P: 101325, v.T: 1200, v.X('AL'): 0.3})
    prf = wks.phase_record_factory
    elements = [str(e) for e in prf.nonvacant_elements]
    phases = solve._phases_for(prf, elements)
    sv = np.array([1.0, 101325.0, 1200.0])
    for ph in phases:
        ph.core.set_state(sv, np.asarray(prf.param_values, dtype=float))
    pricer = HostPricer(wks, phases, elements)
    cores = [ph.core for ph in phases]
    ok, di, dd, ip, dp = describe(cores, pricer.names, pricer.index, pricer.deriv, elements)
    assert ok.sum() == sum(not ph.stoichiometric for ph in phases)
    rng = np.random.default_rng(0)
    RT, C, Dmax = cores[0].RT, len(elements), max(ph.D for ph in phases)
    P = ctypes.c_void_p
    for p, ph in enumerate(phases):
        if not ok[p]:
            continue
        n = 20
        Y0 = np.zeros((n, Dmax)); MU = (-30000 + 20000 * rng.standard_normal((n, C))) / RT
        for k in range(n):
            y = rng.random(ph.D) + 1e-3
            for blk in ph.blocks:
                y[blk] /= y[blk].sum()
            Y0[k, :ph.D] = y
        jobs = np.arange(n, dtype=np.int32); jp = np.full(n, p, dtype=np.int32)
        yo, xo, Go, to = np.zeros((n, Dmax)), np.zeros((n, C)), np.zeros(n), np.zeros(n)
        co, ev = np.zeros(n, dtype=np.int32), np.zeros(2 * n, dtype=np.int32)
        args = [jobs, jp, jobs, MU, Y0, None, yo, xo, Go, to, co, ev, di, dd, ip, dp]
        pricer.lib.cg_price_batch(ctypes.c_int(n), *[ctypes.c_int(Dmax) if a is None else a.ctypes.data_as(P) for a in args],
                                  ctypes.c_int(C), ctypes.c_double(RT), sv.ctypes.data_as(P))
        for k in range(n):
            tpd, y, x, G, conv = ph.core.py_price(MU[k] * RT, Y0[k, :ph.D])
            assert abs(tpd / RT - to[k]) < 1e-11
            np.testing.assert_allclose(yo[k, :ph.D], y, atol=1e-10)
            assert conv == bool(co[k])


def test_compiled_tile_matches_batched_tile(monkeypatch):
    "A whole tile solved by the compiled per-condition steps (cg_tile.h) gives the Cython tile's phase sets and GM."
    import shutil
    if not (shutil.which('clang++') or shutil.which('g++')):
        pytest.skip('no C++ compiler')
    from pycalphad import Database, equilibrium, variables as v
    from pycalphad.backend import backend
    from importlib.resources import files
    import pycalphad.tests.databases as dbs
    dbf = Database(str(files(dbs) / 'alcocrni.tdb'))
    conds = {v.N: 1, v.P: 101325, v.T: 1300, v.X('AL'): list(np.linspace(0.02, 0.5, 30)),
             v.X('CR'): list(np.linspace(0.02, 0.5, 30))}
    out = {}
    monkeypatch.setattr(solve, 'DEVICE', '')
    with backend('cg', threads=1):
        for mode in (2, 3):
            monkeypatch.setattr(solve, 'TILE_MODE', mode)
            eq = equilibrium(dbf, ['AL', 'CR', 'NI', 'VA'], list(dbf.phases), conds)
            NP = eq.NP.values.reshape(-1, eq.NP.shape[-1]); Ph = eq.Phase.values.reshape(NP.shape)
            out[mode] = (eq.GM.values.ravel(),
                         ['+'.join(sorted(p for p, n in zip(Ph[i], NP[i]) if p and n > 1e-6)) for i in range(len(NP))])
    np.testing.assert_allclose(out[3][0], out[2][0], atol=1e-4)
    assert out[3][1] == out[2][1]


def test_device_tile_on_host_matches_host_tile(monkeypatch):
    "The GPU tile driver (cg_tile_device.h's kernels, built for the host) gives the host driver's results and work."
    import shutil
    if not (shutil.which('clang++') or shutil.which('g++')):
        pytest.skip('no C++ compiler')
    from pycalphad import Database, equilibrium, variables as v
    from pycalphad.backend import backend
    from importlib.resources import files
    import pycalphad.tests.databases as dbs
    dbf = Database(str(files(dbs) / 'alcocrni.tdb'))
    conds = {v.N: 1, v.P: 101325, v.T: 1300, v.X('AL'): list(np.linspace(0.02, 0.5, 30)),
             v.X('CR'): list(np.linspace(0.02, 0.5, 30))}
    out = {}
    monkeypatch.setattr(solve, 'TILE_MODE', 3)
    with backend('cg', threads=1):
        for device in ('', 'cuda-on-host'):
            monkeypatch.setattr(solve, 'DEVICE', device)
            eq = equilibrium(dbf, ['AL', 'CR', 'NI', 'VA'], list(dbf.phases), conds)
            NP = eq.NP.values.reshape(-1, eq.NP.shape[-1]); Ph = eq.Phase.values.reshape(NP.shape)
            pricer = [p for (d, *_), p in solve._PRICERS.items() if d == (device or 'host')][-1]
            out[device] = (eq.GM.values.ravel(), pricer.tile_counters,
                           ['+'.join(sorted(p for p, n in zip(Ph[i], NP[i]) if p and n > 1e-6)) for i in range(len(NP))])
    np.testing.assert_allclose(out['cuda-on-host'][0], out[''][0], atol=1e-6)
    assert out['cuda-on-host'][1] == out[''][1]
    assert out['cuda-on-host'][2] == out[''][2]
