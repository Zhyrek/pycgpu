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
