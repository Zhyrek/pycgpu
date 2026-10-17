"""
Tests for the grid sampling options of calculate(): pair_pdens, convex_endmembers_only, disordered_pdens.
"""
import itertools
import numpy as np
import pytest
from numpy.testing import assert_allclose
from pycalphad import Model, Workspace, calculate, equilibrium, variables as v
from pycalphad.core.calculate import _sample_phase_constitution, _convex_mixing_sublattice, \
    _ordering_sublattices
from pycalphad.core.utils import point_sample, endmember_matrix
from pycalphad.tests.fixtures import select_database, load_database


@select_database("alcrni.tdb")
def test_pair_pdens_sets_line_density_independently(load_database):
    "pair_pdens sets the number of points per endmember pair line, without repeating endmembers."
    dbf = load_database()
    mod = Model(dbf, ['AL', 'CR', 'NI', 'VA'], 'L12_FCC')
    num_endmembers = endmember_matrix([3, 3]).shape[0]
    num_pairs = num_endmembers * (num_endmembers - 1) // 2
    num_random = point_sample([3, 3], pdof=60).shape[0]
    for pair_pdens in (2, 5, 10):
        points = _sample_phase_constitution(mod, point_sample, True, 60, {}, pair_pdens)
        assert points.shape[0] == num_endmembers + num_pairs * (pair_pdens - 2) + num_random
    # Default: pdens points per line, endmembers included in each line
    points = _sample_phase_constitution(mod, point_sample, True, 60, {})
    assert points.shape[0] == num_endmembers + num_pairs * 60 + num_random


@select_database("TiO-17Yan.tdb")
def test_convexity_criterion(load_database):
    "Only phases with one ideal mixing sublattice of neutral species and no other energy terms are convex."
    dbf = load_database()
    comps = ['TI', 'O', 'VA']
    assert _convex_mixing_sublattice(Model(dbf, comps, 'GAS')) == 0
    for phase_name in ['LIQUID', 'FCC_A1', 'ION', 'RUTILE_C4', 'O2GAS', 'TIO_ALPHA']:
        # excess terms, ionic liquid, charged species, single endmember
        assert _convex_mixing_sublattice(Model(dbf, comps, phase_name)) is None


@select_database("alcrni.tdb")
def test_ordered_phases_not_convex(load_database):
    dbf = load_database()
    for phase_name in ['B2', 'L12_FCC', 'LIQUID']:
        assert _convex_mixing_sublattice(Model(dbf, ['AL', 'CR', 'NI', 'VA'], phase_name)) is None


@select_database("alcrni.tdb")
def test_disordered_points_on_disordered_manifold(load_database):
    "disordered_pdens adds points with equal site fractions on the ordering sublattices."
    dbf = load_database()
    mod = Model(dbf, ['AL', 'CR', 'NI', 'VA'], 'L12_FCC')
    assert _ordering_sublattices(mod) == ([0, 1], [])
    base = _sample_phase_constitution(mod, point_sample, True, 60, {})
    points = _sample_phase_constitution(mod, point_sample, True, 60, {}, None, False, 60)
    extra = points[base.shape[0]:]
    assert extra.shape[0] == point_sample([3], pdof=60).shape[0]
    assert_allclose(extra[:, :3], extra[:, 3:])
    assert_allclose(base, points[:base.shape[0]])


@select_database("TiO-17Yan.tdb")
def test_convex_endmembers_only_equilibrium(load_database):
    "Endmember-only sampling of the ideal gas gives the dense-grid equilibrium where the gas is stable."
    dbf = load_database()
    comps = ['TI', 'O', 'VA']
    phases = ['GAS', 'RUTILE_C4', 'LIQUID']
    conds = {v.T: [2500, 3200], v.P: 101325, v.N: 1, v.X('O'): [0.7, 0.8, 0.95]}
    dense = equilibrium(dbf, comps, phases, conds, calc_opts={'pdens': 1000})
    sparse = equilibrium(dbf, comps, phases, conds, calc_opts={'convex_endmembers_only': True})
    assert np.all(np.any(sparse.Phase.values == 'GAS', axis=-1) == np.any(dense.Phase.values == 'GAS', axis=-1))
    assert np.any(dense.Phase.values == 'GAS')
    assert_allclose(sparse.GM.values, dense.GM.values, atol=1e-3)
