"""
CGWorkspace against pycalphad's own Workspace on the databases in pycalphad's test suite.

Column generation is a different minimizer, so the results are compared by energy: at every condition its molar
Gibbs energy must not be above pycalphad's (beyond rounding), and where both find the same equilibrium the chemical
potentials agree. pycalphad's grid can miss a phase, so column generation is allowed to be lower.
"""
from importlib.resources import files

import numpy as np
import pytest
from pycalphad import Database, Workspace, variables as v
from pycalphad.core.conditions import ConditionError
from pycalphad.core.solver import Solver

from pycalphad_cg import CGWorkspace, CGCapabilityError, equilibrium

DATABASES = files('pycalphad.tests.databases')
GM_TOL = 0.01  # J/mol


def _db(name):
    return Database(str(DATABASES / name))


def _all_phases(dbf):
    return sorted(dbf.phases.keys())


CASES = [
    ('alfe.tdb', ['AL', 'FE', 'VA'], {v.T: (500, 1901, 100), v.X('AL'): (0.02, 0.99, 0.04), v.P: 101325, v.N: 1}),
    ('cumg.tdb', ['CU', 'MG', 'VA'], {v.T: (500, 1501, 100), v.X('MG'): (0.02, 0.99, 0.04), v.P: 101325, v.N: 1}),
    ('alzn_mey.tdb', ['AL', 'ZN', 'VA'], {v.T: (400, 901, 50), v.X('ZN'): (0.02, 0.99, 0.04), v.P: 101325, v.N: 1}),
    ('FeNi_deep_branching.tdb', ['FE', 'NI', 'VA'],
     {v.T: (300, 1001, 100), v.X('NI'): (0.02, 0.99, 0.04), v.P: 101325, v.N: 1}),
    ('alcrni.tdb', ['AL', 'CR', 'NI', 'VA'],
     {v.T: 1273, v.X('AL'): (0.02, 0.5, 0.06), v.X('CR'): (0.02, 0.5, 0.06), v.P: 101325, v.N: 1}),
    ('CrFeNb_Jacob2016.tdb', ['CR', 'FE', 'NB', 'VA'],
     {v.T: 1473, v.X('CR'): (0.05, 0.6, 0.1), v.X('NB'): (0.05, 0.4, 0.1), v.P: 101325, v.N: 1}),
]


@pytest.mark.parametrize('name,comps,conds', CASES, ids=[c[0] for c in CASES])
def test_energy_not_above_pycalphad(name, comps, conds):
    dbf = _db(name)
    phases = _all_phases(dbf)
    ref = Workspace(dbf, comps, phases, conds).eq
    cg = CGWorkspace(dbf, comps, phases, conds).eq
    gm_ref, gm_cg = np.squeeze(ref.GM), np.squeeze(cg.GM)
    valid = ~np.isnan(gm_ref)
    assert np.all(~np.isnan(gm_cg[valid]))
    assert np.all(gm_cg[valid] <= gm_ref[valid] + GM_TOL), np.max(gm_cg[valid] - gm_ref[valid])
    # where the two agree on the energy, they agree on the chemical potentials
    same = valid & (np.abs(gm_cg - gm_ref) < GM_TOL)
    mu_ref, mu_cg = np.squeeze(ref.MU)[same], np.squeeze(cg.MU)[same]
    np.testing.assert_allclose(mu_cg, mu_ref, atol=1.0)


def test_equilibrium_wrapper_and_properties():
    dbf = _db('alfe.tdb')
    phases = _all_phases(dbf)
    conds = {v.T: (800, 1201, 200), v.X('AL'): (0.1, 0.6, 0.1), v.P: 101325, v.N: 1}
    from pycalphad import equilibrium as pycalphad_equilibrium
    ref = pycalphad_equilibrium(dbf, ['AL', 'FE', 'VA'], phases, conds, output='HM')
    cg = equilibrium(dbf, ['AL', 'FE', 'VA'], phases, conds, output='HM')
    assert set(ref.data_vars) == set(cg.data_vars)
    assert ref.GM.dims == cg.GM.dims
    np.testing.assert_allclose(cg.HM.values, ref.HM.values, atol=1.0)


def test_workspace_get():
    dbf = _db('cumg.tdb')
    conds = {v.T: 1000, v.X('MG'): 0.3, v.P: 101325, v.N: 1}
    phases = _all_phases(dbf)
    ref = Workspace(dbf, ['CU', 'MG', 'VA'], phases, conds)
    cg = CGWorkspace(dbf, ['CU', 'MG', 'VA'], phases, conds)
    for prop in ['GM', 'HM', 'MU(MG)', 'NP(*)', 'X(LIQUID,MG)']:
        np.testing.assert_allclose(np.nan_to_num(cg.get(prop)), np.nan_to_num(ref.get(prop)), atol=1e-3, rtol=1e-6)


def test_input_errors_match_pycalphad():
    dbf = _db('alfe.tdb')
    phases = _all_phases(dbf)
    # missing state variable
    with pytest.raises(ConditionError):
        CGWorkspace(dbf, ['AL', 'FE', 'VA'], phases, {v.X('AL'): 0.3, v.P: 101325, v.N: 1}).eq
    # N != 1
    with pytest.raises(ConditionError):
        CGWorkspace(dbf, ['AL', 'FE', 'VA'], phases, {v.T: 1000, v.X('AL'): 0.3, v.P: 101325, v.N: 2}).eq
    # wrong number of composition conditions
    with pytest.raises(ValueError, match='degrees of freedom'):
        CGWorkspace(dbf, ['AL', 'FE', 'VA'], phases, {v.T: 1000, v.P: 101325, v.N: 1}).eq


class _CustomSolver(Solver):
    pass


def test_fallback_to_pycalphad():
    dbf = _db('alfe.tdb')
    phases = _all_phases(dbf)
    conds = {v.T: 1000, v.X('AL'): 0.3, v.P: 101325, v.N: 1}
    with pytest.warns(UserWarning, match='column generation does not handle'):
        gm = CGWorkspace(dbf, ['AL', 'FE', 'VA'], phases, conds, solver=_CustomSolver()).get('GM')
    np.testing.assert_allclose(gm, Workspace(dbf, ['AL', 'FE', 'VA'], phases, conds).get('GM'))
    with pytest.raises(CGCapabilityError):
        CGWorkspace(dbf, ['AL', 'FE', 'VA'], phases, conds, solver=_CustomSolver(), cg_fallback=False).eq


@pytest.mark.parametrize('threads', [1, 4])
def test_threads_give_the_same_result(threads):
    dbf = _db('alcrni.tdb')
    phases = _all_phases(dbf)
    conds = {v.T: 1273, v.X('AL'): (0.05, 0.5, 0.05), v.X('CR'): (0.05, 0.5, 0.05), v.P: 101325, v.N: 1}
    one = CGWorkspace(dbf, ['AL', 'CR', 'NI', 'VA'], phases, conds, cg_threads=1).get('GM')
    other = CGWorkspace(dbf, ['AL', 'CR', 'NI', 'VA'], phases, conds, cg_threads=threads).get('GM')
    np.testing.assert_allclose(other, one, atol=1e-6)
