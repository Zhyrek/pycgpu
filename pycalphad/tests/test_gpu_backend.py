"""
Tests for the accelerated equilibrium backends (backend='cpp' C++/OpenMP and
backend='cuda' CuPy/CUDA), which run the same generated kernel source.

The 'cpp' tests need a host C++17/OpenMP compiler (g++ or clang++); the 'cuda'
tests need CuPy and a working CUDA device. Each is skipped when unavailable.
First run per environment compiles the kernel (cached in the user cache dir),
so these tests are slower cold than warm.
"""
import shutil

import numpy as np
import pytest

from pycalphad import equilibrium, variables as v
from pycalphad.tests.fixtures import select_database, load_database


def _has_cuda():
    try:
        import cupy as cp
        cp.cuda.Device().compute_capability
        return True
    except Exception:
        return False


def _has_cpp_compiler():
    return shutil.which('g++') is not None or shutil.which('clang++') is not None


needs_cpp = pytest.mark.skipif(not _has_cpp_compiler(),
                               reason="no C++ compiler (g++/clang++) on PATH")
needs_cuda = pytest.mark.skipif(not _has_cuda(),
                                reason="CuPy/CUDA device not available")

CONDS = {v.X('ZN'): (0.2, 0.9, 0.3), v.T: (500, 700, 100), v.P: 101325, v.N: 1}
# Accelerated backends use an SVD-based linear solver instead of LAPACK dgelsd;
# agreement is limited by eps*cond(A) of the equilibrium matrix, not 1e-12.
GM_ATOL = 1e-3


def _compare_backends(dbf, backend):
    comps = ['AL', 'ZN', 'VA']
    phases = list(dbf.phases.keys())
    ref = equilibrium(dbf, comps, phases, CONDS)
    res = equilibrium(dbf, comps, phases, CONDS, backend=backend)
    np.testing.assert_allclose(res.GM.values, ref.GM.values, atol=GM_ATOL)
    np.testing.assert_allclose(res.MU.values, ref.MU.values, atol=1e-2)
    ref_sets = np.sort(ref.Phase.values, axis=-1)
    res_sets = np.sort(res.Phase.values, axis=-1)
    assert (ref_sets == res_sets).all(), "stable phase sets differ from CPU reference"


@needs_cpp
@select_database("alzn_mey.tdb")
def test_cpp_backend_matches_cpu(load_database):
    _compare_backends(load_database(), 'cpp')


@needs_cuda
@select_database("alzn_mey.tdb")
def test_cuda_backend_matches_cpu(load_database):
    _compare_backends(load_database(), 'cuda')


@needs_cpp
@select_database("alzn_mey.tdb")
def test_cpp_backend_deterministic(load_database):
    dbf = load_database()
    comps = ['AL', 'ZN', 'VA']
    phases = list(dbf.phases.keys())
    r1 = equilibrium(dbf, comps, phases, CONDS, backend='cpp')
    r2 = equilibrium(dbf, comps, phases, CONDS, backend='cpp')
    assert np.array_equal(r1.GM.values, r2.GM.values), "cpp backend not run-to-run deterministic"
    assert (r1.Phase.values == r2.Phase.values).all()


@needs_cpp
@select_database("alzn_mey.tdb")
def test_robust_phase_removal_kwarg(load_database):
    dbf = load_database()
    comps = ['AL', 'ZN', 'VA']
    phases = list(dbf.phases.keys())
    # robust_phase_removal only affects the accelerated kernels; on the
    # reference path it is accepted and ignored (reference solver is
    # upstream-unmodified).
    ref = equilibrium(dbf, comps, phases, CONDS)
    res = equilibrium(dbf, comps, phases, CONDS, backend='cpp', robust_phase_removal=True)
    np.testing.assert_allclose(res.GM.values, ref.GM.values, atol=GM_ATOL)


@select_database("alzn_mey.tdb")
def test_invalid_backend_rejected(load_database):
    dbf = load_database()
    with pytest.raises(ValueError, match="backend"):
        equilibrium(dbf, ['AL', 'ZN', 'VA'], list(dbf.phases.keys()), CONDS,
                    backend='opencl')

@select_database("alzn_mey.tdb")
def test_set_backend_api(load_database):
    import pycalphad
    dbf = load_database()
    comps = ['AL', 'ZN', 'VA']
    phases = list(dbf.phases.keys())
    ref = equilibrium(dbf, comps, phases, CONDS)
    # invalid names / options fail eagerly
    with pytest.raises(ValueError):
        pycalphad.set_backend('opencl')
    with pytest.raises(TypeError):
        pycalphad.set_backend('default', warp=9)
    _initial = pycalphad.get_backend()  # may be env-selected, not 'default'
    if not _has_cpp_compiler():
        pytest.skip("no C++ compiler")
    # context manager scoping + result agreement
    with pycalphad.backend('c++'):
        assert pycalphad.get_backend()[0] == 'cpp'
        res = equilibrium(dbf, comps, phases, CONDS)
    assert pycalphad.get_backend() == _initial
    np.testing.assert_allclose(res.GM.values, ref.GM.values, atol=GM_ATOL)


@needs_cpp
@select_database("alzn_mey.tdb")
def test_accelerated_calculate_cpp(load_database):
    import pycalphad
    from pycalphad import calculate
    dbf = load_database()
    comps = ['AL', 'ZN', 'VA']
    phases = list(dbf.phases.keys())
    ref = calculate(dbf, comps, phases, T=[500, 600], P=101325, N=1)
    with pycalphad.backend('c++'):
        res = calculate(dbf, comps, phases, T=[500, 600], P=101325, N=1)
    assert np.array_equal(ref.X.values, res.X.values)
    rel = np.nanmax(np.abs(res.GM.values - ref.GM.values) / np.maximum(np.abs(ref.GM.values), 1.0))
    assert rel < 1e-9, f"accelerated calculate GM rel err {rel}"

@needs_cpp
@select_database("alzn_mey.tdb")
def test_binplot_grid_method(load_database):
    import matplotlib
    matplotlib.use('Agg')
    import pycalphad
    from pycalphad import binplot
    dbf = load_database()
    conds = {v.X('ZN'): (0, 1, 1 / 30), v.T: (500, 900, 400 / 30), v.P: 101325, v.N: 1}
    with pycalphad.backend('c++'):
        ax = binplot(dbf, ['AL', 'ZN', 'VA'], list(dbf.phases.keys()), conds,
                     method='grid')
    # Boundary points exist and lie in valid composition range
    pts = np.concatenate([c.get_offsets() for c in ax.collections if len(c.get_offsets())])
    assert pts.shape[0] > 20, "expected boundary points from two-phase fields"
    assert np.all((pts[:, 0] >= 0) & (pts[:, 0] <= 1))
    assert np.all((pts[:, 1] >= 500) & (pts[:, 1] <= 900))
    ax.figure.clf()


@select_database("alzn_mey.tdb")
def test_binplot_grid_method_invalid(load_database):
    from pycalphad import binplot
    dbf = load_database()
    conds = {v.X('ZN'): (0, 1, 0.1), v.T: (500, 900, 50), v.P: 101325, v.N: 1}
    with pytest.raises(ValueError, match="method"):
        binplot(dbf, ['AL', 'ZN', 'VA'], list(dbf.phases.keys()), conds, method='bogus')

