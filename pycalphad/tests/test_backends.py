"""
Tests for the compiled backends (see pycalphad.backend).

Each compiled backend runs the same generated solver: 'cpp' compiles it for
the host, 'cuda' runs it through CuPy, and 'metal' runs it on Apple-silicon
GPUs in emulated double precision. Tests for a backend are skipped when it
is unavailable on the machine. The first run of a system compiles its
kernels (cached on disk), so these tests are slower cold than warm.
"""
import numpy as np
import pytest

import pycalphad
from pycalphad import Workspace, as_property, calculate, equilibrium, variables as v
from pycalphad.backend import available_backends
from pycalphad.tests.fixtures import select_database, load_database

_AVAILABLE = available_backends()


def _needs(name):
    return pytest.mark.skipif(_AVAILABLE.get(name) is not None,
                              reason=f"backend {name!r} unavailable: {_AVAILABLE.get(name)}")


needs_cpp = _needs('cpp')
needs_cuda = _needs('cuda')
needs_metal = _needs('metal')
COMPILED = [pytest.param('cpp', marks=needs_cpp), pytest.param('cuda', marks=needs_cuda),
            pytest.param('metal', marks=needs_metal)]

CONDS = {v.X('ZN'): (0.2, 0.9, 0.3), v.T: (500, 700, 100), v.P: 101325, v.N: 1}
# The compiled solvers converge to the same states as the reference within
# its tolerances; energies agree far below this.
GM_ATOL = 1e-2


def _assert_matches_reference(res, ref):
    assert np.isfinite(res.GM.values).sum() == np.isfinite(ref.GM.values).sum()
    np.testing.assert_allclose(res.GM.values, ref.GM.values, atol=GM_ATOL)
    np.testing.assert_allclose(res.MU.values, ref.MU.values, atol=0.1)
    ref_sets = np.sort(ref.Phase.values, axis=-1)
    res_sets = np.sort(res.Phase.values, axis=-1)
    assert (ref_sets == res_sets).all(), "stable phase sets differ from the reference"


@pytest.mark.parametrize('backend', COMPILED)
@select_database("alzn_mey.tdb")
def test_backend_matches_reference(load_database, backend):
    dbf = load_database()
    comps, phases = ['AL', 'ZN', 'VA'], list(dbf.phases.keys())
    ref = equilibrium(dbf, comps, phases, CONDS, backend='default')
    res = equilibrium(dbf, comps, phases, CONDS, backend=backend, output='HM')
    _assert_matches_reference(res, ref)
    ref_hm = equilibrium(dbf, comps, phases, CONDS, backend='default', output='HM')
    np.testing.assert_allclose(res.HM.values, ref_hm.HM.values, atol=GM_ATOL)


@pytest.mark.parametrize('backend', COMPILED)
@select_database("alzn_mey.tdb")
def test_backend_deterministic(load_database, backend):
    dbf = load_database()
    comps, phases = ['AL', 'ZN', 'VA'], list(dbf.phases.keys())
    r1 = equilibrium(dbf, comps, phases, CONDS, backend=backend)
    r2 = equilibrium(dbf, comps, phases, CONDS, backend=backend)
    assert np.array_equal(r1.GM.values, r2.GM.values, equal_nan=True)
    assert (r1.Phase.values == r2.Phase.values).all()


@pytest.mark.parametrize('backend', COMPILED)
@select_database("alcocrni.tdb")
def test_backend_quaternary_batch(load_database, backend):
    # Five components (with VA) exercise the per-condition row strides that
    # binaries and ternaries cannot distinguish, and on Metal the solver
    # state of this 21-phase system exceeds a GPU thread's stack, so the
    # backend must hand the system to the C++ solver transparently.
    dbf = load_database()
    comps = ['AL', 'CO', 'CR', 'NI', 'VA']
    phases = list(dbf.phases.keys())
    conds = {v.N: 1, v.P: 101325, v.T: (800, 1601, 400),
             v.X('AL'): 0.05, v.X('CO'): 0.05, v.X('CR'): (0.05, 0.35, 0.1)}
    ref = equilibrium(dbf, comps, phases, conds, backend='default')
    res = equilibrium(dbf, comps, phases, conds, backend=backend)
    assert np.isfinite(res.GM.values).sum() == np.isfinite(ref.GM.values).sum()
    np.testing.assert_allclose(res.GM.values, ref.GM.values, atol=GM_ATOL)


@pytest.mark.parametrize('backend', COMPILED)
@select_database("alzn_mey.tdb")
def test_backend_calculate(load_database, backend):
    dbf = load_database()
    comps, phases = ['AL', 'ZN', 'VA'], list(dbf.phases.keys())
    ref = calculate(dbf, comps, phases, T=[500, 600], P=101325, N=1)
    with pycalphad.backend(backend):
        res = calculate(dbf, comps, phases, T=[500, 600], P=101325, N=1)
    assert np.array_equal(ref.X.values, res.X.values)
    rel = np.nanmax(np.abs(res.GM.values - ref.GM.values) / np.maximum(np.abs(ref.GM.values), 1.0))
    assert rel < 1e-9


@needs_cpp
@select_database("alzn_mey.tdb")
def test_robust_phase_removal_kwarg(load_database):
    dbf = load_database()
    comps, phases = ['AL', 'ZN', 'VA'], list(dbf.phases.keys())
    ref = equilibrium(dbf, comps, phases, CONDS, backend='default')
    res = equilibrium(dbf, comps, phases, CONDS, backend='cpp', robust_phase_removal=True)
    np.testing.assert_allclose(res.GM.values, ref.GM.values, atol=GM_ATOL)


@needs_cpp
@select_database("alzn_mey.tdb")
def test_batched_jansson_derivatives_match_reference(load_database):
    """pycalphad.gpu.jansson_derivatives computes Jansson derivatives for a
    whole grid in one compiled solve; they match the reference property
    framework (Workspace.get) at every condition."""
    from pycalphad import Workspace
    from pycalphad.property_framework.computed_property import JanssonDerivative
    from pycalphad.gpu.jansson_derivatives import jansson_derivative
    dbf = load_database()
    comps, phases = ['AL', 'ZN', 'VA'], list(dbf.phases.keys())
    with pycalphad.backend('default'):
        wks = Workspace(dbf, comps, phases, CONDS)
        refs = {v.T: np.asarray(wks.get('HM.T')).ravel(),
                v.X('ZN'): np.asarray(wks.get(JanssonDerivative('HM', v.X('ZN')))).ravel()}
    for denominator, ref in refs.items():
        for name in ('cpp', 'metal', 'cpu'):  # all of these run the compiled C++ epilogue
            res = np.asarray(jansson_derivative(dbf, comps, phases, CONDS, 'HM', denominator,
                                                backend=name)['values']).ravel()
            assert np.array_equal(np.isnan(res), np.isnan(ref))
            np.testing.assert_allclose(res, ref, rtol=1e-6, atol=1e-6)


@needs_cpp
@select_database("alni_dupin_2001.tdb")
def test_workspace_jansson_derivatives_batched(load_database):
    """With a compiled backend, Workspace.get and equilibrium(output=...) compute
    Jansson derivatives of Model properties for the whole grid in the compiled
    solver; they match the reference derivative code evaluated on the same
    converged states (Al-Ni: B2 and AL3NI2 have vacancies / several atoms per
    formula unit). Other derivatives still use the per-condition path."""
    import pycalphad.gpu.dispatch as dispatch
    dbf = load_database()
    comps, phases = ['AL', 'NI', 'VA'], list(dbf.phases.keys())
    conds = {v.N: 1, v.P: 101325, v.T: (900, 1700, 200), v.X('AL'): (0.1, 0.9, 0.1)}
    props = ['HM.T', 'GM.X(AL)', 'NP(BCC_B2).T']
    with pycalphad.backend('cpp'):
        wks = Workspace(dbf, comps, phases, conds)
        batched = [np.asarray(x, dtype=float) for x in wks.get(*props)]
        assert set(dispatch.workspace_properties(wks, [as_property(p) for p in props])) == \
            {as_property('HM.T'), as_property('GM.X(AL)')}
        eq = equilibrium(dbf, comps, phases, conds, output='HM.T')
        original = dispatch.workspace_properties
        dispatch.workspace_properties = lambda *args, **kwargs: {}
        try:
            reference = [np.asarray(x, dtype=float) for x in wks.get(*props)]
        finally:
            dispatch.workspace_properties = original
    for res, ref in zip(batched, reference):
        assert np.array_equal(np.isnan(res), np.isnan(ref))
        np.testing.assert_allclose(res, ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(eq['HM.T'].values.squeeze(), reference[0].squeeze(), rtol=1e-6, atol=1e-6)


@needs_cpp
@select_database("alni_dupin_2001.tdb")
def test_workspace_properties_vectorized(load_database):
    """On a backend-solved Workspace, common properties are evaluated for the
    whole grid at once; they are bit-identical to the per-condition evaluation
    (miscibility-gap copies, wildcards, display units and failed conditions
    included)."""
    import pycalphad.gpu.dispatch as dispatch
    dbf = load_database()
    conds = {v.N: 1, v.P: 101325, v.T: (700, 1800, 100), v.X('AL'): (0.02, 0.98, 0.08)}
    props = ['T', 'GM', 'HM', 'MU(AL)', 'X(AL)', 'NP(*)', 'X(*,AL)', 'GM(*)', 'Y(FCC_L12,0,AL)',
             'NP(FCC_L12#2)', as_property('HM')['kJ/mol']]
    for max_iters in (21, 1000):  # with and without failed conditions
        with pycalphad.backend('cpp', max_iters=max_iters):
            wks = Workspace(dbf, ['AL', 'NI', 'VA'], list(dbf.phases.keys()), conds)
            vectorized = wks.get_dict(*props)
            original = dispatch.workspace_properties
            dispatch.workspace_properties = lambda *args, **kwargs: {}
            try:
                reference = wks.get_dict(*props)
            finally:
                dispatch.workspace_properties = original
        assert [str(k) for k in vectorized] == [str(k) for k in reference]
        for key in reference:
            assert np.array_equal(np.asarray(vectorized[key]), np.asarray(reference[key]), equal_nan=True), key


@needs_cpp
@select_database("alni_dupin_2001.tdb")
def test_isolated_phase_compiled_chain(load_database):
    """IsolatedPhase properties on a compiled backend replay Workspace.get's
    warm-started sequence of single-phase solves (conditions in order, then
    each property sharing the IsolatedPhase) on the compiled solver; values
    and the final composition set match the reference sequence."""
    import pycalphad.gpu.dispatch as dispatch
    from pycalphad.property_framework import IsolatedPhase
    dbf = load_database()
    conds = {v.N: 1, v.P: 101325, v.T: (900, 1700, 100), v.X('AL'): (0.1, 0.6, 0.05)}
    results, final_dof = {}, {}
    for mode in ('compiled', 'reference'):
        with pycalphad.backend('cpp'):
            wks = Workspace(dbf, ['AL', 'NI', 'VA'], list(dbf.phases.keys()), conds)
            iph = IsolatedPhase('BCC_B2', wks=wks)
            props = [iph('GM'), iph('MU(AL)'), iph('Y(BCC_B2,0,AL)'), iph.driving_force]
            original = dispatch.workspace_properties
            if mode == 'reference':
                dispatch.workspace_properties = lambda *args, **kwargs: {}
            try:
                results[mode] = [np.asarray(x, dtype=float) for x in wks.get(*props)]
            finally:
                dispatch.workspace_properties = original
            final_dof[mode] = np.array(iph._compset.dof)
    for compiled, reference in zip(results['compiled'], results['reference']):
        assert np.array_equal(np.isnan(compiled), np.isnan(reference))
        np.testing.assert_allclose(compiled, reference, rtol=1e-8, atol=1e-6)
    np.testing.assert_allclose(final_dof['compiled'], final_dof['reference'], atol=1e-10)


@needs_cpp
@select_database("alzn_mey.tdb")
def test_mapping_global_min_scan_matches_calculate(load_database):
    """On a compiled backend, the mapping's global-minimum check samples
    phases without calculate()'s dataset assembly; every check picks the same
    candidate composition set with the same driving force as the
    calculate()-based path."""
    import matplotlib
    matplotlib.use('Agg')
    import pycalphad.gpu.mapping_scan as mapping_scan
    import pycalphad.mapping.zpf_equilibrium as zpf
    from pycalphad import binplot
    dbf = load_database()
    original = zpf._find_global_min_cs
    checks = []

    def compare(point, system_info, *args, **kwargs):
        new = original(point, system_info, *args, **kwargs)
        scan = mapping_scan.sampled_energies
        mapping_scan.sampled_energies = lambda *a, **k: None
        try:
            old = original(point, system_info, *args, **kwargs)
        finally:
            mapping_scan.sampled_energies = scan
        if new is None or old is None:
            checks.append(new is None and old is None)
        else:
            checks.append(new[0].phase_record.phase_name == old[0].phase_record.phase_name
                          and np.array_equal(np.asarray(new[0].dof), np.asarray(old[0].dof))
                          and new[1] == old[1])
        return new

    zpf._find_global_min_cs = compare
    try:
        with pycalphad.backend('cpp'):
            ax = binplot(dbf, ['AL', 'ZN', 'VA'], list(dbf.phases.keys()),
                         {v.X('ZN'): (0, 1, 0.05), v.T: (400, 900, 25), v.P: 101325, v.N: 1})
    finally:
        zpf._find_global_min_cs = original
    ax.figure.clf()
    assert len(checks) > 20
    assert all(checks)


@select_database("alzn_mey.tdb")
def test_invalid_backend_rejected(load_database):
    dbf = load_database()
    with pytest.raises(ValueError, match="backend"):
        equilibrium(dbf, ['AL', 'ZN', 'VA'], list(dbf.phases.keys()), CONDS, backend='opencl')


def test_set_backend_api():
    initial = pycalphad.get_backend()
    with pytest.raises(ValueError):
        pycalphad.set_backend('opencl')
    with pytest.raises(TypeError):
        pycalphad.set_backend('default', warp=9)
    assert pycalphad.get_backend() == initial
    # 'auto' always resolves to something usable
    with pycalphad.backend('auto'):
        assert pycalphad.get_backend()[0] in ('default', 'cpp', 'cuda')
    with pycalphad.backend('default', threads=3):
        assert pycalphad.get_backend() == ('default', {'threads': 3})
    # 'cpu' is the reference solver (as in earlier releases); only 'cpp'/'c++' compile
    for name in ('cpu', 'reference'):
        with pycalphad.backend(name):
            assert pycalphad.get_backend()[0] == 'default'
    if _AVAILABLE['cpp'] is None:
        for name in ('cpp', 'c++', 'openmp'):
            with pycalphad.backend(name):
                assert pycalphad.get_backend()[0] == 'cpp'
    assert pycalphad.get_backend() == initial


def test_gpu_selector():
    usable = [name for name in ('cuda', 'metal') if _AVAILABLE[name] is None]
    if not usable:
        with pytest.raises(RuntimeError, match="no usable GPU"):
            with pycalphad.backend('gpu'):
                pass
        return
    with pycalphad.backend('gpu'):
        assert pycalphad.get_backend()[0] == usable[0]


@needs_cpp
@select_database("alzn_mey.tdb")
def test_binplot_grid_method(load_database):
    import matplotlib
    matplotlib.use('Agg')
    from pycalphad import binplot
    dbf = load_database()
    conds = {v.X('ZN'): (0, 1, 1 / 30), v.T: (500, 900, 400 / 30), v.P: 101325, v.N: 1}
    with pycalphad.backend('cpp'):
        ax = binplot(dbf, ['AL', 'ZN', 'VA'], list(dbf.phases.keys()), conds, method='grid')
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


@needs_metal
def test_metal_double_float_arithmetic():
    """The Metal kernels' emulated double precision: ~1e-14 relative
    accuracy over double's full exponent range (magnetic models produce
    T**25-sized intermediates)."""
    from pycalphad.gpu import metal_backend as mb
    body = """
kernel void t(device const pyc_f64* a [[buffer(0)]], device const pyc_f64* b [[buffer(1)]],
              device pyc_f64* out [[buffer(2)]], constant uint& n [[buffer(3)]],
              uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    double x = a[i], y = b[i];
    out[i*8+0] = x + y; out[i*8+1] = x * y; out[i*8+2] = x / y; out[i*8+3] = sqrt(fabs(x));
    out[i*8+4] = log(fabs(x)); out[i*8+5] = exp(fmod(y, 700.0));
    out[i*8+6] = pow(fabs(x), 25) * pow(fabs(y), -24); out[i*8+7] = (x < y) ? 1.0 : 0.0;
}
"""
    rt = mb.runtime()
    kernel = rt.kernel(rt.compile(mb.build_msl_source(mb.lower_to_msl(body))), 't')
    rng = np.random.default_rng(0)
    n = 20000
    a = rng.uniform(-3, 3, n) * 10.0 ** rng.integers(-12, 12, n)
    b = rng.uniform(-300, 300, n)
    out = rt.buffer(8 * 8 * n)
    kernel([rt.buffer_from(a), rt.buffer_from(b), out], np.uint32(n).tobytes(), n, 64)
    got = out.view(np.float64).reshape(n, 8)
    with np.errstate(all='ignore'):
        want = np.column_stack([a + b, a * b, a / b, np.sqrt(abs(a)), np.log(abs(a)),
                                np.exp(np.fmod(b, 700.0)), abs(a) ** 25 * abs(b) ** -24.0,
                                (a < b).astype(float)])
    finite = np.isfinite(want) & (np.abs(want) > 1e-280) & (np.abs(want) < 1e280)
    # inputs carry 48 significant bits on the GPU; cancellation in a+b and
    # large exp arguments amplify that, so compare those against the
    # magnitude of the operands
    scale = np.abs(want)
    scale[:, 0] = np.maximum(np.abs(a), np.abs(b))
    scale[:, 5] *= np.maximum(1, np.abs(np.fmod(b, 700.0)))
    scale[:, 4] = np.maximum(scale[:, 4], 1.0)
    rel = np.abs(got - want) / np.where(scale > 0, scale, 1)
    assert np.all(rel[finite] < 1e-12), np.max(np.where(finite, rel, 0), axis=0)
