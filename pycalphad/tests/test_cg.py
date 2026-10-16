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
