"""
Test the grand potential phase diagram solver against pycalphad equilibrium.

Compares results for the NbTi binary system.
"""

import sys
import os
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pycalphad import Database, equilibrium, variables as v
from advanced_strategy.grand_potential import (
    compute_phase_diagram, extract_phase_boundaries
)


def test_nbti_detailed_verification():
    """Verify grand potential boundaries match pycalphad at T=2200K."""
    print("=" * 70)
    print("TEST: Detailed verification at T=2200K")
    print("=" * 70)

    dbf = Database('NbTi.tdb')
    comps = ['NB', 'TI']
    phases = list(dbf.phases.keys())

    T = 2200
    result = compute_phase_diagram(
        dbf, comps, phases,
        conditions={v.T: T, v.P: 101325},
        mu_resolution=500,
        verbose=False
    )

    res = result['results_per_T'][0]
    boundaries = res['phase_boundaries']

    if boundaries:
        bnd = boundaries[0]
        x_solidus = bnd['x_phase_a'][1]
        x_liquidus = bnd['x_phase_b'][1]
        print(f"GP: solidus x_TI = {x_solidus:.6f}, liquidus x_TI = {x_liquidus:.6f}")
    else:
        print("ERROR: No boundaries found at T=2200K")
        return

    # Check against pycalphad
    mismatches = 0
    total = 0
    for x_ti in np.arange(0.05, 1.0, 0.05):
        eq = equilibrium(dbf, comps, phases,
                         {v.T: T, v.P: 101325, v.X('TI'): x_ti})
        stable = []
        for idx in range(eq.Phase.shape[-1]):
            pname = str(eq.Phase.values.squeeze()[idx])
            if pname and pname != '' and pname != 'nan':
                np_val = float(eq.NP.values.squeeze()[idx])
                if np_val > 1e-6:
                    stable.append(pname)

        low_x = min(x_solidus, x_liquidus)
        high_x = max(x_solidus, x_liquidus)
        if x_ti < low_x:
            gp_pred = "BCC_A2"
        elif x_ti > high_x:
            gp_pred = "LIQUID"
        else:
            gp_pred = "2-phase"

        eq_str = '+'.join(stable)
        match = (gp_pred == eq_str) or (gp_pred == "2-phase" and len(stable) >= 2)
        total += 1
        if not match:
            mismatches += 1
            print(f"  MISMATCH at x_TI={x_ti:.2f}: GP={gp_pred}, EQ={eq_str}")

    print(f"Result: {total - mismatches}/{total} match ({mismatches} mismatches)")


def test_nbti_phase_diagram():
    """Compute full NbTi phase diagram with fine T resolution."""
    print("\n" + "=" * 70)
    print("TEST: NbTi phase diagram (fine resolution)")
    print("=" * 70)

    dbf = Database('NbTi.tdb')
    comps = ['NB', 'TI']
    phases = list(dbf.phases.keys())

    # Fine temperature sweep to capture full liquidus/solidus
    T_values = np.arange(1900, 2800, 25)
    P = 101325

    t0 = time.time()
    result = compute_phase_diagram(
        dbf, comps, phases,
        conditions={v.T: T_values, v.P: P},
        mu_resolution=500,
        verbose=False
    )
    t_gp = time.time() - t0

    all_boundaries = extract_phase_boundaries(result)
    print(f"\nGrand potential: {len(T_values)} temperatures in {t_gp:.3f}s "
          f"({t_gp/len(T_values)*1000:.1f} ms/T)")
    print(f"Phase boundaries: {len(all_boundaries)} points")

    print(f"\n{'T (K)':>8s}  {'Solidus x_TI':>14s}  {'Liquidus x_TI':>14s}")
    print("-" * 42)
    for bnd in sorted(all_boundaries, key=lambda b: b['T']):
        # solidus = BCC_A2 side, liquidus = LIQUID side
        x_s = bnd['x_a']
        x_l = bnd['x_b']
        print(f"{bnd['T']:8.0f}  {x_s:14.6f}  {x_l:14.6f}")


def test_speed_comparison():
    """Compare GP speed against pycalphad equilibrium for equivalent coverage."""
    print("\n" + "=" * 70)
    print("TEST: Speed comparison")
    print("=" * 70)

    dbf = Database('NbTi.tdb')
    comps = ['NB', 'TI']
    phases = list(dbf.phases.keys())

    # --- Grand potential: 100 temperatures, all compositions ---
    T_values = np.linspace(1800, 2800, 100)

    t0 = time.time()
    result = compute_phase_diagram(
        dbf, comps, phases,
        conditions={v.T: T_values, v.P: 101325},
        mu_resolution=500,
        verbose=False
    )
    t_gp = time.time() - t0

    # Each temperature covers 500 mu points = ~500 composition points
    gp_coverage = len(T_values) * 500

    print(f"\nGrand potential:")
    print(f"  100 temperatures x 500 compositions = {gp_coverage} points")
    print(f"  Total time: {t_gp:.3f}s")
    print(f"  Per temperature (all compositions): {t_gp/len(T_values)*1000:.1f} ms")
    print(f"  Per equivalent equilibrium point: {t_gp/gp_coverage*1000:.3f} ms")

    # --- pycalphad: sample 20 conditions for timing baseline ---
    conditions_list = []
    for T in [1900, 2000, 2200, 2500]:
        for x in [0.1, 0.3, 0.5, 0.7, 0.9]:
            conditions_list.append((T, x))

    t0 = time.time()
    for T, x in conditions_list:
        _ = equilibrium(dbf, comps, phases,
                        {v.T: T, v.P: 101325, v.X('TI'): x})
    t_eq = time.time() - t0

    per_eq = t_eq / len(conditions_list)
    eq_equivalent_time = per_eq * gp_coverage

    print(f"\npycalphad equilibrium:")
    print(f"  {len(conditions_list)} conditions in {t_eq:.3f}s")
    print(f"  Per condition: {per_eq*1000:.1f} ms")
    print(f"  Equivalent {gp_coverage} conditions would take: {eq_equivalent_time:.1f}s")

    print(f"\nSpeedup: {eq_equivalent_time / t_gp:.0f}x")
    print(f"  (GP: {t_gp:.2f}s vs EQ estimated: {eq_equivalent_time:.1f}s)")
    print(f"  for {gp_coverage} phase diagram grid points")


if __name__ == '__main__':
    test_nbti_detailed_verification()
    test_nbti_phase_diagram()
    test_speed_comparison()
