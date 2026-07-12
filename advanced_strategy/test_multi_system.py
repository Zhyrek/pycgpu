"""
Test GPU grand potential solver on multiple binary phase diagram systems.
Compare against CPU grand potential and pycalphad equilibrium.
"""

import sys
import os
import numpy as np
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pycalphad import Database, equilibrium, variables as v
from advanced_strategy.grand_potential import (
    compute_phase_diagram as compute_cpu,
    extract_phase_boundaries,
)
from advanced_strategy.grand_potential_gpu import (
    compute_phase_diagram_gpu as compute_gpu,
)


# Define test systems
SYSTEMS = {
    'NbTi': {
        'tdb': 'NbTi.tdb',
        'comps': ['NB', 'TI'],
        'T_range': (2000, 2700, 8),   # start, stop, n_points
        'T_verify': 2200,
    },
    'AlZn': {
        'tdb': 'pycalphad/tests/databases/alzn_mey.tdb',
        'comps': ['AL', 'ZN'],
        'T_range': (400, 700, 8),
        'T_verify': 600,
    },
    'PbSn': {
        'tdb': 'pycalphad/tests/databases/pbsn.tdb',
        'comps': ['PB', 'SN'],
        'T_range': (350, 550, 8),
        'T_verify': 450,
    },
    'AuBi': {
        'tdb': 'AuBi-07Wan.tdb',
        'comps': ['AU', 'BI'],
        'T_range': (300, 700, 8),
        'T_verify': 500,
    },
    'AlFe': {
        'tdb': 'pycalphad/tests/databases/alfe.tdb',
        'comps': ['AL', 'FE'],
        'T_range': (800, 1800, 10),
        'T_verify': 1200,
    },
    'CuMg': {
        'tdb': 'pycalphad/tests/databases/cumg.tdb',
        'comps': ['CU', 'MG'],
        'T_range': (600, 1100, 10),
        'T_verify': 800,
    },
    'AlMg': {
        'tdb': 'pycalphad/tests/databases/Al-Mg_Zhong.tdb',
        'comps': ['AL', 'MG'],
        'T_range': (400, 900, 10),
        'T_verify': 700,
    },
}


def test_single_system(name, config):
    """Run GPU vs CPU vs pycalphad comparison for one binary system."""
    print(f"\n{'='*70}")
    print(f"  SYSTEM: {name}")
    print(f"{'='*70}")

    dbf = Database(config['tdb'])
    comps = config['comps']
    phases = list(dbf.phases.keys())
    print(f"  Components: {comps}")
    print(f"  Phases: {phases}")

    T_start, T_stop, n_T = config['T_range']
    T_values = np.linspace(T_start, T_stop, n_T)
    T_verify = config['T_verify']

    # ---- 1. CPU grand potential ----
    print(f"\n--- CPU Grand Potential ---")
    try:
        t0 = time.time()
        cpu_result = compute_cpu(
            dbf, comps, phases,
            conditions={v.T: T_values, v.P: 101325},
            mu_resolution=500,
            verbose=False
        )
        t_cpu = time.time() - t0
        cpu_boundaries = extract_phase_boundaries(cpu_result)
        print(f"  Time: {t_cpu:.3f}s for {n_T} temperatures")
        print(f"  Boundaries found: {len(cpu_boundaries)}")
        for bnd in sorted(cpu_boundaries, key=lambda b: b['T']):
            print(f"    T={bnd['T']:.0f}K: {bnd['phase_a']} x={bnd['x_a']:.4f} | "
                  f"{bnd['phase_b']} x={bnd['x_b']:.4f}")
    except Exception as e:
        print(f"  CPU FAILED: {e}")
        traceback.print_exc()
        cpu_result = None
        cpu_boundaries = []

    # ---- 2. GPU grand potential ----
    print(f"\n--- GPU Grand Potential ---")
    try:
        t0 = time.time()
        gpu_result = compute_gpu(
            dbf, comps, phases,
            conditions={v.T: T_values, v.P: 101325},
            mu_resolution=500,
            x_resolution=2000,
            verbose=False
        )
        t_gpu = time.time() - t0
        gpu_boundaries = extract_phase_boundaries(gpu_result)
        skipped = gpu_result.get('skipped_phases', [])
        if skipped:
            print(f"  Skipped phases (multi-sublattice): {skipped}")
        print(f"  GPU phases: {gpu_result['phases']}")
        print(f"  Time: {t_gpu:.3f}s ({gpu_result['compile_time']:.3f}s compile, "
              f"{max(0, gpu_result['compute_time'])*1000:.1f}ms compute)")
        print(f"  Boundaries found: {len(gpu_boundaries)}")
        for bnd in sorted(gpu_boundaries, key=lambda b: b['T']):
            print(f"    T={bnd['T']:.0f}K: {bnd['phase_a']} x={bnd['x_a']:.4f} | "
                  f"{bnd['phase_b']} x={bnd['x_b']:.4f}")
    except Exception as e:
        print(f"  GPU FAILED: {e}")
        traceback.print_exc()
        gpu_result = None
        gpu_boundaries = []

    # ---- 3. Compare CPU vs GPU boundaries ----
    if cpu_boundaries and gpu_boundaries:
        print(f"\n--- CPU vs GPU Comparison ---")
        cpu_by_T = {int(round(b['T'])): b for b in cpu_boundaries}
        gpu_by_T = {int(round(b['T'])): b for b in gpu_boundaries}
        all_T = sorted(set(cpu_by_T.keys()) | set(gpu_by_T.keys()))
        mismatches = 0
        for T in all_T:
            cpu_b = cpu_by_T.get(T)
            gpu_b = gpu_by_T.get(T)
            if cpu_b and gpu_b:
                d_a = abs(cpu_b['x_a'] - gpu_b['x_a'])
                d_b = abs(cpu_b['x_b'] - gpu_b['x_b'])
                ok = d_a < 0.02 and d_b < 0.02
                if not ok:
                    mismatches += 1
                    print(f"  T={T}K: DIFF  x_a: {cpu_b['x_a']:.4f} vs {gpu_b['x_a']:.4f} "
                          f"(d={d_a:.4f}), x_b: {cpu_b['x_b']:.4f} vs {gpu_b['x_b']:.4f} "
                          f"(d={d_b:.4f})")
            elif cpu_b:
                mismatches += 1
                print(f"  T={T}K: CPU-only {cpu_b['phase_a']}/{cpu_b['phase_b']}")
            elif gpu_b:
                print(f"  T={T}K: GPU-only {gpu_b['phase_a']}/{gpu_b['phase_b']}")
        print(f"  Result: {len(all_T) - mismatches}/{len(all_T)} match")

    # ---- 4. Verify against pycalphad at single T ----
    print(f"\n--- Pycalphad Verification at T={T_verify}K ---")
    result_to_check = gpu_result if gpu_result else cpu_result
    if result_to_check is None:
        print("  SKIP (no GP result)")
        return

    # Find boundaries at the verification temperature
    boundaries = None
    for res in result_to_check['results_per_T']:
        if abs(res['T'] - T_verify) < 1:
            boundaries = res['phase_boundaries']
            break

    if not boundaries:
        # Try nearest temperature
        print(f"  No boundaries at T={T_verify}K, trying to find closest...")
        all_bnds = extract_phase_boundaries(result_to_check)
        if all_bnds:
            closest = min(all_bnds, key=lambda b: abs(b['T'] - T_verify))
            T_verify = closest['T']
            print(f"  Using T={T_verify:.0f}K instead")
            for res in result_to_check['results_per_T']:
                if abs(res['T'] - T_verify) < 1:
                    boundaries = res['phase_boundaries']
                    break

    if not boundaries:
        print("  No phase boundaries found for verification")
        return

    # Check GP predictions against pycalphad
    comp_name = comps[1]  # second component
    comps_with_va = sorted(set(comps) | {'VA'})  # pycalphad needs VA for sublattice phases
    mismatches = 0
    total = 0
    x_test = np.arange(0.05, 1.0, 0.05)

    for x_val in x_test:
        try:
            eq = equilibrium(dbf, comps_with_va, phases,
                             {v.T: T_verify, v.P: 101325, v.X(comp_name): x_val})
            stable = []
            for idx in range(eq.Phase.shape[-1]):
                pname = str(eq.Phase.values.squeeze()[idx])
                if pname and pname != '' and pname != 'nan':
                    np_val = float(eq.NP.values.squeeze()[idx])
                    if np_val > 1e-6:
                        stable.append(pname)

            # Determine GP prediction from boundaries
            # Build phase regions from all boundaries at this temperature
            boundary_x_pairs = []
            for bnd in boundaries:
                x_a = bnd['x_phase_a']
                x_b = bnd['x_phase_b']
                # x_a and x_b are composition arrays [x_comp1, x_comp2]
                boundary_x_pairs.append({
                    'phase_a': bnd['phase_low_mu'],
                    'phase_b': bnd['phase_high_mu'],
                    'x_a': x_a[1],  # second component mole fraction
                    'x_b': x_b[1],
                })

            # Simple single-boundary check
            if len(boundary_x_pairs) == 1:
                bnd = boundary_x_pairs[0]
                low_x = min(bnd['x_a'], bnd['x_b'])
                high_x = max(bnd['x_a'], bnd['x_b'])
                if x_val < low_x:
                    gp_pred = bnd['phase_a'] if bnd['x_a'] < bnd['x_b'] else bnd['phase_b']
                elif x_val > high_x:
                    gp_pred = bnd['phase_b'] if bnd['x_b'] > bnd['x_a'] else bnd['phase_a']
                else:
                    gp_pred = "2-phase"
            else:
                # Multiple boundaries - check which region
                gp_pred = "unknown"
                for bnd in sorted(boundary_x_pairs, key=lambda b: min(b['x_a'], b['x_b'])):
                    low_x = min(bnd['x_a'], bnd['x_b'])
                    high_x = max(bnd['x_a'], bnd['x_b'])
                    if low_x <= x_val <= high_x:
                        gp_pred = "2-phase"
                        break

            eq_str = '+'.join(stable)
            match = (gp_pred == eq_str) or (gp_pred == "2-phase" and len(stable) >= 2)
            # Also accept if GP says single phase and pycalphad agrees on the single phase name
            if not match and len(stable) == 1 and gp_pred == stable[0]:
                match = True
            total += 1
            if not match:
                mismatches += 1
                print(f"  MISMATCH at x_{comp_name}={x_val:.2f}: GP={gp_pred}, EQ={eq_str}")
        except Exception as e:
            total += 1
            mismatches += 1
            print(f"  ERROR at x_{comp_name}={x_val:.2f}: {e}")

    print(f"  Result: {total - mismatches}/{total} match ({mismatches} mismatches)")


if __name__ == '__main__':
    systems_to_run = sys.argv[1:] if len(sys.argv) > 1 else list(SYSTEMS.keys())

    for name in systems_to_run:
        if name in SYSTEMS:
            test_single_system(name, SYSTEMS[name])
        else:
            print(f"Unknown system: {name}. Available: {list(SYSTEMS.keys())}")
