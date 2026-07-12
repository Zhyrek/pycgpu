"""
Test GPU grand potential solver against CPU version and pycalphad equilibrium.
"""

import sys
import os
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pycalphad import Database, equilibrium, variables as v
from advanced_strategy.grand_potential import (
    compute_phase_diagram as compute_cpu,
    extract_phase_boundaries,
)
from advanced_strategy.grand_potential_gpu import (
    compute_phase_diagram_gpu as compute_gpu,
    _build_kernel_source,
)
from pycalphad import Model


def test_kernel_generation():
    """Test that CUDA kernel generation works for NbTi."""
    print("=" * 70)
    print("TEST: Kernel generation")
    print("=" * 70)

    dbf = Database('NbTi.tdb')
    comps = sorted(['NB', 'TI', 'VA'])
    phases = list(dbf.phases.keys())

    models = {ph: Model(dbf, comps, ph) for ph in phases}
    nonvacant = ['NB', 'TI']

    kernel_source = _build_kernel_source(models, phases, nonvacant)

    # Print first 100 lines for inspection
    lines = kernel_source.split('\n')
    print(f"Kernel: {len(lines)} lines, {len(kernel_source)} chars")
    print("\n--- First 80 lines ---")
    for i, line in enumerate(lines[:80]):
        print(f"{i+1:4d}: {line}")

    # Save full kernel for debugging
    with open('/tmp/gp_kernel_nbti.cu', 'w') as f:
        f.write(kernel_source)
    print(f"\nFull kernel saved to /tmp/gp_kernel_nbti.cu")


def test_gpu_single_temperature():
    """Test GPU solver at a single temperature."""
    print("\n" + "=" * 70)
    print("TEST: GPU solver at T=2200K")
    print("=" * 70)

    dbf = Database('NbTi.tdb')
    comps = ['NB', 'TI']
    phases = list(dbf.phases.keys())

    result = compute_gpu(
        dbf, comps, phases,
        conditions={v.T: 2200, v.P: 101325},
        mu_resolution=500,
        x_resolution=2000,
        verbose=True
    )

    res = result['results_per_T'][0]
    boundaries = res['phase_boundaries']

    if boundaries:
        bnd = boundaries[0]
        print(f"\nGPU boundaries:")
        print(f"  Solidus  (BCC_A2): x_TI = {bnd['x_phase_a'][1]:.6f}")
        print(f"  Liquidus (LIQUID): x_TI = {bnd['x_phase_b'][1]:.6f}")
    else:
        print("No boundaries found")


def test_gpu_vs_cpu_accuracy():
    """Compare GPU and CPU grand potential results."""
    print("\n" + "=" * 70)
    print("TEST: GPU vs CPU accuracy")
    print("=" * 70)

    dbf = Database('NbTi.tdb')
    comps = ['NB', 'TI']
    phases = list(dbf.phases.keys())

    T_values = np.arange(2000, 2800, 100)

    # CPU
    t0 = time.time()
    cpu_result = compute_cpu(
        dbf, comps, phases,
        conditions={v.T: T_values, v.P: 101325},
        mu_resolution=500,
        verbose=False
    )
    t_cpu = time.time() - t0
    cpu_boundaries = extract_phase_boundaries(cpu_result)

    # GPU
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

    print(f"\nCPU time: {t_cpu:.3f}s ({len(T_values)} temps)")
    print(f"GPU time: {t_gpu:.3f}s ({len(T_values)} temps)")
    print(f"  (GPU compile: {gpu_result['compile_time']:.3f}s, "
          f"compute: {gpu_result['compute_time']*1000:.1f}ms)")

    # Compare boundaries
    print(f"\n{'T (K)':>8s}  {'CPU solidus':>12s}  {'GPU solidus':>12s}  "
          f"{'CPU liquidus':>12s}  {'GPU liquidus':>12s}  {'Match':>6s}")
    print("-" * 80)

    cpu_by_T = {int(b['T']): b for b in cpu_boundaries}
    gpu_by_T = {int(b['T']): b for b in gpu_boundaries}

    all_temps = sorted(set(cpu_by_T.keys()) | set(gpu_by_T.keys()))
    mismatches = 0

    for T in all_temps:
        cpu_b = cpu_by_T.get(T)
        gpu_b = gpu_by_T.get(T)

        if cpu_b and gpu_b:
            cpu_s = cpu_b['x_a']
            cpu_l = cpu_b['x_b']
            gpu_s = gpu_b['x_a']
            gpu_l = gpu_b['x_b']
            match = abs(cpu_s - gpu_s) < 0.02 and abs(cpu_l - gpu_l) < 0.02
            if not match:
                mismatches += 1
            print(f"{T:8d}  {cpu_s:12.6f}  {gpu_s:12.6f}  "
                  f"{cpu_l:12.6f}  {gpu_l:12.6f}  "
                  f"{'OK' if match else 'DIFF':>6s}")
        elif cpu_b:
            print(f"{T:8d}  {cpu_b['x_a']:12.6f}  {'---':>12s}  "
                  f"{cpu_b['x_b']:12.6f}  {'---':>12s}  {'MISS':>6s}")
            mismatches += 1
        elif gpu_b:
            print(f"{T:8d}  {'---':>12s}  {gpu_b['x_a']:12.6f}  "
                  f"{'---':>12s}  {gpu_b['x_b']:12.6f}  {'EXTRA':>6s}")

    print(f"\n{len(all_temps) - mismatches}/{len(all_temps)} temperatures match")


def test_gpu_vs_pycalphad():
    """Compare GPU grand potential against pycalphad equilibrium."""
    print("\n" + "=" * 70)
    print("TEST: GPU vs pycalphad verification")
    print("=" * 70)

    dbf = Database('NbTi.tdb')
    comps = ['NB', 'TI']
    phases = list(dbf.phases.keys())

    T = 2200
    gpu_result = compute_gpu(
        dbf, comps, phases,
        conditions={v.T: T, v.P: 101325},
        mu_resolution=500,
        x_resolution=2000,
        verbose=False
    )

    boundaries = gpu_result['results_per_T'][0]['phase_boundaries']
    if not boundaries:
        print("No GPU boundaries found")
        return

    bnd = boundaries[0]
    x_solidus = bnd['x_phase_a'][1]
    x_liquidus = bnd['x_phase_b'][1]
    print(f"GPU: solidus={x_solidus:.6f}, liquidus={x_liquidus:.6f}")

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
            print(f"  MISMATCH at x_TI={x_ti:.2f}: GPU={gp_pred}, EQ={eq_str}")

    print(f"Result: {total - mismatches}/{total} match ({mismatches} mismatches)")


def test_gpu_speed_scaling():
    """Measure GPU performance at increasing scale."""
    print("\n" + "=" * 70)
    print("TEST: GPU speed at scale")
    print("=" * 70)

    dbf = Database('NbTi.tdb')
    comps = ['NB', 'TI']
    phases = list(dbf.phases.keys())

    # Warm up (includes compilation)
    _ = compute_gpu(
        dbf, comps, phases,
        conditions={v.T: 2000, v.P: 101325},
        mu_resolution=100,
        x_resolution=500,
        verbose=False
    )

    configs = [
        (10, 200, 1000),
        (50, 500, 2000),
        (100, 500, 2000),
        (250, 500, 2000),
        (500, 500, 2000),
        (1000, 500, 2000),
    ]

    print(f"\n{'n_T':>6s}  {'n_mu':>6s}  {'n_x':>6s}  {'Total Evals':>14s}  "
          f"{'GPU time':>10s}  {'Throughput':>14s}")
    print("-" * 75)

    for n_T, n_mu, n_x in configs:
        T_values = np.linspace(1800, 2800, n_T)

        t0 = time.time()
        result = compute_gpu(
            dbf, comps, phases,
            conditions={v.T: T_values, v.P: 101325},
            mu_resolution=n_mu,
            x_resolution=n_x,
            verbose=False
        )
        t_total = time.time() - t0
        t_compute = result['compute_time']

        total_evals = n_T * n_mu * n_x * len(phases)
        throughput = total_evals / t_compute

        print(f"{n_T:6d}  {n_mu:6d}  {n_x:6d}  {total_evals:14,d}  "
              f"{t_compute*1000:8.1f}ms  {throughput/1e9:11.2f} B/s")

    # Compare with pycalphad at the largest scale
    print(f"\n--- pycalphad comparison at n_T=100, n_x=500 ---")
    # pycalphad needs one call per (T, x) pair
    # Sample a few to estimate
    sample_conds = [(2000 + i*100, 0.1 + j*0.2) for i in range(3) for j in range(3)]
    t0 = time.time()
    for T, x in sample_conds:
        _ = equilibrium(dbf, comps, phases,
                        {v.T: T, v.P: 101325, v.X('TI'): x})
    t_sample = time.time() - t0
    per_eq = t_sample / len(sample_conds)

    n_equiv = 100 * 500  # equivalent coverage
    t_equiv = per_eq * n_equiv
    gp_time = configs[2][0]  # 100 temp config

    print(f"  pycalphad per-point: {per_eq*1000:.1f}ms")
    print(f"  Equivalent {n_equiv} points: {t_equiv:.1f}s (estimated)")


if __name__ == '__main__':
    test_kernel_generation()
    test_gpu_single_temperature()
    test_gpu_vs_cpu_accuracy()
    test_gpu_vs_pycalphad()
    test_gpu_speed_scaling()
