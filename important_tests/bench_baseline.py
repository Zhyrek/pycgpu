#!/usr/bin/env python
"""Baseline CPU vs GPU benchmark. NO pdens (per CLAUDE.md, comparisons must omit it).

Usage: python bench_baseline.py [aubi|alcufe] [nx] [nt]
Runs CPU once, GPU twice (cold = includes compile, warm = cached module),
compares GM/MU/stable-phase sets, prints timing and correctness summary.
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import warnings
warnings.filterwarnings("ignore")
from pycalphad import Database, equilibrium, variables as v

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup(system, nx, nt):
    if system == 'aubi':
        dbf = Database(os.path.join(ROOT, 'AuBi-07Wan.tdb'))
        comps = ['AU', 'BI', 'VA']
        els = ['AU', 'BI']
        conds = {v.X('BI'): (0.05, 0.951, 0.9 / (nx - 1)),
                 v.T: (500, 1000.1, 500.0 / (nt - 1)),
                 v.P: 101325, v.N: 1}
    elif system == 'alcufe':
        dbf = Database(os.path.join(ROOT, 'Al-Cu-Fe.tdb'))
        comps = ['AL', 'CU', 'FE', 'VA']
        els = ['AL', 'CU', 'FE']
        conds = {v.X('AL'): (0.2, 0.61, 0.4 / max(nx - 1, 1)),
                 v.X('CU'): (0.1, 0.41, 0.3 / max(nx - 1, 1)),
                 v.T: (700, 1400.1, 700.0 / max(nt - 1, 1)),
                 v.P: 101325, v.N: 1}
    else:
        raise ValueError(system)
    phases = sorted(dbf.phases.keys())
    return dbf, comps, els, phases, conds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('system', nargs='?', default='aubi', choices=['aubi', 'alcufe'])
    ap.add_argument('nx', nargs='?', type=int, default=11)
    ap.add_argument('nt', nargs='?', type=int, default=5)
    ap.add_argument('--skip-cpu', action='store_true')
    args = ap.parse_args()

    dbf, comps, els, phases, conds = setup(args.system, args.nx, args.nt)
    print(f"System: {args.system}  phases={len(phases)}  grid nx={args.nx} nt={args.nt}")

    t0 = time.time()
    cpu = None
    if not args.skip_cpu:
        cpu = equilibrium(dbf, comps, phases, conds, gpu=False)
        t_cpu = time.time() - t0
        print(f"CPU time:      {t_cpu:8.2f} s   ({cpu.GM.values.size} conditions)")

    t0 = time.time()
    gpu1 = equilibrium(dbf, comps, phases, conds, gpu=True)
    t_cold = time.time() - t0
    print(f"GPU cold time: {t_cold:8.2f} s")

    t0 = time.time()
    gpu2 = equilibrium(dbf, comps, phases, conds, gpu=True)
    t_warm = time.time() - t0
    print(f"GPU warm time: {t_warm:8.2f} s")
    if not args.skip_cpu:
        print(f"Speedup (warm): {t_cpu / t_warm:6.1f}x")

    # GPU run-to-run consistency
    g1, g2 = gpu1.GM.values.flatten(), gpu2.GM.values.flatten()
    m = ~(np.isnan(g1) | np.isnan(g2))
    print(f"GPU run-to-run max |dGM|: {np.max(np.abs(g1[m] - g2[m])) if m.any() else float('nan'):.3e}")

    if args.skip_cpu:
        return

    # Correctness vs CPU
    c = cpu.GM.values.flatten()
    m = ~(np.isnan(c) | np.isnan(g1))
    gm_diff = np.abs(c[m] - g1[m])
    print(f"\nGM:  n={m.sum()}  max diff={gm_diff.max():.6g}  mean={gm_diff.mean():.6g}")
    for el in els:
        cm = cpu.MU.sel(component=el).values.flatten()
        gm_ = gpu1.MU.sel(component=el).values.flatten()
        mm = ~(np.isnan(cm) | np.isnan(gm_))
        d = np.abs(cm[mm] - gm_[mm])
        print(f"MU_{el}: max diff={d.max():.6g}")

    cnp = cpu.NP.values.reshape(-1, cpu.NP.shape[-1])
    gnp = gpu1.NP.values.reshape(-1, gpu1.NP.shape[-1])
    # phase name per vertex
    cph = cpu.Phase.values.reshape(-1, cpu.Phase.shape[-1])
    gph = gpu1.Phase.values.reshape(-1, gpu1.Phase.shape[-1])
    n_mismatch = 0
    n_valid = 0
    mismatches = []
    for i in range(cnp.shape[0]):
        if np.isnan(c[i]) or np.isnan(g1[i]):
            continue
        n_valid += 1
        cs = {cph[i, j] for j in range(cnp.shape[1]) if cnp[i, j] > 1e-6 and cph[i, j] != ''}
        gs = {gph[i, j] for j in range(gnp.shape[1]) if gnp[i, j] > 1e-6 and gph[i, j] != ''}
        if cs != gs:
            n_mismatch += 1
            if len(mismatches) < 15:
                mismatches.append((i, sorted(cs), sorted(gs), abs(c[i] - g1[i])))
    print(f"\nStable-phase sets: {n_valid - n_mismatch}/{n_valid} match ({n_mismatch} mismatches)")
    for i, cs, gs, d in mismatches:
        print(f"  cond {i}: CPU={cs} GPU={gs} dGM={d:.4g}")


if __name__ == '__main__':
    main()
