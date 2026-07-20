"""Warm-start calibration: does neighbor seeding pay on a dense grid?

Motivation: dense composition sweeps (billions of equilibria in a 6-
component space) revisit the same ~hundred phase regions over and over.
A batched solver would march through the grid seeding each point from an
already-solved neighbor instead of from the convex-hull starting point.
This study measures, on a real system, the three numbers that design
needs:

  1. ITERATION CDF — what solver-iteration budget do warm-started
     points need vs cold (hull-started) points?  (Sweeps PYCGPU_MAXITER
     and counts converged conditions at each budget.)
  2. WRONG-BASIN RATE — when warm-started points converge, how often do
     they converge to a DIFFERENT answer than the faithful cold solve
     (the known ordering-basin failure mode of naive warm starts)?
  3. VERIFICATION EFFICACY — does a global driving-force check
     (max over sampled grid of mu.X - GM, a GEMM) catch the wrong-basin
     points, and how many correct points does it falsely flag?

Mechanics: no solver code is modified. The initial phase data (ipd) the
kernel starts from is injected at the Python packing stage — each
condition gets the CONVERGED state (phase set, amounts, site fractions,
chemical potentials) of its neighbor one grid step back along the
marching axis (the last composition axis). Points in the first slab
have no solved neighbor; they are seeded from their own cold solution
and excluded from the warm statistics.

Usage:
    python warmstart_calibration.py quaternary                 # alcocrni, c++
    python warmstart_calibration.py ternary --backend gpu      # AlCuFe
    python warmstart_calibration.py quaternary --nx 14 --backend gpu
    python warmstart_calibration.py binary --ladder 2 3 5 8 12 20

Interpretation guide (for the batched dense-grid solver decision):
  - Warm CDF far left of cold CDF -> neighbor seeding removes most of
    the work; the dense-grid marching solver is worth building. Note
    the SELF column: the faithful convergence test needs a step ramp
    plus 10 consecutive quiet iterations, so even a perfect seed costs
    ~20-25 budget. If warm tracks self, seeded points are converging at
    that gate floor and the true saving is LARGER than the CDF spread —
    a marching solver (gpu-fast, allowed to deviate) can replace the
    quiet gate with the df acceptance check and bank the difference.
  - Wrong-basin mismatches split three ways: df-caught (the requeue
    fixes them — this is the safety net working), df-clean with tiny
    |dGM| (eps-equivalent alternate optima, the same symmetric
    bifurcation class where reference and backends already disagree),
    and df-clean with large |dGM| (REAL escapes — if these exist, the
    acceptance test needs strengthening before a marching solver can
    be trusted).
  - The cold run's own positive-df tail is the reference's miss rate on
    this grid (it is not zero on many-phase systems); judge the warm
    run against that baseline, not against perfection.
  - Watch the escape SHAPE: mismatches like (P, P) -> (P,) — a lost
    second composition set of the same phase — carry NEGATIVE grid-df,
    because a finite grid scan cannot see inside ordering/miscibility
    gaps (the same reason the gpu-fast lockstep pass needs the
    continuous driving-force check). A production marching solver must
    use the in-kernel continuous-df acceptance, not this grid scan;
    this script's grid scan is the cheap outer net only.
"""
import argparse
import contextlib
import io
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')

# Pin the pipeline into the configuration the study depends on BEFORE
# pycalphad imports: host-side ipd packing (that is the injection point),
# no condition sorting (keeps condition order = C-order over the grid),
# no two-pass (PYCGPU_MAXITER must be the ONLY iteration budget).
os.environ['PYCGPU_DEVICE_PIPE'] = '0'
os.environ['PYCGPU_SORT'] = '0'
os.environ['PYCGPU_PASS1_ITERS'] = '0'
os.environ.pop('PYCGPU_GPU_FAST', None)

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DB_DIR = os.path.normpath(os.path.join(_HERE, '..', '..', 'databases'))


def _quaternary_db():
    import pycalphad.tests
    return os.path.join(os.path.dirname(pycalphad.tests.__file__),
                        'databases', 'alcocrni.tdb')


# (database, components, T, [(element, lo, hi)], default points/axis)
SYSTEMS = {
    'binary': (os.path.join(_DB_DIR, 'AuBi-07Wan.tdb'), ['AU', 'BI', 'VA'],
               600.0, [('BI', 0.01, 0.99)], 400),
    'ternary': (os.path.join(_DB_DIR, 'Al-Cu-Fe.tdb'), ['AL', 'CU', 'FE', 'VA'],
                1100.0, [('CU', 0.02, 0.60), ('FE', 0.02, 0.60)], 24),
    'quaternary': (None, ['AL', 'CO', 'CR', 'NI', 'VA'],
                   1300.0, [('AL', 0.05, 0.30), ('CO', 0.05, 0.30),
                            ('CR', 0.05, 0.30)], 10),
}

STABLE_TOL = 1e-10  # NP threshold for "phase is present" (matches result NaN-padding)


class RawCapture:
    """Capture the kernel's raw per-condition results (pre-xarray), in
    condition order: structured fields phase_ids / NP / Y_phases /
    X_phases / final_chemical_potentials / final_system_gm / converged,
    plus the unique-model list and dynamic sizes the run used."""

    def __init__(self):
        self.raw = None
        self.models = None
        self.sizes = None

    @contextlib.contextmanager
    def active(self):
        import pycalphad.gpu.gpu_equilibrium as ge
        orig = ge._process_gpu_results

        def wrapper(results_cpu_flat, wks_obj, num_conditions_total,
                    unique_py_models, py_phase_name_to_unique_idx_map,
                    original_properties=None, dynamic_sizes=None):
            self.raw = np.copy(results_cpu_flat)
            self.models = list(unique_py_models)
            self.sizes = dict(dynamic_sizes) if dynamic_sizes else None
            return orig(results_cpu_flat, wks_obj, num_conditions_total,
                        unique_py_models, py_phase_name_to_unique_idx_map,
                        original_properties=original_properties,
                        dynamic_sizes=dynamic_sizes)

        ge._process_gpu_results = wrapper
        try:
            yield self
        finally:
            ge._process_gpu_results = orig


@contextlib.contextmanager
def seeded_ipd(seed):
    """Replace the hull-derived initial phase data with `seed` (a dict in
    the same layout as the pipeline's initial_phase_data_arrays)."""
    import pycalphad.gpu.gpu_equilibrium as ge
    orig = ge._create_initial_phase_data_struct_array

    def wrapper(arrays, num_conditions, dynamic_sizes=None, verbose=False):
        assert arrays['phase_indices'].shape == seed['phase_indices'].shape, (
            f"seed shape {seed['phase_indices'].shape} != run shape "
            f"{arrays['phase_indices'].shape}: grid/system mismatch between "
            "the reference run and the seeded run")
        for key in ('phase_indices', 'phase_amounts', 'site_fractions',
                    'compositions', 'chemical_potentials', 'num_phases'):
            arrays[key][...] = seed[key]
        return orig(arrays, num_conditions, dynamic_sizes, verbose)

    ge._create_initial_phase_data_struct_array = wrapper
    try:
        yield
    finally:
        ge._create_initial_phase_data_struct_array = orig


def solve(dbf, comps, phases, conds, budget_iters, seed=None):
    """One accelerated equilibrium solve at the given iteration budget,
    optionally warm-started from `seed`. Returns (raw_capture, wall_s)."""
    from pycalphad import equilibrium
    os.environ['PYCGPU_MAXITER'] = str(int(budget_iters))
    cap = RawCapture()
    t0 = time.time()
    with cap.active():
        ctx = seeded_ipd(seed) if seed is not None else contextlib.nullcontext()
        with ctx, contextlib.redirect_stdout(io.StringIO()):
            eq = equilibrium(dbf, comps, phases, conds)
    wall = time.time() - t0
    os.environ.pop('PYCGPU_MAXITER', None)
    if cap.raw is None:
        raise RuntimeError('accelerated path did not run (raw capture empty) — '
                           'is the backend set and the system supported?')
    return cap, eq, wall


def atoms_per_formula_table(models, max_dof):
    """(n_models, max_dof) weights: moles of atoms contributed per unit of
    each site-fraction dof (site ratio x atoms in species, 0 for VA)."""
    tbl = np.zeros((len(models), max_dof))
    for mi, mod in enumerate(models):
        ratios = [float(r) for r in mod.site_ratios]
        for k, sf in enumerate(mod.site_fractions):
            if k >= max_dof:
                break
            sp = sf.species
            if sp.name == 'VA':
                continue
            n_atoms = sum(v for el, v in sp.constituents.items() if el != 'VA')
            tbl[mi, k] = ratios[sf.sublattice_index] * float(n_atoms)
    return tbl


def build_seed(raw, nbr_idx, models, sizes):
    """ipd arrays seeding every condition from raw[nbr_idx[c]]'s converged
    state. Amount conversion: result NP is moles of atoms; ipd amounts are
    formula units (phase_amt), so divide by atoms-per-formula-unit(Y)."""
    mp, mc, md = sizes['MAX_PHASES'], sizes['MAX_COMPONENTS'], sizes['MAX_DOF_PER_PHASE']
    n = len(nbr_idx)
    pid = raw['phase_ids'][nbr_idx].astype(np.int64)          # (n, mp)
    np_m = raw['NP'][nbr_idx]                                  # moles of atoms
    Y = raw['Y_phases'][nbr_idx].reshape(n, mp, md)
    X = raw['X_phases'][nbr_idx].reshape(n, mp, mc)
    mu = raw['final_chemical_potentials'][nbr_idx]

    stable = (np_m > STABLE_TOL) & (pid >= 0)
    apf_tbl = atoms_per_formula_table(models, md)              # (n_models, md)
    apf = np.einsum('cpk,cpk->cp', Y, apf_tbl[np.clip(pid, 0, len(models) - 1)])
    apf = np.where(apf > 1e-12, apf, 1.0)
    amounts = np.where(stable, np_m / apf, 0.0)

    # compact stable compsets to the leading slots (kernel reads the first
    # num_phases slots)
    order = np.argsort(~stable, axis=1, kind='stable')
    take = lambda a: np.take_along_axis(a, order.reshape(order.shape + (1,) * (a.ndim - 2)), axis=1)
    pid_c = np.take_along_axis(pid, order, axis=1)
    amt_c = np.take_along_axis(amounts, order, axis=1)
    stable_c = np.take_along_axis(stable, order, axis=1)
    Y_c = take(Y)
    X_c = take(X)
    pid_c = np.where(stable_c, pid_c, -1)
    amt_c = np.where(stable_c, amt_c, 0.0)

    return {
        'phase_indices': pid_c.astype(np.int32),
        'phase_amounts': amt_c,
        'site_fractions': np.where(stable_c[..., None], Y_c, 0.0),
        'compositions': np.where(stable_c[..., None], X_c, 0.0),
        'chemical_potentials': mu.copy(),
        'num_phases': stable_c.sum(axis=1).astype(np.float64),
    }


def assemblages(raw, models):
    """Per-condition sorted tuple of stable phase names (multiset: repeated
    names = miscibility-gap compsets)."""
    names = np.array([m.phase_name for m in models] + ['-'])
    pid = raw['phase_ids'].astype(np.int64)
    stable = (raw['NP'] > STABLE_TOL) & (pid >= 0)
    out = []
    for c in range(pid.shape[0]):
        out.append(tuple(sorted(names[pid[c][stable[c]]])))
    return out


def df_scan(raw, grid_gm, grid_x, n_comps, chunk=2048):
    """Max driving force over the sampled grid for each condition's
    converged hyperplane: max_i mu.X_i - GM_i. Positive = some phase
    undercuts the claimed equilibrium (reject)."""
    mu = raw['final_chemical_potentials'][:, :n_comps]
    out = np.empty(mu.shape[0])
    for s in range(0, mu.shape[0], chunk):
        e = min(s + chunk, mu.shape[0])
        # (chunk, n_comps) @ (n_comps, n_grid) - (n_grid,)
        out[s:e] = (mu[s:e] @ grid_x.T - grid_gm).max(axis=1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('system', choices=sorted(SYSTEMS))
    ap.add_argument('--backend', default='c++', choices=['c++', 'gpu'],
                    help='accelerated backend to study (default c++; results '
                         'are algorithmic, so c++ answers the same questions '
                         'without a GPU)')
    ap.add_argument('--nx', type=int, default=None,
                    help='points per composition axis (default per system)')
    ap.add_argument('--ladder', type=int, nargs='+',
                    default=[10, 15, 20, 25, 30, 40, 60, 100, 200],
                    help='iteration budgets for the convergence CDF. NOTE: the '
                         'faithful convergence test needs ~10 consecutive quiet '
                         'iterations after the step ramp, so nothing converges '
                         'below ~20-25 regardless of start; the self-seed '
                         'column shows that floor directly')
    ap.add_argument('--full-iters', type=int, default=1000,
                    help='full budget for the reference/agreement runs')
    ap.add_argument('--df-tol', type=float, default=1e-3,
                    help='driving-force acceptance tolerance in J (default 1e-3)')
    ap.add_argument('--save', default=None, help='write raw arrays to this .npz')
    args = ap.parse_args()

    from pycalphad import Database, calculate, variables as v, set_backend
    set_backend(args.backend)

    db_path, comps, T, x_axes, nx_default = SYSTEMS[args.system]
    if db_path is None:
        db_path = _quaternary_db()
    nx = args.nx or nx_default
    dbf = Database(db_path)
    phases = sorted(dbf.phases.keys())
    conds = {v.N: 1, v.P: 101325.0, v.T: T}
    for el, lo, hi in x_axes:
        conds[v.X(el)] = np.linspace(lo, hi, nx)
    n_cond = nx ** len(x_axes)
    print(f'{args.system}: {os.path.basename(db_path)}, T={T} K, '
          f'{len(x_axes)} composition axes x {nx} points = {n_cond} conditions, '
          f'backend {args.backend}')

    # --- reference cold solve (full budget) -------------------------------
    print(f'\n[1/4] cold reference solve (budget {args.full_iters})...', flush=True)
    cap_ref, eq_ref, wall_ref = solve(dbf, comps, phases, conds, args.full_iters)
    raw_ref, models, sizes = cap_ref.raw, cap_ref.models, cap_ref.sizes
    shape = eq_ref.GM.values.shape
    assert int(np.prod(shape)) == n_cond == raw_ref.shape[0]
    # pin the raw-order == C-order-over-dims assumption before using it
    assert np.allclose(raw_ref['final_system_gm'].reshape(shape),
                       eq_ref.GM.values, equal_nan=True), \
        'raw condition order does not match the result grid order'
    conv_ref = raw_ref['converged']
    print(f'      {conv_ref.sum()}/{n_cond} converged, {wall_ref:.1f}s wall '
          f'({1000 * wall_ref / n_cond:.2f} ms/cond incl. grid+hull)')

    # neighbor map: one step back along the LAST grid axis (marching order);
    # first-slab points (index 0 on that axis) self-seed and are excluded
    # from warm statistics.
    idx = np.arange(n_cond).reshape(shape)
    nbr = np.concatenate([idx[..., :1], idx[..., :-1]], axis=-1).reshape(-1)
    first_slab = (idx == nbr.reshape(shape)).reshape(-1)
    interior = ~first_slab
    seed = build_seed(raw_ref, nbr, models, sizes)

    asm_ref = assemblages(raw_ref, models)
    n_regions = len(set(asm_ref))
    # boundary = assemblage differs from the marching neighbor
    boundary = np.array([asm_ref[c] != asm_ref[nbr[c]] for c in range(n_cond)])
    print(f'      {n_regions} distinct assemblages; '
          f'{boundary[interior].sum()}/{interior.sum()} interior points sit on '
          f'a marching-direction region boundary '
          f'({100 * boundary[interior].mean():.1f}%)')

    # --- warm full-budget solve + agreement -------------------------------
    print(f'\n[2/4] warm full-budget solve (neighbor-seeded)...', flush=True)
    cap_w, eq_w, wall_w = solve(dbf, comps, phases, conds, args.full_iters, seed=seed)
    raw_warm = cap_w.raw
    asm_warm = assemblages(raw_warm, models)
    gm_diff = np.abs(raw_warm['final_system_gm'] - raw_ref['final_system_gm'])
    mismatch = np.array([asm_warm[c] != asm_ref[c] for c in range(n_cond)])
    both_conv = conv_ref & raw_warm['converged']
    wrong = mismatch & both_conv & interior
    print(f'      {raw_warm["converged"].sum()}/{n_cond} converged, {wall_w:.1f}s wall')
    print(f'      phase-set agreement (interior, both converged): '
          f'{(both_conv & interior).sum() - wrong.sum()}/{(both_conv & interior).sum()}'
          f'  -> WRONG-BASIN RATE {100 * wrong.sum() / max((both_conv & interior).sum(), 1):.2f}%')
    agree_mask = both_conv & interior & ~mismatch
    if agree_mask.any():
        print(f'      |dGM| over agreeing points: max {gm_diff[agree_mask].max():.3e} J')
    if wrong.any():
        print(f'      wrong points on region boundaries: '
              f'{(wrong & boundary).sum()}/{wrong.sum()}')

    # --- driving-force verification ---------------------------------------
    print(f'\n[3/4] driving-force verification scan (grid GEMM)...', flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        grid = calculate(dbf, comps, phases, T=T, P=101325.0, N=1)
    n_comps = eq_ref.MU.shape[-1]
    grid_gm = np.asarray(grid.GM.values).reshape(-1)
    grid_x = np.asarray(grid.X.values).reshape(-1, n_comps)
    ok = np.isfinite(grid_gm)
    grid_gm, grid_x = grid_gm[ok], grid_x[ok]
    t0 = time.time()
    df_ref = df_scan(raw_ref, grid_gm, grid_x, n_comps)
    df_warm = df_scan(raw_warm, grid_gm, grid_x, n_comps)
    t_df = time.time() - t0
    q = lambda a: ' / '.join(f'{np.quantile(a, p):.2e}' for p in (0.5, 0.99, 1.0))
    print(f'      {len(grid_gm)} grid points, scan {t_df:.2f}s for 2x{n_cond} conditions')
    print(f'      cold max-df  (median/p99/max): {q(df_ref[conv_ref])}  '
          f'(baseline noise of a correct solve)')
    print(f'      warm max-df  (median/p99/max): {q(df_warm[both_conv])}')
    ref_misses = (df_ref > args.df_tol) & conv_ref
    if ref_misses.any():
        print(f'      NOTE: {ref_misses.sum()} converged REFERENCE points have '
              f'positive df (the reference itself missed a lower state there — '
              f'the known symmetric-bifurcation class); treat those as the '
              f'baseline miss rate, not warm-start damage')
    reject = df_warm > args.df_tol
    caught = (reject & wrong).sum()
    false_flags = (reject & interior & both_conv & ~mismatch).sum()
    n_agree = (interior & both_conv & ~mismatch).sum()
    print(f'      at tol {args.df_tol:g} J: rejects {reject[interior].sum()}'
          f'/{interior.sum()} interior points')
    print(f'      CATCH RATE  : {caught}/{wrong.sum()} wrong-basin points caught'
          if wrong.any() else '      CATCH RATE  : n/a (no wrong-basin points)')
    print(f'      FALSE FLAGS : {false_flags}/{n_agree} correct points flagged '
          f'({100 * false_flags / max(n_agree, 1):.2f}%)')
    # split the mismatches: df-caught (requeue fixes them) vs df-clean.
    # A df-clean mismatch with tiny |dGM| is an eps-equivalent alternate
    # optimum (both answers sit on hyperplanes nothing undercuts) — the
    # class where reference and backend disagree symmetrically anyway.
    if wrong.any():
        benign = wrong & ~reject & (gm_diff < 1.0)
        serious = wrong & ~reject & (gm_diff >= 1.0)
        print(f'      mismatch breakdown: {caught} df-caught (requeued), '
              f'{benign.sum()} df-clean eps-equivalent (|dGM|<1 J), '
              f'{serious.sum()} df-clean with |dGM|>=1 J (REAL escapes)')
        for c in np.where(wrong)[0][:10]:
            tag = ('caught' if reject[c] else
                   'eps-equiv' if gm_diff[c] < 1.0 else 'ESCAPE')
            print(f'        cond {c}: {asm_ref[c]} -> {asm_warm[c]}, '
                  f'|dGM|={gm_diff[c]:.2e} J, df={df_warm[c]:+.2e} J [{tag}]')

    # --- iteration-budget CDF ---------------------------------------------
    # self-seed control: every point seeded with its OWN converged state —
    # the best any warm start can do; where it converges is the convergence
    # gate's floor (~step ramp + 10 quiet iterations), not real work.
    seed_self = build_seed(raw_ref, np.arange(n_cond), models, sizes)
    print(f'\n[4/4] convergence CDF over iteration budgets {args.ladder}...', flush=True)
    print(f'      {"budget":>8} {"cold %":>8} {"warm %":>8} {"self %":>8}'
          f'   (warm/self exclude first slab; self = gate floor)')
    ladder_rows = []
    for b in args.ladder:
        cap_c, _, _ = solve(dbf, comps, phases, conds, b)
        cap_h, _, _ = solve(dbf, comps, phases, conds, b, seed=seed)
        cap_s, _, _ = solve(dbf, comps, phases, conds, b, seed=seed_self)
        c_frac = 100 * cap_c.raw['converged'].mean()
        w_frac = 100 * cap_h.raw['converged'][interior].mean()
        s_frac = 100 * cap_s.raw['converged'][interior].mean()
        ladder_rows.append((b, c_frac, w_frac, s_frac))
        print(f'      {b:>8} {c_frac:>7.1f}% {w_frac:>7.1f}% {s_frac:>7.1f}%', flush=True)

    print('\n=== summary ===')
    w90 = next((b for b, _, w, _ in ladder_rows if w >= 90), None)
    c90 = next((b for b, c, _, _ in ladder_rows if c >= 90), None)
    s90 = next((b for b, _, _, s in ladder_rows if s >= 90), None)
    print(f'  budget for 90% convergence: cold {c90 or ">" + str(args.ladder[-1])}, '
          f'warm {w90 or ">" + str(args.ladder[-1])}, '
          f'gate floor (self-seed) {s90 or ">" + str(args.ladder[-1])}')
    # the 90% crossing is set by the straggler TAIL, which neither start
    # helps — the warm-start payoff is the BULK converging at the gate
    # floor. Report the separation there so the summary carries it.
    b_floor = s90 or args.ladder[-1]
    row = next((r for r in ladder_rows if r[0] == b_floor), ladder_rows[-1])
    print(f'  bulk separation at the gate-floor budget ({row[0]} iters): '
          f'cold {row[1]:.1f}% vs warm {row[2]:.1f}% converged '
          f'(warm-start payoff = this gap; the shared tail past 90% is '
          f'straggler work for the pass-2/semismooth lever, not seeding)')
    n_serious = int((wrong & ~reject & (gm_diff >= 1.0)).sum()) if wrong.any() else 0
    print(f'  wrong-basin rate: {100 * wrong.sum() / max((both_conv & interior).sum(), 1):.2f}% '
          f'({caught} df-caught, {n_serious} real escapes)')
    print(f'  distinct assemblages: {n_regions}; boundary fraction '
          f'{100 * boundary[interior].mean():.1f}%')
    print('  -> see module docstring for how these three numbers feed the '
          'dense-grid marching-solver decision.')

    if args.save:
        np.savez_compressed(
            args.save, shape=shape, ladder=np.array(ladder_rows),
            gm_ref=raw_ref['final_system_gm'], gm_warm=raw_warm['final_system_gm'],
            conv_ref=conv_ref, conv_warm=raw_warm['converged'],
            mismatch=mismatch, boundary=boundary, interior=interior,
            df_ref=df_ref, df_warm=df_warm)
        print(f'  raw arrays written to {args.save}')


if __name__ == '__main__':
    main()
