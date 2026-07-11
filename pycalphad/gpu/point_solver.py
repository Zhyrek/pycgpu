"""Point-list batch solving: per-point input-array builders.

Solves an arbitrary LIST of conditions (T, P, N, X-vector per point) in one
kernel launch, with optional per-point phase restriction (single-phase
"isolated" solves) and per-point fit-parameter vectors (walker batching).
See POINT_SOLVER_DESIGN.md for the verified array-format notes.

This module deliberately does NOT modify the reference CPU code: the hull
twin below calls the same Cython ``hyperplane()`` routine the reference
``lower_convex_hull`` uses, over point arrays instead of a cartesian grid.
"""
import hashlib
import os
from types import SimpleNamespace

import numpy as np

from pycalphad.core.hyperplane import hyperplane

MIN_PHASE_FRACTION = 1e-6


def _combo_of_point(points, n_combos, grid_T=None, grid_P=None):
    """Map each point to its statevar-combo index in the grid's C-order
    (P axis outer, T axis inner when both vary). grid_T/grid_P override the
    axes when the grid covers MORE combos than the point list touches."""
    uT = np.asarray(grid_T) if grid_T is not None else np.unique(points.T)
    uP = np.asarray(grid_P) if grid_P is not None else np.unique(points.P)
    if n_combos == len(uT):
        return np.searchsorted(uT, points.T)
    if n_combos == len(uP) * len(uT):
        return (np.searchsorted(uP, points.P) * len(uT)
                + np.searchsorted(uT, points.T))
    raise ValueError(f"grid has {n_combos} statevar combos; expected "
                     f"{len(uT)} or {len(uP) * len(uT)}")


class PointList:
    """A batch of independent condition points for one system.

    Parameters
    ----------
    T, P, N : (n,) float64 arrays (N is moles, typically ones)
    X : (n, n_nonvacant) float64 — full overall composition per point in
        sorted nonvacant pure-element order (rows sum to 1).
    x_cond_mask : (n, n_nonvacant) bool — which components are PRESCRIBED
        mole-fraction conditions for this point (exactly n_nonvacant-1 True
        per row for a fully-determined problem).
    phase_restrict : (n,) object/None — phase name the point is restricted
        to, or None for all-phase solves.
    params : (n, n_params) float64 or None — per-point fit-parameter vectors.
    """

    def __init__(self, T, P, N, X, x_cond_mask, phase_restrict=None, params=None):
        self.T = np.ascontiguousarray(T, dtype=np.float64)
        n = self.T.shape[0]
        self.P = np.ascontiguousarray(np.broadcast_to(P, (n,)), dtype=np.float64)
        self.N = np.ascontiguousarray(np.broadcast_to(N, (n,)), dtype=np.float64)
        self.X = np.ascontiguousarray(X, dtype=np.float64)
        self.x_cond_mask = np.ascontiguousarray(x_cond_mask, dtype=bool)
        self.phase_restrict = (np.asarray(phase_restrict, dtype=object)
                               if phase_restrict is not None
                               else np.full(n, None, dtype=object))
        self.params = (np.ascontiguousarray(params, dtype=np.float64)
                       if params is not None else None)
        assert self.X.shape[0] == n and self.x_cond_mask.shape == self.X.shape

    def __len__(self):
        return self.T.shape[0]


def point_hull(points, grid, nonvacant_elements, grid_T=None, grid_P=None):
    """Per-point lower convex hull (twin of pycalphad lower_convex_hull).

    Parameters
    ----------
    points : PointList
    grid : the (to_xarray=False) calculate() result covering every unique
        (P, T) of the point list, with fake points. Expected statevar-major
        layout: grid.GM shape (..., n_statevar_combos, n_samples) — pass the
        per-combo VIEWS via grid_slices instead if already extracted.
    nonvacant_elements : list of component names (grid X column order)

    Returns
    -------
    dict with per-point starting data:
      GM (n,), MU (n, ncomp), NP (n, ncomp+1), points_idx (n, ncomp+1),
      Phase (n, ncomp+1) object, X (n, ncomp+1, ncomp), Y (n, ncomp+1, maxdof)
    """
    ncomp = len(nonvacant_elements)
    n = len(points)

    grid_GM = np.asarray(grid.GM)          # (n_sv_combos, M)
    grid_X = np.asarray(grid.X)            # (n_sv_combos, M, ncomp)
    grid_Y = np.asarray(grid.Y)            # (n_sv_combos, M, maxdof)
    grid_Phase = np.asarray(grid.Phase)    # (n_sv_combos, M)
    if grid_GM.ndim > 2:
        # collapse leading statevar axes (N, P, T, ...) to one combo axis
        grid_GM = grid_GM.reshape(-1, grid_GM.shape[-1])
        grid_X = grid_X.reshape(-1, *grid_X.shape[-2:])
        grid_Y = grid_Y.reshape(-1, *grid_Y.shape[-2:])
        grid_Phase = grid_Phase.reshape(-1, grid_Phase.shape[-1])
    n_combos = grid_GM.shape[0]

    combo_of_point = _combo_of_point(points, n_combos, grid_T, grid_P)

    # phase-restriction masks are shared per (combo, phase)
    mask_cache = {}

    def _mask_for(combo, phase):
        key = (combo, phase)
        m = mask_cache.get(key)
        if m is None:
            ph = grid_Phase[combo]
            m = np.flatnonzero((ph == phase) | (ph == '_FAKE_'))
            mask_cache[key] = m
        return m

    out = {
        'GM': np.empty(n, dtype=np.float64),
        'MU': np.zeros((n, ncomp), dtype=np.float64),
        'NP': np.empty((n, ncomp + 1), dtype=np.float64),
        'points_idx': np.empty((n, ncomp + 1), dtype=np.int32),
        'Phase': np.empty((n, ncomp + 1), dtype=object),
        'X': np.empty((n, ncomp + 1, ncomp), dtype=np.float64),
        'Y': np.empty((n, ncomp + 1, grid_Y.shape[-1]), dtype=np.float64),
    }

    result_fractions = np.empty(ncomp + 1, dtype=np.float64)
    result_simplex = np.empty(ncomp + 1, dtype=np.intc)

    for i in range(n):
        combo = int(combo_of_point[i])
        restrict = points.phase_restrict[i]
        if restrict is None:
            comps_view = grid_X[combo]
            ener_view = grid_GM[combo]
            back_map = None
        else:
            back_map = _mask_for(combo, restrict)
            comps_view = np.ascontiguousarray(grid_X[combo][back_map])
            ener_view = np.ascontiguousarray(grid_GM[combo][back_map])

        # fixed linear-combination rows: one per prescribed X + the N row.
        # Mirrors lower_convex_hull's MoleFraction / SystemMolesType handling.
        xmask = points.x_cond_mask[i]
        coefs = []
        rhs = []
        for c in np.flatnonzero(xmask):
            row = np.zeros(ncomp)
            row[c] = 1.0
            coefs.append(row)
            rhs.append(points.X[i, c])
        coefs.append(np.ones(ncomp))          # N condition
        rhs.append(points.N[i])
        coefs = np.atleast_2d(np.asarray(coefs, dtype=np.float64))
        rhs = np.asarray(rhs, dtype=np.float64)

        mu = out['MU'][i]
        mu[:] = 0.0
        gm = hyperplane(comps_view, ener_view, mu,
                        np.array([], dtype=np.uintp), coefs, rhs,
                        result_fractions, result_simplex)

        idx = result_simplex.astype(np.int32)
        if back_map is not None:
            idx = back_map[idx].astype(np.int32)
        out['GM'][i] = gm
        out['NP'][i] = result_fractions
        out['points_idx'][i] = idx
        # Reference semantics (lower_convex_hull.py:219-222): only the first
        # num_comps vertex slots are copied from the grid; trailing slots stay
        # empty/NaN and are therefore never valid starting phases downstream.
        out['Phase'][i, :] = ''
        out['X'][i, :] = np.nan
        out['Y'][i, :] = np.nan
        out['Phase'][i, :ncomp] = grid_Phase[combo].take(idx[:ncomp])
        out['X'][i, :ncomp] = grid_X[combo].take(idx[:ncomp], axis=0)
        out['Y'][i, :ncomp] = grid_Y[combo].take(idx[:ncomp], axis=0)

        # Fake-point dissolution incl. the GM recompute over non-fake
        # vertices (lower_convex_hull.py:224-239).
        row = out['Phase'][i]
        if '_FAKE_' in row:
            new_energy = 0.0
            molesum = 0.0
            for j in range(row.shape[0]):
                if row[j] == '_FAKE_':
                    row[j] = ''
                    out['X'][i, j] = np.nan
                    out['Y'][i, j] = np.nan
                    out['NP'][i, j] = np.nan
                else:
                    new_energy += out['NP'][i, j] * grid_GM[combo][idx[j]]
                    molesum += out['NP'][i, j]
            if molesum != 0:
                out['GM'][i] = new_energy / molesum
    return out


def build_condition_args(points, state_variables, components, max_statevars,
                         max_components, nonvacant_elements):
    """(n, MSV+MC) condition-args rows; layout per gpu_equilibrium.py:311."""
    n = len(points)
    rows = np.zeros((n, max_statevars + max_components), dtype=np.float64)
    sv_map = {'N': points.N, 'P': points.P, 'T': points.T}
    for sv_idx, sv in enumerate([str(s) for s in state_variables][:max_statevars]):
        if sv in sv_map:
            rows[:, sv_idx] = sv_map[sv]
    # X column per wks component (VA and non-conditions stay 0, matching
    # _condition_column which zero-fills unknown conditions)
    for comp_idx, comp in enumerate([str(c) for c in components][:max_components]):
        if comp in nonvacant_elements:
            el = nonvacant_elements.index(comp)
            col = np.where(points.x_cond_mask[:, el], points.X[:, el], 0.0)
            rows[:, max_statevars + comp_idx] = col
    return rows


def build_spec_rows(points, spec_row0, hull, dynamic_sizes, nonvacant_elements,
                    core_len=None):
    """Tile spec row 0 and overwrite the per-point field groups.

    spec_row0 : one padded flat spec row built by the existing machinery for
        a representative point (create_flat_system_specification +
        apply_safe_padding). Constraint enumeration in row 0 must match
        x_cond_mask column order (sorted nonvacant elements).
    """
    n = len(points)
    MC = int(dynamic_sizes['MAX_COMPONENTS'])
    MAX_PARAMS = int(dynamic_sizes.get('MAX_PARAMS', 0))
    rows = np.tile(np.asarray(spec_row0, dtype=np.float64), (n, 1))

    off_mu = 3
    off_rhs = 3 + MC + MC * MC

    # starting chemical potentials from the point hull
    ncomp = len(nonvacant_elements)
    rows[:, off_mu:off_mu + min(ncomp, MC)] = hull['MU'][:, :min(ncomp, MC)]

    # prescribed mole-fraction rhs, one constraint slot per prescribed X in
    # nonvacant order (matches _populate_system_specification's enumeration
    # for standard X conditions)
    cmask = points.x_cond_mask
    n_constraints = int(cmask[0].sum())
    if not np.all(cmask.sum(axis=1) == n_constraints):
        raise ValueError("all points must prescribe the same NUMBER of X conditions")
    for i in range(n):
        slot = 0
        for c in np.flatnonzero(cmask[i]):
            rows[i, off_rhs + slot] = points.X[i, c]
            slot += 1

    # NOTE: if points prescribe DIFFERENT component sets, the coefficient
    # matrix must also be per-point. For same-component batches (the ZPF
    # binary case groups by X component) row 0's coefficients are correct;
    # mixed-component batches overwrite coefficients too:
    off_coef = 3 + MC
    base_coef = rows[0, off_coef:off_coef + MC * MC].copy()
    for i in range(n):
        coef = np.zeros((MC, MC))
        slot = 0
        for c in np.flatnonzero(cmask[i]):
            coef[slot, c] = 1.0
            slot += 1
        rows[i, off_coef:off_coef + MC * MC] = coef.reshape(-1)
    del base_coef

    # per-point fit parameters live at the tail of the CORE section
    if MAX_PARAMS > 0 and points.params is not None:
        if core_len is None:
            core_len = _spec_core_len(dynamic_sizes)
        p_off = core_len - (MAX_PARAMS + 1)
        k = min(points.params.shape[1], MAX_PARAMS)
        rows[:, p_off:p_off + k] = points.params[:, :k]
        rows[:, core_len - 1] = points.params.shape[1]
    return rows


def _spec_core_len(dynamic_sizes):
    """Length in doubles of the spec CORE section (gpu_systemspec_flat.py)."""
    MC = int(dynamic_sizes['MAX_COMPONENTS'])
    MSV = int(dynamic_sizes['MAX_STATEVARS'])
    MP = int(dynamic_sizes['MAX_PHASES'])
    MFIX = int(dynamic_sizes['MAX_FIXED_MOLE_FRACTION_CONDITIONS'])
    MPAR = int(dynamic_sizes.get('MAX_PARAMS', 0))
    return (3 + MC + MFIX * MC + MFIX + 2 + (MC + 1) + (MSV + 1) + (MC + 1)
            + (MSV + 1) + (MP + 1) + 1 + 1 + (MPAR + 1))


def build_initial_phase_data(points, hull, py_phase_name_to_unique_idx_map,
                             dynamic_sizes):
    """Vectorized ipd dict arrays from the point hull (template:
    gpu_equilibrium.py fast path, lines 486-529)."""
    n = len(points)
    MP = int(dynamic_sizes['MAX_PHASES'])
    MDOF = int(dynamic_sizes['MAX_DOF_PER_PHASE'])
    MC = int(dynamic_sizes['MAX_COMPONENTS'])

    phase_arr = np.asarray([[p if p is not None else '' for p in row]
                            for row in hull['Phase']], dtype=object)
    np_arr = hull['NP']
    y_arr = hull['Y']
    x_arr = hull['X']
    mu_arr = hull['MU']

    uniq, inv = np.unique(phase_arr.astype(str), return_inverse=True)
    lut = np.array([py_phase_name_to_unique_idx_map.get(name, -1)
                    if name not in ('', '_FAKE_') else -1 for name in uniq],
                   dtype=np.int64)
    model_idx_arr = lut[inv].reshape(phase_arr.shape)

    with np.errstate(invalid='ignore'):
        valid = (model_idx_arr >= 0) & (np.nan_to_num(np_arr) > 1e-10)
    order = np.argsort(~valid, axis=1, kind='stable')
    valid_sorted = np.take_along_axis(valid, order, axis=1)
    counts = valid_sorted.sum(axis=1)

    n_slots = min(MP, phase_arr.shape[1])
    slot_order = order[:, :n_slots]
    slot_valid = valid_sorted[:, :n_slots]
    rows_s = np.broadcast_to(np.arange(n)[:, None], (n, n_slots))

    ipd = {
        'phase_indices': np.full((n, MP), -1, dtype=np.int32),
        'phase_amounts': np.zeros((n, MP), dtype=np.float64),
        'site_fractions': np.zeros((n, MP, MDOF), dtype=np.float64),
        'compositions': np.zeros((n, MP, MC), dtype=np.float64),
        'chemical_potentials': np.zeros((n, MC), dtype=np.float64),
        'num_phases': np.minimum(counts, MP).astype(np.int32),
    }
    mu_cols = min(mu_arr.shape[1], MC)
    ipd['chemical_potentials'][:, :mu_cols] = mu_arr[:, :mu_cols]

    pid = model_idx_arr[rows_s, slot_order]
    ipd['phase_indices'][:, :n_slots] = np.where(slot_valid, pid, 0)

    amounts = np.maximum(np.nan_to_num(np_arr[rows_s, slot_order]), MIN_PHASE_FRACTION)
    ipd['phase_amounts'][:, :n_slots] = np.where(slot_valid, amounts, 0.0)

    # NaN paddings (Y columns beyond a phase's dof) are copied through
    # unmodified — the reference pipeline does the same and the kernel bounds
    # every loop by phase_rec->phase_dof.
    y_cols = min(y_arr.shape[2], MDOF)
    y_g = y_arr[rows_s, slot_order][:, :, :y_cols]
    ipd['site_fractions'][:, :n_slots, :y_cols] = np.where(slot_valid[:, :, None], y_g, 0.0)

    x_cols = min(x_arr.shape[2], MC)
    x_g = x_arr[rows_s, slot_order][:, :, :x_cols]
    ipd['compositions'][:, :n_slots, :x_cols] = np.where(slot_valid[:, :, None], x_g, 0.0)
    return ipd


# --------------------------------------------------------------------------
# Solver acquisition + launch (C++ backend)
# --------------------------------------------------------------------------

_solver_cache = {}


class PointBatchSolver:
    """Compile/acquire the batch solver for one system and launch point lists.

    Follows gpu_calculate._build_module's standalone acquisition pattern: the
    generated full kernel source (disk-cached by model expressions + statevar
    layout) compiled via cpu_backend.build_cpu_library exposes BOTH the grid
    evaluator and pycgpu_cpu_run_all, so no part of the calculate_equilibrium_gpu
    monolith is needed.
    """

    def __init__(self, components, phases, models, phase_record_factory,
                 robust=True, verbose=False):
        self.verbose = verbose
        shim = SimpleNamespace(components=list(components), phases=list(phases),
                               models=models, phase_record_factory=phase_record_factory,
                               conditions={}, verbose=verbose)
        self.shim = shim
        from pycalphad.gpu.gpu_codegen import (compute_dynamic_kernel_sizes,
                                               _unique_models_for_gpu,
                                               _generate_c_code_for_phase_models,
                                               _generate_full_gpu_source)
        from pycalphad.gpu.gpu_equilibrium import _kernel_cache_dir
        self.dynamic_sizes = compute_dynamic_kernel_sizes(shim)
        define_flags = [f'-D{k}={v}' for k, v in self.dynamic_sizes.items()]
        if robust:
            define_flags.append('-DPYCGPU_ROBUST_REMOVAL')

        gpu_dir = os.path.dirname(os.path.abspath(__file__))
        hasher = hashlib.md5()
        for hdr in ("svd.c", "phase_rec.h", "comp_set.h", "lu_solver.h",
                    "minimizer.h", "eqsolver.h", "gpu_codegen.py"):
            with open(os.path.join(gpu_dir, hdr), "rb") as f:
                hasher.update(f.read())
        model_hasher = hashlib.md5()
        for ph in sorted(shim.phases):
            model_hasher.update(ph.encode())
            model_hasher.update(str(shim.models[ph].GM).encode())
        key_input = "|".join([
            "pointsolve", ",".join(sorted(shim.phases)),
            ",".join(sorted(c.name for c in shim.components)),
            "statevars:" + ",".join(str(sv) for sv in shim.phase_record_factory.state_variables),
            "models:" + model_hasher.hexdigest(),
            str(sorted(self.dynamic_sizes.items())), hasher.hexdigest(),
        ])
        cache_key = hashlib.md5(key_input.encode()).hexdigest()
        cache_dir = _kernel_cache_dir()
        cache_file = cache_dir / f"{cache_key}.cu"
        if cache_file.exists():
            full_source = cache_file.read_text()
        else:
            model_funcs_c, pr_init_calls_c, unique_models, _ = \
                _generate_c_code_for_phase_models(shim, include_hess=True, validate=False)
            full_source = _generate_full_gpu_source(shim, model_funcs_c, pr_init_calls_c,
                                                    len(unique_models))
            cache_file.write_text(full_source)

        from pycalphad.gpu.cpu_backend import build_cpu_library
        self.lib = build_cpu_library(full_source, define_flags,
                                     cache_dir=str(cache_dir), verbose=verbose)
        _, self.name_to_idx = _unique_models_for_gpu(shim, validate=False)

        ds = self.dynamic_sizes
        self.MC = int(ds['MAX_COMPONENTS'])
        self.MP = int(ds['MAX_PHASES'])
        self.MSV = int(ds['MAX_STATEVARS'])
        self.MDOF = int(ds['MAX_DOF_PER_PHASE'])
        self.results_per_condition = (7 + self.MC + self.MP
                                      + self.MP * self.MDOF
                                      + self.MP * self.MC + self.MP)

    # ---------------------------------------------------------------- spec0
    def build_spec_row0(self, point0_conds, x_component):
        """Padded flat spec row for a representative point.

        point0_conds: dict {'N':1.0,'P':...,'T':...}; x_component: element name
        of the prescribed mole fraction (value taken from the point later —
        rhs/coefs/MU/params are overwritten per point by build_spec_rows).
        """
        import pycalphad.variables as v
        from pycalphad.gpu.gpu_equilibrium import _populate_system_specification
        from pycalphad.gpu.gpu_systemspec_flat import (create_flat_system_specification,
                                                       apply_safe_padding)
        conds = {v.N: point0_conds.get('N', 1.0), v.P: point0_conds['P'],
                 v.T: point0_conds['T'], v.X(x_component): point0_conds['X0']}
        shim = SimpleNamespace(components=self.shim.components,
                               phases=self.shim.phases,
                               models=self.shim.models,
                               phase_record_factory=self.shim.phase_record_factory,
                               conditions=conds, verbose=False)
        scalars = np.zeros(50, dtype=np.float64)
        MC, MSV, MP = self.MC, self.MSV, self.MP
        MFIX = int(self.dynamic_sizes['MAX_FIXED_MOLE_FRACTION_CONDITIONS'])
        arrays = {
            'initial_chemical_potentials': np.zeros(MC, dtype=np.float64),
            'prescribed_mole_fraction_coefficients': np.zeros((MFIX, MC), dtype=np.float64),
            'prescribed_mole_fraction_rhs': np.zeros(MFIX, dtype=np.float64),
            'free_chemical_potential_indices': np.full(MC, -1, dtype=np.int32),
            'free_statevar_indices': np.full(MSV, -1, dtype=np.int32),
            'fixed_chemical_potential_indices': np.full(MC, -1, dtype=np.int32),
            'fixed_statevar_indices': np.full(MSV, -1, dtype=np.int32),
            'fixed_stable_compset_indices': np.full(MP, -1, dtype=np.int32),
        }
        _populate_system_specification(scalars, arrays, shim, self.dynamic_sizes, None)
        row = create_flat_system_specification(scalars, arrays, self.dynamic_sizes)
        return apply_safe_padding(row, verbose=False)

    # --------------------------------------------------------------- launch
    def solve(self, points, hull, grid, spec_row0, state_variables,
              nonvacant_elements, restrict_grid_views=None,
              max_solver_iterations=1000, grid_T=None, grid_P=None):
        """Launch one batch. Returns dict with per-point flat results.

        grid: calculate() result (to_xarray=False) covering the points'
        statevar combos. restrict_grid_views: optional pre-filtered grid shim
        replacing `grid` for block packing (single-phase batches).
        """
        from pycalphad.gpu.gpu_equilibrium import (
            _create_initial_phase_data_struct_array,
            _prepare_grid_data_for_gpu_from_calculate_result)
        from pycalphad.gpu.cpu_backend import run_cpu_backend
        ds = self.dynamic_sizes
        n = len(points)
        MC, MP, MSV, MDOF = self.MC, self.MP, self.MSV, self.MDOF

        cond_args = build_condition_args(points, state_variables,
                                         [str(c) for c in self.shim.components],
                                         MSV, MC, nonvacant_elements)
        specs = build_spec_rows(points, spec_row0, hull, ds, nonvacant_elements)
        ipd_arrays = build_initial_phase_data(points, hull, self.name_to_idx, ds)
        ipd_struct = _create_initial_phase_data_struct_array(ipd_arrays, n, ds, False)
        ipd_stride = ipd_struct.shape[1]

        grid_src = restrict_grid_views if restrict_grid_views is not None else grid
        grid_blocks, block_shape = _prepare_grid_data_for_gpu_from_calculate_result(
            grid_src, self.name_to_idx, MP, MDOF, MC, self.verbose)
        # map each point to its statevar-combo block (same C-order as point_hull)
        n_blocks = int(np.prod(block_shape))
        gbi = _combo_of_point(points, n_blocks, grid_T, grid_P).astype(np.int32)

        # work arrays (23 slots; layout mirrors gpu_equilibrium.py:2566+)
        MFIX = int(ds['MAX_FIXED_MOLE_FRACTION_CONDITIONS'])
        MIC = int(ds['MAX_INTERNAL_CONSTRAINTS'])
        SVD = MP + MFIX + MC + MSV + 2
        PMD = MDOF + MIC
        DOFS = MSV + MDOF
        EQM = int(ds['MAX_EQ_MATRIX_SIZE'])
        EQR = int(ds['MAX_EQ_MATRIX_ROWS'])
        EQS = int(ds['MAX_EQ_SOLN_LEN'])
        SSS = int(ds['SYSTEM_STATE_SIZE'])
        tpb = 64
        nt = ((n + tpb - 1) // tpb) * tpb
        wa_shapes = [SVD * SVD, SVD * SVD, SVD * SVD, SVD, SVD,
                     PMD * PMD, PMD * PMD, PMD, PMD, PMD * PMD,
                     DOFS, DOFS, DOFS * DOFS, MC, MC * DOFS, PMD * PMD,
                     EQM, EQR, EQS, SSS, MP * MC, MP * MC, MP * MC]
        self._work = [np.empty((nt, s), dtype=np.float64) for s in wa_shapes]
        ptr_table = np.array([w.ctypes.data for w in self._work], dtype=np.uint64)

        results = np.zeros(n * self.results_per_condition, dtype=np.float64)
        run_cpu_backend(self.lib,
                        system_spec=np.ascontiguousarray(specs.reshape(-1)),
                        condition_args_doubles=np.ascontiguousarray(cond_args.reshape(-1)),
                        results=results,
                        num_conditions=n,
                        condition_stride=cond_args.shape[1],
                        python_max_statevars=MSV,
                        initial_phase_data=np.ascontiguousarray(ipd_struct.reshape(-1)),
                        initial_phase_data_stride=ipd_stride,
                        system_spec_stride=specs.shape[1],
                        grid_data=grid_blocks,
                        grid_block_indices=gbi,
                        grid_block_stride_bytes=int(grid_blocks.dtype.itemsize),
                        work_arrays_ptr_table=ptr_table,
                        max_solver_iterations=max_solver_iterations,
                        verbose=self.verbose)

        r = results.reshape(n, self.results_per_condition)
        y0 = 6 + MC + MP
        x0 = y0 + MP * MDOF
        pid0 = x0 + MP * MC
        return {
            'GM': r[:, 0].copy(),
            'MU': r[:, 1:1 + MC].copy(),
            'NP': r[:, 1 + MC:1 + MC + MP].copy(),
            'converged': r[:, 1 + MC + MP] > 0.5,
            'num_stable_phases': r[:, 2 + MC + MP].astype(np.int32),
            'Y': r[:, y0:x0].reshape(n, MP, MDOF).copy(),
            'X': r[:, x0:pid0].reshape(n, MP, MC).copy(),
            'phase_ids': r[:, pid0:pid0 + MP].astype(np.int32),
        }
