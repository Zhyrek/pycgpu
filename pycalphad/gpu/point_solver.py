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


def _combo_of_point(points, n_combos, grid_T=None, grid_P=None, combo_idx=None):
    """Map each point to its statevar-combo index in the grid's C-order
    (P axis outer, T axis inner when both vary). grid_T/grid_P override the
    axes when the grid covers MORE combos than the point list touches;
    combo_idx passes fully explicit per-point indices (e.g. walker-major
    stacked grids where the combo axis is walker x T)."""
    if combo_idx is not None:
        ci = np.asarray(combo_idx, dtype=np.int64)
        if ci.shape[0] != len(points) or ci.min() < 0 or ci.max() >= n_combos:
            raise ValueError("combo_idx out of range for grid")
        return ci
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

    def __init__(self, T, P, N, X, x_cond_mask, phase_restrict=None, params=None,
                 fixed_mu_mask=None, fixed_mu_values=None):
        self.T = np.ascontiguousarray(T, dtype=np.float64)
        n = self.T.shape[0]
        # fixed chemical-potential conditions (reference: lower_convex_hull's
        # ChemicalPotential handling — an index into the fixed set plus a
        # preset MU value, NOT a linear-combination row)
        self.fixed_mu_mask = (np.ascontiguousarray(fixed_mu_mask, dtype=bool)
                              if fixed_mu_mask is not None else None)
        self.fixed_mu_values = (np.ascontiguousarray(fixed_mu_values, dtype=np.float64)
                                if fixed_mu_values is not None else None)
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


def point_hull(points, grid, nonvacant_elements, grid_T=None, grid_P=None,
               combo_idx=None):
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

    combo_of_point = _combo_of_point(points, n_combos, grid_T, grid_P, combo_idx)

    # Broadcast (walker-stacked) grids share one X row across all combos
    # (stride 0): materialize a single writable copy for the Cython
    # hyperplane(), which requires writable C-contiguous buffers.
    _x_shared = grid_X.strides[0] == 0
    _x_row0 = np.array(grid_X[0]) if _x_shared else None  # forced writable copy

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
            comps_view = _x_row0 if _x_shared else grid_X[combo]
            ener_view = grid_GM[combo]
            back_map = None
        else:
            back_map = _mask_for(combo, restrict)
            comps_view = np.array((_x_row0 if _x_shared else grid_X[combo])[back_map])
            ener_view = np.ascontiguousarray(grid_GM[combo][back_map])

        # fixed linear-combination rows: the reference (lower_convex_hull)
        # iterates conditions in sorted-key order, so the N (SystemMolesType)
        # row comes FIRST, then the X rows in component order. Row order
        # changes dgesv pivoting for 3+ components — keep it exact.
        xmask = points.x_cond_mask[i]
        coefs = [np.ones(ncomp)]              # N condition first
        rhs = [points.N[i]]
        for c in np.flatnonzero(xmask):
            row = np.zeros(ncomp)
            row[c] = 1.0
            coefs.append(row)
            rhs.append(points.X[i, c])
        coefs = np.atleast_2d(np.asarray(coefs, dtype=np.float64))
        rhs = np.asarray(rhs, dtype=np.float64)

        mu = out['MU'][i]
        mu[:] = 0.0
        if points.fixed_mu_mask is not None and points.fixed_mu_mask[i].any():
            fixed_idx_i = np.flatnonzero(points.fixed_mu_mask[i]).astype(np.uintp)
            mu[fixed_idx_i] = points.fixed_mu_values[i, fixed_idx_i]
        else:
            fixed_idx_i = np.array([], dtype=np.uintp)
        gm = hyperplane(comps_view, ener_view, mu,
                        fixed_idx_i, coefs, rhs,
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
    # vectorized: constraint slot k of point i is its k-th prescribed
    # component (column order), same enumeration as the per-point loops
    rows_i, cols_c = np.nonzero(cmask)
    slot = np.concatenate([np.arange(c) for c in
                           np.bincount(rows_i, minlength=n)]) if rows_i.size else rows_i
    rows[rows_i, off_rhs + slot] = points.X[rows_i, cols_c]

    # per-point coefficient matrices (points may prescribe different
    # component sets): zero the block, set coef[slot, comp] = 1
    off_coef = 3 + MC
    rows[:, off_coef:off_coef + MC * MC] = 0.0
    rows[rows_i, off_coef + slot * MC + cols_c] = 1.0

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
                 robust=True, verbose=False, backend='cpp'):
        self.verbose = verbose
        self.backend = backend
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
        if os.environ.get('PYCGPU_OUTER_ADD', '1') not in ('0', 'off', ''):
            define_flags.append('-DPYCGPU_OUTER_ADD')

        gpu_dir = os.path.dirname(os.path.abspath(__file__))
        hasher = hashlib.md5()
        for hdr in ("phase_rec.h", "comp_set.h", "hyperplane.h",
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

        if backend == 'cpp':
            from pycalphad.gpu.cpu_backend import build_cpu_library
            self.lib = build_cpu_library(full_source, define_flags,
                                         cache_dir=str(cache_dir), verbose=verbose)
        elif backend == 'cuda':
            import cupy as cp
            self._cp = cp
            from pycalphad.gpu.kernel_manager import cuda_raw_module
            self.module = cuda_raw_module(
                full_source, ['-std=c++11', '-O2'] + define_flags,
                verbose=self.verbose)
            from pycalphad.gpu.kernel_manager import ensure_device_stack_limit
            ensure_device_stack_limit(65536, verbose=self.verbose)
            self.module.get_function('init_all_gpu_phase_records')((1,), (1,), ())
            cp.cuda.runtime.deviceSynchronize()
            self._top_kernel = self.module.get_function('top_level_equilibrium_kernel')
        else:
            raise ValueError(f"unknown backend {backend!r} (use 'cpp' or 'cuda')")
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
    def build_spec_row0(self, point0_conds, x_component=None, x_conditions=None):
        """Padded flat spec row for a representative point.

        point0_conds: dict {'N':1.0,'P':...,'T':...}; x_component: element name
        of the prescribed mole fraction (value taken from the point later —
        rhs/coefs/MU/params are overwritten per point by build_spec_rows).
        x_conditions: alternatively, {element: value} for MULTIPLE prescribed
        mole fractions (ternary+ points); the template's constraint COUNT must
        match the batch's per-point count, which build_spec_rows checks is
        uniform.
        """
        import pycalphad.variables as v
        from pycalphad.gpu.gpu_equilibrium import _populate_system_specification
        from pycalphad.gpu.gpu_systemspec_flat import (create_flat_system_specification,
                                                       apply_safe_padding)
        if x_conditions is None:
            x_conditions = {x_component: point0_conds['X0']}
        conds = {v.N: point0_conds.get('N', 1.0), v.P: point0_conds['P'],
                 v.T: point0_conds['T'],
                 **{v.X(el): val for el, val in x_conditions.items()}}
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
              max_solver_iterations=1000, grid_T=None, grid_P=None,
              combo_idx=None, grid_blocks=None):
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

        if grid_blocks is None:
            grid_src = restrict_grid_views if restrict_grid_views is not None else grid
            grid_blocks, block_shape = _prepare_grid_data_for_gpu_from_calculate_result(
                grid_src, self.name_to_idx, MP, MDOF, MC, self.verbose)
            n_blocks = int(np.prod(block_shape))
        else:
            n_blocks = grid_blocks.shape[0]
        # map each point to its statevar-combo block (same C-order as point_hull)
        gbi = _combo_of_point(points, n_blocks, grid_T, grid_P, combo_idx).astype(np.int32)

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
        wa_shapes = [SVD * SVD, SVD * SVD, SVD * SVD, SVD, SVD,
                     PMD * PMD, PMD * PMD, PMD, PMD, PMD * PMD,
                     DOFS, DOFS, DOFS * DOFS, MC, MC * DOFS, PMD * PMD,
                     EQM, EQR, EQS, SSS, MP * MC, MP * MC, MP * MC]
        # Thread-chunked launches bound the dominant work-array memory (same
        # rationale as PYCGPU_CHUNK in the main pipeline); slices of the
        # per-condition buffers are zero-copy views and sequential launches
        # reuse the same work arrays race-free.
        chunk = int(os.environ.get('PYCGPU_POINT_CHUNK', 65536))
        chunk = min(n, chunk)
        nt = ((chunk + tpb - 1) // tpb) * tpb

        spec_flat = np.ascontiguousarray(specs.reshape(-1))
        cond_flat = np.ascontiguousarray(cond_args.reshape(-1))
        ipd_flat = np.ascontiguousarray(ipd_struct.reshape(-1))
        results = np.zeros(n * self.results_per_condition, dtype=np.float64)
        spec_stride = specs.shape[1]
        cond_stride = cond_args.shape[1]
        rpc = self.results_per_condition
        gbs = int(grid_blocks.dtype.itemsize)

        if self.backend == 'cpp':
            self._work = [np.empty((nt, s), dtype=np.float64) for s in wa_shapes]
            ptr_table = np.array([w.ctypes.data for w in self._work], dtype=np.uint64)
            for cs in range(0, n, chunk):
                ce = min(cs + chunk, n)
                run_cpu_backend(self.lib,
                                system_spec=spec_flat[cs * spec_stride:ce * spec_stride],
                                condition_args_doubles=cond_flat[cs * cond_stride:ce * cond_stride],
                                results=results[cs * rpc:ce * rpc],
                                num_conditions=ce - cs,
                                condition_stride=cond_stride,
                                python_max_statevars=MSV,
                                initial_phase_data=ipd_flat[cs * ipd_stride:ce * ipd_stride],
                                initial_phase_data_stride=ipd_stride,
                                system_spec_stride=spec_stride,
                                grid_data=grid_blocks,
                                grid_block_indices=gbi[cs:ce],
                                grid_block_stride_bytes=gbs,
                                work_arrays_ptr_table=ptr_table,
                                max_solver_iterations=max_solver_iterations,
                                verbose=self.verbose)
        else:
            cp = self._cp
            d_spec = cp.asarray(spec_flat)
            d_cond = cp.asarray(cond_flat)
            d_ipd = cp.asarray(ipd_flat)
            d_res = cp.asarray(results)
            d_grid = cp.asarray(np.frombuffer(grid_blocks.tobytes(), dtype=np.uint8))
            d_gbi = cp.asarray(gbi)
            # Work arrays and the pointer table persist across calls (sized to
            # the largest chunk seen) — reallocating ~GBs per launch dominated
            # small-step ensemble wall time.
            wk = getattr(self, '_dev_work', None)
            if wk is None or wk[0] < nt:
                d_work = [cp.empty((nt, s), dtype=cp.float64) for s in wa_shapes]
                d_ptrs = cp.asarray(np.array([w.data.ptr for w in d_work], dtype=np.uint64))
                self._dev_work = wk = (nt, d_work, d_ptrs)
            _, d_work, d_ptrs = wk
            for cs in range(0, n, chunk):
                ce = min(cs + chunk, n)
                cn = ce - cs
                args = (int(d_spec.data.ptr + cs * spec_stride * 8),
                        int(d_cond.data.ptr + cs * cond_stride * 8),
                        int(d_res.data.ptr + cs * rpc * 8),
                        np.int32(cn), np.int32(cond_stride), np.int32(MSV),
                        int(d_ipd.data.ptr + cs * ipd_stride * 8),
                        np.int32(ipd_stride), np.int32(spec_stride),
                        int(d_grid.data.ptr),
                        0, 0, 0, 0, np.int32(0),
                        int(d_ptrs.data.ptr),
                        int(d_gbi.data.ptr + cs * 4),
                        np.int64(gbs),
                        np.int32(max_solver_iterations))
                blocks = (cn + tpb - 1) // tpb
                self._top_kernel((blocks,), (tpb,), args)
            cp.cuda.runtime.deviceSynchronize()
            results = cp.asnumpy(d_res)

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


def device_point_hull(points, solver, X_row, GM_rows, combo_idx, Phase_row,
                      Y_row, nonvacant_elements, light=False):
    """Compiled point hull (hyperplane.h) — same result dict as point_hull.

    Parameters
    ----------
    points : PointList (no fixed chemical potentials; X + N conditions)
    solver : PointBatchSolver (provides the compiled module for its backend)
    X_row : (m, ncomp) sample compositions — identical for every statevar
        combo (and walker); pass the phase-filtered copy for restricted sets.
    GM_rows : (n_combos, m) sample energies per combo.
    combo_idx : (n,) combo index per point.
    Phase_row / Y_row : (m,) names and (m, maxdof) site fractions aligned
        with X_row (for the starting-data extraction).
    """
    import ctypes
    ncomp = len(nonvacant_elements)
    n = len(points)
    m = X_row.shape[0]
    # X_row/GM_rows may be device (cupy) arrays on the CUDA backend — used
    # as-is, no host round trip. Host copies are made only where host code
    # needs them (fake-GM recompute below).
    _is_dev = type(X_row).__module__.startswith('cupy') or \
        type(GM_rows).__module__.startswith('cupy')
    if not _is_dev:
        X_row = np.ascontiguousarray(X_row, dtype=np.float64)
        GM_flat = np.ascontiguousarray(GM_rows, dtype=np.float64).reshape(-1)
    else:
        GM_flat = GM_rows.reshape(-1)

    x_base = np.zeros(n, dtype=np.int64)
    gm_base = (np.asarray(combo_idx, dtype=np.int64) * m)
    m_points = np.full(n, m, dtype=np.int32)
    fixed_idx = np.full((n, ncomp), -1, dtype=np.int32)
    nfixed = np.zeros(n, dtype=np.int32)
    if points.fixed_mu_mask is not None and points.fixed_mu_mask.any():
        fr, fc = np.nonzero(points.fixed_mu_mask)
        slot_f = np.concatenate([np.arange(c) for c in
                                 np.bincount(fr, minlength=n)]) if fr.size else fr
        fixed_idx[fr, slot_f] = fc.astype(np.int32)
        nfixed[:] = np.bincount(fr, minlength=n).astype(np.int32)

    # lincomb rows in the reference's sorted-key order: N row FIRST, then
    # one row per prescribed X (row order changes dgesv pivoting for 3+
    # components); constraint counts are uniform per batch by construction.
    n_x = int(points.x_cond_mask[0].sum())
    max_lc = n_x + 1
    coefs = np.zeros((n, max_lc, ncomp), dtype=np.float64)
    rhs = np.zeros((n, max_lc), dtype=np.float64)
    nlc = np.full(n, max_lc, dtype=np.int32)
    coefs[:, 0, :] = 1.0
    rhs[:, 0] = points.N
    rows_i, cols_c = np.nonzero(points.x_cond_mask)
    slot = np.concatenate([np.arange(c) for c in
                           np.bincount(rows_i, minlength=n)]) if rows_i.size else rows_i
    coefs[rows_i, 1 + slot, cols_c] = 1.0
    rhs[rows_i, 1 + slot] = points.X[rows_i, cols_c]

    mu = np.zeros((n, ncomp), dtype=np.float64)
    if points.fixed_mu_mask is not None and points.fixed_mu_mask.any():
        mu[points.fixed_mu_mask] = points.fixed_mu_values[points.fixed_mu_mask]
    oe = np.zeros(n, dtype=np.float64)
    fr = np.zeros((n, ncomp + 1), dtype=np.float64)
    sx = np.zeros((n, ncomp + 1), dtype=np.int32)

    if solver.backend == 'cpp':
        def p(a):
            return a.ctypes.data
        solver.lib.pycgpu_cpu_point_hull(
            p(X_row), p(GM_flat), p(x_base), p(gm_base), p(m_points),
            ctypes.c_int(ncomp), p(fixed_idx), p(nfixed),
            p(coefs), p(rhs), p(nlc), ctypes.c_int(max_lc),
            p(mu), p(oe), p(fr), p(sx), ctypes.c_int(n))
    else:
        cp = solver._cp
        d = {k: cp.asarray(v) for k, v in
             dict(X=X_row, GM=GM_flat, xb=x_base, gb=gm_base, mp=m_points,
                  fi=fixed_idx, nf=nfixed, co=coefs, rh=rhs, nl=nlc,
                  mu=mu, oe=oe, fr=fr, sx=sx).items()}
        if _is_dev:
            d['X'] = X_row
            d['GM'] = GM_flat
        tpb = 64
        kern = solver.module.get_function('point_hull_kernel')
        kern(((n + tpb - 1) // tpb,), (tpb,),
             (d['X'], d['GM'], d['xb'], d['gb'], d['mp'], np.int32(ncomp),
              d['fi'], d['nf'], d['co'], d['rh'], d['nl'], np.int32(max_lc),
              d['mu'], d['oe'], d['fr'], d['sx'], np.int32(n)))
        cp.cuda.runtime.deviceSynchronize()
        _dev_handles = {'sx': d['sx'], 'fr': d['fr'], 'mu': d['mu']}
        mu, oe, fr, sx = (cp.asnumpy(d['mu']), cp.asnumpy(d['oe']),
                          cp.asnumpy(d['fr']), cp.asnumpy(d['sx']))

    if _is_dev:
        import cupy as _cp
        X_row = _cp.asnumpy(X_row)
        GM_flat = _cp.asnumpy(GM_flat) if False else None  # only fake rows needed
    # ---- vectorized reference post-processing (matches point_hull) ----
    # Only the first ncomp vertex slots are meaningful; trailing stays
    # ''/NaN. Fake vertices dissolve with the non-fake GM recompute.
    idx = sx[:, :ncomp]
    if light:
        # Device-pipeline mode: the per-vertex NP/X/Y/Phase host arrays are
        # unused downstream (the starting rows are built on device from the
        # raw outputs), so skip the large gathers. GM keeps the fake-vertex
        # recompute (numeric fake mask instead of the object-array gather);
        # MU is already host-resident.
        fake_row = (np.asarray(Phase_row) == '_FAKE_')
        fake = fake_row[idx]
        has_fake = fake.any(axis=1)
        if has_fake.any():
            if GM_flat is None:
                import cupy as _cp
                gm_at = _cp.asnumpy(GM_rows.reshape(-1)[_cp.asarray(gm_base[:, None] + idx)])
            else:
                gm_at = GM_flat[(gm_base[:, None] + idx)]
            w = np.where(fake, 0.0, fr[:, :ncomp])
            molesum = w.sum(axis=1)
            new_e = (w * gm_at).sum(axis=1)
            recompute = has_fake & (molesum != 0)
            oe = np.where(recompute, np.divide(new_e, molesum,
                                               out=np.zeros_like(new_e),
                                               where=molesum != 0), oe)
        out = {'GM': oe, 'MU': mu, 'points_idx': sx.astype(np.int32),
               'NP': np.zeros((n, ncomp + 1)),
               'X': np.zeros((n, ncomp + 1, ncomp)),
               'Y': np.zeros((n, ncomp + 1, Y_row.shape[1])),
               'Phase': np.zeros((n, ncomp + 1), dtype='U1')}
        if solver.backend != 'cpp':
            out['_dev'] = _dev_handles
        return out
    phase = Phase_row[idx]                        # (n, ncomp) object/str
    fake = phase == '_FAKE_'
    NP = np.full((n, ncomp + 1), np.nan)
    NP[:, :ncomp] = fr[:, :ncomp]
    Xv = np.full((n, ncomp + 1, ncomp), np.nan)
    Yv = np.full((n, ncomp + 1, Y_row.shape[1]), np.nan)
    Xv[:, :ncomp] = X_row[idx]
    Yv[:, :ncomp] = Y_row[idx]
    Phase = np.full((n, ncomp + 1), '', dtype=object)
    Phase[:, :ncomp] = phase
    has_fake = fake.any(axis=1)
    if has_fake.any():
        if GM_flat is None:  # device GM: gather just the (n, ncomp) values
            import cupy as _cp
            gm_at = _cp.asnumpy(GM_rows.reshape(-1)[_cp.asarray(gm_base[:, None] + idx)])
        else:
            gm_at = GM_flat[(gm_base[:, None] + idx)]     # (n, ncomp)
        w = np.where(fake, 0.0, fr[:, :ncomp])
        molesum = w.sum(axis=1)
        new_e = (w * gm_at).sum(axis=1)
        recompute = has_fake & (molesum != 0)
        oe = np.where(recompute, np.divide(new_e, molesum,
                                           out=np.zeros_like(new_e),
                                           where=molesum != 0), oe)
        Phase[:, :ncomp][fake] = ''
        NP[:, :ncomp][fake] = np.nan
        Xv[:, :ncomp][fake] = np.nan
        Yv[:, :ncomp][fake] = np.nan
    out = {'GM': oe, 'MU': mu, 'NP': NP, 'points_idx': sx.astype(np.int32),
           'Phase': Phase, 'X': Xv, 'Y': Yv}
    if solver.backend != 'cpp':
        # Device-resident pipeline: hand the raw device outputs to the
        # caller so the solve's starting rows can be built ON DEVICE
        # (gpu_equilibrium builds the packed ipd rows with a small kernel
        # instead of gathering/packing/uploading them host-side).
        out['_dev'] = _dev_handles
    return out


def get_point_solver(components, phases, models, phase_record_factory,
                     robust=True, backend='cpp', verbose=False):
    """In-process cached PointBatchSolver (compiled artifacts are disk/cupy
    cached; this avoids re-running codegen bookkeeping per equilibrium call)."""
    mh = hashlib.md5()
    for ph in sorted(phases):
        mh.update(ph.encode())
        mh.update(str(models[ph].GM).encode())
    key = (backend, robust, tuple(sorted(phases)),
           tuple(sorted(getattr(c, 'name', str(c)) for c in components)),
           tuple(str(sv) for sv in phase_record_factory.state_variables),
           mh.hexdigest())
    entry = _solver_cache.get(key)
    if entry is None:
        entry = PointBatchSolver(components, phases, models, phase_record_factory,
                                 robust=robust, verbose=verbose, backend=backend)
        _solver_cache[key] = entry
    return entry


def device_starting_point(unitless_conds, state_variables, phase_record_factory,
                          grid, solver, verbose=False):
    """Compiled-hull replacement for starting_point() on the accelerated
    equilibrium path (PYCGPU_DEVICE_HULL=1).

    Valid for the capability-gated condition set (N / P / T / X only,
    fully determined). Returns a LightDataset with the same variables,
    coordinate order, and array semantics as pycalphad's starting_point —
    the hull values themselves come from hyperplane.h, which is verified
    bit-identical to the Cython hyperplane().
    """
    from collections import OrderedDict
    from pycalphad.core.light_dataset import LightDataset
    from pycalphad import __version__ as pycalphad_version
    import pycalphad.variables as v

    active_phases = sorted(phase_record_factory.keys()) if hasattr(phase_record_factory, 'keys') else None
    nonvacant = None
    for ph in solver.shim.phases:
        nonvacant = phase_record_factory[ph].nonvacant_elements
        break
    nonvacant = list(nonvacant)
    ncomp = len(nonvacant)

    conds_items = list(unitless_conds.items())
    # Reference validation (starting_point:101-110): every non-statevar,
    # non-phase-local condition consumes one degree of freedom; under/
    # overdetermined problems must raise the same error here.
    number_dof = ncomp - 1
    for k, _ in conds_items:
        if not (hasattr(k, 'species') or type(k).__name__ == 'LinearCombination'):
            continue
        if hasattr(k, 'species') and getattr(k, 'phase_name', None) is not None:
            continue
        number_dof -= 1
    if number_dof != 0:
        raise ValueError('Number of degrees of freedom is not zero')

    axes = [np.atleast_1d(np.asarray(val, dtype=np.float64)) for _, val in conds_items]
    shape = tuple(len(a) for a in axes)
    n = int(np.prod(shape))
    mesh = np.meshgrid(*axes, indexing='ij') if n > 1 or len(axes) > 1 else \
        [a.reshape(1) for a in axes]
    cols = {str(k): m.reshape(-1) for (k, _), m in zip(conds_items, mesh)}

    T = cols.get('T')
    P = cols.get('P', np.full(n, 101325.0))
    N = cols.get('N', np.ones(n))
    X = np.zeros((n, ncomp))
    mask = np.zeros((n, ncomp), dtype=bool)
    mu_mask = np.zeros((n, ncomp), dtype=bool)
    mu_vals = np.zeros((n, ncomp))
    for key, colv in cols.items():
        if key.startswith('X_'):
            ci = nonvacant.index(key[2:])
            X[:, ci] = colv
            mask[:, ci] = True
        elif key.startswith('MU_'):
            ci = nonvacant.index(key[3:])
            mu_vals[:, ci] = colv
            mu_mask[:, ci] = True
    # dependent component by mass balance (single unknown under the gate)
    free = ~mask[0]
    if free.sum() == 1:
        X[:, free] = (1.0 - X[:, mask[0]].sum(axis=1))[:, None]

    pts = PointList(T=T, P=P, N=N, X=X, x_cond_mask=mask,
                    fixed_mu_mask=mu_mask if mu_mask.any() else None,
                    fixed_mu_values=mu_vals if mu_mask.any() else None)

    # statevar-combo index per point (C-order over the statevar axes, which
    # lead the conditions ordering under the gate: N, P, T sort before X_*)
    sv_names = [str(k) for k, _ in conds_items if str(k) in ('N', 'P', 'T')]
    sv_axes = [np.unique(np.atleast_1d(np.asarray(dict(cols)[s]))) for s in sv_names]
    sv_sizes = [len(a) for a in sv_axes]
    combo = np.zeros(n, dtype=np.int64)
    stride = 1
    for name, ax, size in zip(reversed(sv_names), reversed(sv_axes), reversed(sv_sizes)):
        combo += np.searchsorted(ax, cols[name]).astype(np.int64) * stride
        stride *= size

    gm = np.asarray(grid.GM)
    M = gm.shape[-1]
    GM_rows = np.ascontiguousarray(gm.reshape(-1, M))
    gx = np.asarray(grid.X).reshape((-1,) + np.asarray(grid.X).shape[-2:])
    gy = np.asarray(grid.Y).reshape((-1,) + np.asarray(grid.Y).shape[-2:])
    gp = np.asarray(grid.Phase).reshape(-1, M)

    _gx0 = np.ascontiguousarray(gx[0])
    _gy0 = np.ascontiguousarray(gy[0])
    _light = (solver.backend != 'cpp'
              and os.environ.get('PYCGPU_DEVICE_PIPE', '1') != '0')
    hull = device_point_hull(pts, solver, _gx0, GM_rows,
                             combo, gp[0], _gy0, nonvacant, light=_light)

    # ---- LightDataset with starting_point's exact structure ----
    max_phase_name_len = max(max(len(x) for x in solver.shim.phases), 6)
    maximum_internal_dof = gy.shape[-1]
    coord_dict = OrderedDict((str(k), np.atleast_1d(np.asarray(val)))
                             for k, val in conds_items)
    coord_dict['vertex'] = np.arange(ncomp + 1)
    coord_dict['component'] = nonvacant
    conds_as_strings = [str(k) for k, _ in conds_items]

    ds_vars = {
        'NP': (conds_as_strings + ['vertex'],
               hull['NP'].reshape(shape + (ncomp + 1,))),
        'GM': (conds_as_strings, hull['GM'].reshape(shape)),
        'MU': (conds_as_strings + ['component'],
               hull['MU'].reshape(shape + (ncomp,))),
        'X': (conds_as_strings + ['vertex', 'component'],
              hull['X'].reshape(shape + (ncomp + 1, ncomp))),
        'Y': (conds_as_strings + ['vertex', 'internal_dof'],
              hull['Y'].reshape(shape + (ncomp + 1, maximum_internal_dof))),
        'Phase': (conds_as_strings + ['vertex'],
                  hull['Phase'].astype('U%s' % max_phase_name_len)
                  .reshape(shape + (ncomp + 1,))),
    }
    ds = LightDataset(ds_vars, coords=coord_dict,
                       attrs={'engine': 'pycalphad %s' % pycalphad_version})
    if '_dev' in hull:
        # Device-resident pipeline handoff (cuda backend): everything the
        # ipd-builder kernel needs. X/Y grid rows and the Phase-name row are
        # per-system (small); the per-condition device arrays avoid the
        # host gather/pack/upload of the starting rows entirely.
        ds._hull_dev = dict(hull['_dev'],
                            X_row=_gx0, Y_row=_gy0, Phase_row=gp[0],
                            ncomp=ncomp)
    return ds
