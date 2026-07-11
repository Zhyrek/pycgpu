"""Point-list batch solving: per-point input-array builders.

Solves an arbitrary LIST of conditions (T, P, N, X-vector per point) in one
kernel launch, with optional per-point phase restriction (single-phase
"isolated" solves) and per-point fit-parameter vectors (walker batching).
See POINT_SOLVER_DESIGN.md for the verified array-format notes.

This module deliberately does NOT modify the reference CPU code: the hull
twin below calls the same Cython ``hyperplane()`` routine the reference
``lower_convex_hull`` uses, over point arrays instead of a cartesian grid.
"""
import numpy as np

from pycalphad.core.hyperplane import hyperplane

MIN_PHASE_FRACTION = 1e-6


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


def point_hull(points, grid, nonvacant_elements, phase_names_of_grid=None):
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

    # map point -> statevar combo index: unique (P, T) in the same C-order
    # the grid was calculated with (P axis outer, T axis inner when both vary)
    uP = np.unique(points.P)
    uT = np.unique(points.T)
    if n_combos == len(uT):
        combo_of_point = np.searchsorted(uT, points.T)
    elif n_combos == len(uP) * len(uT):
        combo_of_point = (np.searchsorted(uP, points.P) * len(uT)
                          + np.searchsorted(uT, points.T))
    else:
        raise ValueError(f"grid has {n_combos} statevar combos; expected "
                         f"{len(uT)} or {len(uP) * len(uT)}")

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
