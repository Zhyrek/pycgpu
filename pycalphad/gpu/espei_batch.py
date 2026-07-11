"""Batched ZPF driving-force evaluation for ESPEI on the accelerated backends.

Replaces ESPEI's per-vertex serial Workspace solves (``estimate_hyperplane`` +
``driving_force_to_hyperplane`` in ``espei.error_functions.zpf_error``) with
point-list batch launches (pycalphad.gpu.point_solver):

- all hyperplane vertices with full composition conditions ride ONE all-phase
  launch;
- driving-force vertices ride ONE single-phase launch per distinct phase
  (phase-filtered grid blocks; no kernel changes);
- vertices the batch path cannot serve exactly (missing composition
  conditions -> sampling estimate, disordered configurations, near-pure-edge
  compositions) fall back to ESPEI's own serial code paths.

Driving-force/weight assembly reproduces ``calculate_zpf_driving_forces``
semantics exactly (vertex ordering, NaN-hyperplane -> zero driving force,
underdetermined-vertex exclusion from the hyperplane average).

Per ``driving_forces(parameters)`` call the parameter-dependent work is: one
``calculate()`` grid over the unique temperatures, the per-point hulls, and
the launches — the point lists, spec template, and compiled solver are built
once in ``__init__``.
"""
import numpy as np

from pycalphad.gpu.point_solver import (PointList, point_hull, PointBatchSolver,
                                        device_point_hull)

# Compositions outside this window go to the serial fallback: the accelerated
# capability gate routes dilute conditions to the reference path anyway, and
# pure-edge vertices have dedicated clamping there.
_X_MIN = 1e-6
_X_MAX = 1.0 - 1e-6


def _vertex_x_condition(vertex):
    """The single independent (X(comp), value) pair of a binary vertex."""
    items = list(vertex.comp_conds.items())
    if len(items) != 1:
        return None  # ternary+ vertex grouping not implemented yet
    return items[0]


def _batchable(vertex):
    if vertex.has_missing_comp_cond or vertex.is_disordered:
        return False
    xc = _vertex_x_condition(vertex)
    if xc is None:
        return False
    return _X_MIN <= float(xc[1]) <= _X_MAX


class BatchedZPFCalculator:
    """Batched drop-in for ``calculate_zpf_driving_forces`` (C++ backend).

    Parameters
    ----------
    zpf_data : output of ``espei.error_functions.zpf_error.get_zpf_data``
    param_names : sorted fit-parameter symbol names (str order must match
        ``extract_parameters``, e.g. ['VV0000', 'VV0001'])
    pdens : grid point density for the shared energy grid (matches the
        default the backend equilibrium path uses)
    """

    def __init__(self, zpf_data, param_names, pdens=60, verbose=False,
                 backend='cpp', walker_chunk=64, sample_df='grid'):
        self.zpf_data = zpf_data
        self.param_names = [str(p) for p in param_names]
        self.pdens = pdens
        self.verbose = verbose
        self.backend = backend
        self.walker_chunk = int(walker_chunk)
        # 'grid': vectorized max(mu.X - GM) over the shared grid sample
        # (fast; different sample set than the reference's fresh pdens=50
        # draw, so estimates differ within sampling noise, ~0.02 sigma).
        # 'reference': ESPEI's serial calculate_ per vertex (exact match).
        self.sample_df = sample_df
        self._grid_tmpl = None
        self._grid_eval = None

        region0 = self.zpf_data[0]['phase_regions'][0]
        self.species = region0.species
        self.phases = list(region0.phases)
        self.models = dict(region0.models)
        self.dbf = self.zpf_data[0]['dbf']
        self.prf = region0.hyperplane_vertices[0].phase_record_factory \
            if region0.hyperplane_vertices else region0.vertices[0].phase_record_factory
        self.nonvacant = sorted(
            {el.upper() for sp in self.species for el in sp.constituents} - {'VA'})
        self.ncomp = len(self.nonvacant)
        self.state_variables = self.prf.state_variables

        self._plan()

        self.solver = PointBatchSolver(self.species, self.phases, self.models,
                                       self.prf, robust=True, verbose=verbose,
                                       backend=backend)
        # phase_dof per model index, for the underdetermined-vertex rule
        self.dof_of_model = {}
        for ph in self.phases:
            self.dof_of_model[self.solver.name_to_idx[ph]] = self.prf[ph].phase_dof
        if self._hyp_rows:
            T0, P0, comp0, x0 = self._hyp_rows[0][3:]
            self.spec_row0 = self.solver.build_spec_row0(
                {'N': 1.0, 'P': P0, 'T': T0, 'X0': x0}, self.nonvacant[comp0])
        else:
            self.spec_row0 = None

    # ------------------------------------------------------------------ plan
    def _plan(self):
        """Enumerate vertex solves into point-list rows (parameter-free)."""
        # job maps: (dg, pr, hv/vi) -> ('hyp', row) | ('iso', phase, row) | ('serial',)
        self.hyp_jobs = {}
        self.df_jobs = {}
        self._hyp_rows = []                 # (dg, pr, hv, T, P, comp_idx, xval)
        self._iso_rows = {}                 # phase -> list of rows

        for dg, data in enumerate(self.zpf_data):
            for pr, region in enumerate(data['phase_regions']):
                pot = {str(k): float(vv) for k, vv in region.potential_conds.items()}
                T, P = pot['T'], pot.get('P', 101325.0)
                for hv, vtx in enumerate(region.hyperplane_vertices):
                    if vtx.has_missing_comp_cond:
                        continue  # contributes nothing to the hyperplane
                    if _batchable(vtx):
                        xkey, xval = _vertex_x_condition(vtx)
                        comp_idx = self.nonvacant.index(str(xkey)[2:])
                        self.hyp_jobs[(dg, pr, hv)] = ('hyp', len(self._hyp_rows))
                        self._hyp_rows.append((dg, pr, hv, T, P, comp_idx, float(xval)))
                    else:
                        self.hyp_jobs[(dg, pr, hv)] = ('serial',)
                for vi, vtx in enumerate(region.vertices):
                    if _batchable(vtx):
                        xkey, xval = _vertex_x_condition(vtx)
                        comp_idx = self.nonvacant.index(str(xkey)[2:])
                        rows = self._iso_rows.setdefault(vtx.phase_name, [])
                        self.df_jobs[(dg, pr, vi)] = ('iso', vtx.phase_name, len(rows))
                        rows.append((dg, pr, vi, T, P, comp_idx, float(xval)))
                    elif vtx.has_missing_comp_cond and not vtx.is_disordered:
                        # driving force is a sampling ESTIMATE (reference:
                        # max over a fresh pdens=50 sample of mu.X - GM);
                        # evaluated vectorized on the shared grid sample.
                        self.df_jobs[(dg, pr, vi)] = ('sample', vtx.phase_name, T)
                    else:
                        self.df_jobs[(dg, pr, vi)] = ('serial',)

        def _mk_points(rows):
            n = len(rows)
            T = np.array([r[3] for r in rows])
            P = np.array([r[4] for r in rows])
            X = np.zeros((n, self.ncomp))
            mask = np.zeros((n, self.ncomp), dtype=bool)
            for i, r in enumerate(rows):
                ci, xv = r[5], r[6]
                X[i, ci] = xv
                mask[i, ci] = True
                # dependent components split the remainder (binary: 1-x)
                rest = np.setdiff1d(np.arange(self.ncomp), [ci])
                X[i, rest] = (1.0 - xv) / len(rest)
            return T, P, X, mask

        self._hyp_pts_raw = _mk_points(self._hyp_rows) if self._hyp_rows else None
        self._iso_pts_raw = {ph: _mk_points(rows)
                             for ph, rows in self._iso_rows.items()}
        allT = [r[3] for r in self._hyp_rows]
        for rows in self._iso_rows.values():
            allT += [r[3] for r in rows]
        allP = [r[4] for r in self._hyp_rows]
        for rows in self._iso_rows.values():
            allP += [r[4] for r in rows]
        self._unique_T = np.unique(np.asarray(allT)) if allT else np.array([300.0])
        self._plan_vectorized()
        assert len(np.unique(allP)) <= 1, "mixed pressures not supported yet"
        self._P = float(allP[0]) if allP else 101325.0
        self._grid_phase_masks = None  # built after first grid

    def _plan_vectorized(self):
        """Index arrays for the vectorized ensemble assembly. Only valid when
        NO vertex needs the serial fallback (checked; else the per-walker
        loop assembly runs)."""
        self._vec_ok = all(j[0] != 'serial' for j in self.hyp_jobs.values()) \
            and all(j[0] != 'serial' for j in self.df_jobs.values())
        regions = []           # (dg, weight, hyp_start, hyp_stop, vertex specs)
        hyp_starts, hyp_stops = [], []
        vtx = []               # per vertex: (dg, region_idx, kind, ...)
        hyp_cursor = 0
        for dg, data in enumerate(self.zpf_data):
            for pr, region in enumerate(data['phase_regions']):
                r_id = len(hyp_starts)
                n_h = sum(1 for hv in range(len(region.hyperplane_vertices))
                          if (dg, pr, hv) in self.hyp_jobs
                          and self.hyp_jobs[(dg, pr, hv)][0] == 'hyp')
                hyp_starts.append(hyp_cursor)
                hyp_stops.append(hyp_cursor + n_h)
                hyp_cursor += n_h
                for vi, v in enumerate(region.vertices):
                    job = self.df_jobs[(dg, pr, vi)]
                    if job[0] == 'iso':
                        vtx.append((dg, r_id, 0, job[1], job[2],
                                    np.asarray(v.composition, dtype=np.float64), 0.0))
                    elif job[0] == 'sample':
                        vtx.append((dg, r_id, 1, job[1], -1, None, job[2]))
                    else:
                        vtx.append((dg, r_id, 2, None, -1, None, 0.0))
                regions.append((dg, data['weight']))
        self._v_hyp_starts = np.asarray(hyp_starts, dtype=np.intp)
        self._v_hyp_stops = np.asarray(hyp_stops, dtype=np.intp)
        self._v_region_dg = np.asarray([r[0] for r in regions], dtype=np.intp)
        self._v_region_wt = np.asarray([r[1] for r in regions], dtype=np.float64)
        self._v_vtx = vtx
        self._v_vtx_region = np.asarray([t[1] for t in vtx], dtype=np.intp)
        self._v_vtx_dg = np.asarray([t[0] for t in vtx], dtype=np.intp)
        self._v_vtx_wt = self._v_region_wt[self._v_vtx_region]

    def _assemble_all(self, hyp_res, iso_res, params_matrix, grid_GM):
        """Vectorized _assemble across the whole ensemble (no serial jobs)."""
        W = len(params_matrix)
        nT = len(self._unique_T)
        ncomp = self.ncomp
        n_hyp = hyp_res['MU'].shape[0] // W if hyp_res is not None else 0
        n_reg = len(self._v_hyp_starts)
        MP = int(self.solver.dynamic_sizes['MAX_PHASES'])

        # per-row MU with the underdetermined rule, vectorized
        if n_hyp:
            MU = hyp_res['MU'][:, :ncomp].reshape(W, n_hyp, ncomp).copy()
            NPh = hyp_res['NP'].reshape(W, n_hyp, MP)
            ids = hyp_res['phase_ids'].reshape(W, n_hyp, MP)
            Yh = hyp_res['Y'].reshape(W, n_hyp, MP, -1)
            nst = hyp_res['num_stable_phases'].reshape(W, n_hyp)
            stable = NPh > 1e-9
            scount = stable.sum(-1)
            single = (nst == 1) | (scount == 1)
            k = np.argmax(stable, axis=-1)          # first stable; 0 if none
            model_k = np.take_along_axis(ids, k[..., None], -1)[..., 0]
            max_model = max(self.dof_of_model) if self.dof_of_model else 0
            dof_lut = np.zeros(max_model + 1, dtype=np.intp)
            for mi, pd in self.dof_of_model.items():
                dof_lut[mi] = pd
            pdof = dof_lut[np.clip(model_k, 0, max_model)]
            Yk = np.take_along_axis(Yh, k[..., None, None], 2)[..., 0, :]
            in_dof = np.arange(Yh.shape[-1])[None, None, :] < pdof[..., None]
            ok = np.isclose(Yk, 1.0) | np.isnan(Yk) | ~in_dof
            undet = single & (ok.all(-1) | (pdof == 0))
            MU[undet] = np.nan

            # region targets: segment nanmean over contiguous hyp rows
            zed = np.zeros((W, 1, ncomp))
            csum = np.concatenate([zed, np.cumsum(np.nan_to_num(MU), axis=1)], 1)
            ccnt = np.concatenate([zed, np.cumsum(~np.isnan(MU), axis=1)], 1)
            sums = csum[:, self._v_hyp_stops] - csum[:, self._v_hyp_starts]
            cnts = ccnt[:, self._v_hyp_stops] - ccnt[:, self._v_hyp_starts]
            with np.errstate(invalid='ignore', divide='ignore'):
                target = np.where(cnts > 0, sums / cnts, np.nan)   # (W, n_reg, ncomp)
        else:
            target = np.full((W, n_reg, ncomp), np.nan)
        bad_region = np.isnan(target).any(-1)                       # (W, n_reg)

        # per-vertex driving forces
        n_v = len(self._v_vtx)
        dfs = np.zeros((W, n_v))
        t = self._grid_tmpl
        # iso vertices grouped by phase
        by_phase_iso = {}
        by_phase_sample = {}
        for gvi, spec in enumerate(self._v_vtx):
            if spec[2] == 0:
                by_phase_iso.setdefault(spec[3], []).append(gvi)
            elif spec[2] == 1:
                by_phase_sample.setdefault(spec[3], []).append(gvi)
        for ph, gvis in by_phase_iso.items():
            gvis = np.asarray(gvis, dtype=np.intp)
            rows = np.asarray([self._v_vtx[g][4] for g in gvis], dtype=np.intp)
            comp = np.stack([self._v_vtx[g][5] for g in gvis])       # (nv, ncomp)
            reg = self._v_vtx_region[gvis]
            n_ph = iso_res[ph]['GM'].shape[0] // W
            gm = iso_res[ph]['GM'].reshape(W, n_ph)[:, rows]         # (W, nv)
            tv = target[:, reg, :]                                   # (W, nv, ncomp)
            dfs[:, gvis] = np.einsum('wvc,vc->wv', tv, comp) - gm
        for ph, gvis in by_phase_sample.items():
            gvis = np.asarray(gvis, dtype=np.intp)
            cols = t.phase_cols.get(ph)
            reg = self._v_vtx_region[gvis]
            if cols is None or cols.size == 0:
                dfs[:, gvis] = 0.0
                continue
            t_idx = np.searchsorted(
                self._unique_T, np.asarray([self._v_vtx[g][6] for g in gvis]))
            x_cols = t.X[0][cols, :ncomp]                            # (m, ncomp)
            combo = (np.arange(W)[:, None] * nT) + t_idx[None, :]    # (W, nv)
            gm = grid_GM[combo][:, :, cols]                          # (W, nv, m)
            proj = np.einsum('mc,wvc->wvm', x_cols, target[:, reg, :])
            with np.errstate(invalid='ignore'):
                dfs[:, gvis] = np.nanmax(proj - gm, axis=-1) \
                    if np.isnan(proj).any() else np.max(proj - gm, axis=-1)

        # NaN-target regions: zero driving force (reference semantics)
        dfs = np.where(bad_region[:, self._v_vtx_region], 0.0, dfs)

        # ragged per-data-group lists in original vertex order
        out = []
        n_dg = len(self.zpf_data)
        dg_slices = [np.flatnonzero(self._v_vtx_dg == dg) for dg in range(n_dg)]
        for w in range(W):
            d = [dfs[w, sl].tolist() for sl in dg_slices]
            v = [self._v_vtx_wt[sl].tolist() for sl in dg_slices]
            out.append((d, v))
        return out

    # ------------------------------------------------------------ grid + run
    def _make_grid(self, params_dict):
        from pycalphad import calculate
        grid = calculate(self.dbf, [str(s) for s in self.species], self.phases,
                         model=self.models, fake_points=True,
                         phase_records=self.prf, output='GM',
                         parameters=params_dict, to_xarray=False,
                         pdens=self.pdens, N=1.0, P=self._P,
                         T=self._unique_T.tolist())
        return grid

    def _filtered_grid(self, grid, phase):
        """Phase-restricted grid views (rows of `phase` + fake points)."""
        from types import SimpleNamespace
        gp = np.asarray(grid.Phase)
        row0 = gp.reshape(-1, gp.shape[-1])[0]
        if self._grid_phase_masks is None:
            self._grid_phase_masks = {}
        sel = self._grid_phase_masks.get(phase)
        if sel is None:
            sel = np.flatnonzero((row0 == phase) | (row0 == '_FAKE_'))
            self._grid_phase_masks[phase] = sel
        gm = np.asarray(grid.GM)
        gx = np.asarray(grid.X)
        gy = np.asarray(grid.Y)
        return SimpleNamespace(
            GM=gm.reshape(-1, gm.shape[-1])[:, sel],
            X=gx.reshape((-1,) + gx.shape[-2:])[:, sel],
            Y=gy.reshape((-1,) + gy.shape[-2:])[:, sel],
            Phase=gp.reshape(-1, gp.shape[-1])[:, sel],
            attrs={},
        )

    def _mu_with_underdetermined_rule(self, res, row):
        """Per-vertex MU; NaN row if single stable phase with no internal dof
        (reference rule in estimate_hyperplane)."""
        mu = res['MU'][row, :self.ncomp].copy()
        stable = np.flatnonzero(res['NP'][row] > 1e-9)
        if res['num_stable_phases'][row] == 1 or stable.size == 1:
            k = stable[0] if stable.size else 0
            model_idx = int(res['phase_ids'][row, k])
            pdof = self.dof_of_model.get(model_idx, 0)
            y = res['Y'][row, k, :pdof]
            if pdof == 0 or np.all(np.isclose(y, 1.0) | np.isnan(y)):
                return np.full_like(mu, np.nan)
        return mu

    # --------------------------------------------------------- serial pieces
    def _serial_hyperplane_mu(self, region, vertex, params_dict):
        """Reference per-vertex MU (Workspace), for non-batchable vertices."""
        from pycalphad import Workspace, variables as v
        cond_dict = {**vertex.comp_conds, **region.potential_conds}
        wks = Workspace(database=self.dbf, components=self.species,
                        phases=self.phases, models=self.models,
                        phase_record_factory=vertex.phase_record_factory,
                        conditions=cond_dict, parameters=params_dict)
        mu = np.array([float(wks.get(v.MU(el))) for el in self.nonvacant])
        num_phases = int(np.sum(wks.eq.Phase.squeeze() != ''))
        Y_values = wks.eq.Y.squeeze()
        no_internal_dof = np.all(np.isclose(Y_values, 1.0) | np.isnan(Y_values))
        if num_phases == 1 and no_internal_dof:
            return np.full_like(mu, np.nan)
        return mu

    def _serial_driving_force(self, data, region, vertex, target, parameters):
        from espei.error_functions.zpf_error import driving_force_to_hyperplane
        return driving_force_to_hyperplane(
            target, region, data['dbf'], data['parameter_dict'], vertex,
            parameters)

    # ------------------------------------------------------------------ main
    def driving_forces(self, parameters):
        """Batched equivalent of ``calculate_zpf_driving_forces``."""
        parameters = np.asarray(parameters, dtype=np.float64)
        params_dict = dict(zip(self.param_names, parameters))
        try:
            from espei.shadow_functions import update_phase_record_parameters
            update_phase_record_parameters(self.prf, parameters)
        except Exception:
            pass

        grid = self._make_grid(params_dict)

        hyp_res = None
        if self._hyp_pts_raw is not None:
            T, P, X, mask = self._hyp_pts_raw
            prow = np.tile(parameters, (len(T), 1))
            pts = PointList(T=T, P=P, N=1.0, X=X, x_cond_mask=mask, params=prow)
            hull = point_hull(pts, grid, self.nonvacant, grid_T=self._unique_T)
            hyp_res = self.solver.solve(pts, hull, grid, self.spec_row0,
                                        self.state_variables, self.nonvacant,
                                        grid_T=self._unique_T)

        iso_res = {}
        for ph, (T, P, X, mask) in self._iso_pts_raw.items():
            prow = np.tile(parameters, (len(T), 1))
            pts = PointList(T=T, P=P, N=1.0, X=X, x_cond_mask=mask,
                            phase_restrict=np.array([ph] * len(T), dtype=object),
                            params=prow)
            hull = point_hull(pts, grid, self.nonvacant, grid_T=self._unique_T)
            gfilt = self._filtered_grid(grid, ph)
            iso_res[ph] = self.solver.solve(pts, hull, grid, self.spec_row0,
                                            self.state_variables, self.nonvacant,
                                            restrict_grid_views=gfilt,
                                            grid_T=self._unique_T)

        self._ensure_grid_template()
        return self._assemble(hyp_res, iso_res, parameters, params_dict,
                              grid_GM=np.asarray(grid.GM).reshape(-1, np.asarray(grid.GM).shape[-1]))

    def _sample_df(self, grid_GM, combo_off, phase, T, target):
        """Vectorized sampling estimate: max over the phase's grid sample of
        target.X - GM (reference: driving_force_to_hyperplane missing-comp
        branch, which uses a fresh pdens=50 sample; here the shared grid
        sample serves — same estimator, slightly different sample set)."""
        t = self._grid_tmpl
        cols = t.phase_cols.get(phase)
        if cols is None or cols.size == 0:
            return 0.0
        ti = int(np.searchsorted(self._unique_T, T))
        gm_row = grid_GM[combo_off + ti, cols]
        x_cols = t.X[0][cols, :self.ncomp]
        return float(np.max(x_cols @ target - gm_row))

    def _assemble(self, hyp_res, iso_res, parameters, params_dict,
                  row_off_hyp=0, row_off_iso=None, grid_GM=None, combo_off=0):
        """Driving-force/weight assembly (reference semantics). Row offsets
        select a walker's slice out of walker-major batched results."""
        row_off_iso = row_off_iso or {}
        driving_forces, weights = [], []
        for dg, data in enumerate(self.zpf_data):
            data_dfs, data_wts = [], []
            weight = data['weight']
            for pr, region in enumerate(data['phase_regions']):
                rows = []
                for hv, vtx in enumerate(region.hyperplane_vertices):
                    job = self.hyp_jobs.get((dg, pr, hv))
                    if job is None:
                        continue
                    if job[0] == 'hyp':
                        rows.append(self._mu_with_underdetermined_rule(
                            hyp_res, row_off_hyp + job[1]))
                    else:
                        rows.append(self._serial_hyperplane_mu(region, vtx, params_dict))
                if rows:
                    with np.errstate(invalid='ignore'):
                        target = np.nanmean(np.asarray(rows), axis=0)
                else:
                    target = np.full(self.ncomp, np.nan)

                if np.any(np.isnan(target)):
                    data_dfs.extend([0] * len(region.vertices))
                    data_wts.extend([weight] * len(region.vertices))
                    continue

                for vi, vtx in enumerate(region.vertices):
                    job = self.df_jobs[(dg, pr, vi)]
                    if job[0] == 'iso':
                        gm = float(iso_res[job[1]]['GM'][row_off_iso.get(job[1], 0) + job[2]])
                        df = float(np.dot(target, vtx.composition) - gm)
                    elif (job[0] == 'sample' and grid_GM is not None
                          and self.sample_df == 'grid'):
                        df = self._sample_df(grid_GM, combo_off, job[1], job[2], target)
                    else:
                        df = self._serial_driving_force(data, region, vtx,
                                                        target, parameters)
                    data_dfs.append(df)
                    data_wts.append(weight)
            driving_forces.append(data_dfs)
            weights.append(data_wts)
        return driving_forces, weights

    # ----------------------------------------------------- walker batching
    def _stacked_grids(self, params_matrix):
        """One grid per walker, stacked along the statevar-combo axis.

        Sampling points are parameter-independent — only GM changes — so
        X/Y/Phase are tiled views of walker 0's arrays."""
        from types import SimpleNamespace
        gm_rows = []
        first = None
        for pvec in params_matrix:
            params_dict = dict(zip(self.param_names, pvec))
            try:
                from espei.shadow_functions import update_phase_record_parameters
                update_phase_record_parameters(self.prf, np.asarray(pvec, dtype=np.float64))
            except Exception:
                pass
            g = self._make_grid(params_dict)
            gm = np.asarray(g.GM)
            gm_rows.append(gm.reshape(-1, gm.shape[-1]))
            if first is None:
                gx = np.asarray(g.X)
                gy = np.asarray(g.Y)
                gp = np.asarray(g.Phase)
                first = (gx.reshape((-1,) + gx.shape[-2:]),
                         gy.reshape((-1,) + gy.shape[-2:]),
                         gp.reshape(-1, gp.shape[-1]))
        W = len(params_matrix)
        return SimpleNamespace(
            GM=np.concatenate(gm_rows, axis=0),
            X=np.tile(first[0], (W, 1, 1)),
            Y=np.tile(first[1], (W, 1, 1)),
            Phase=np.tile(first[2], (W, 1)),
            attrs={},
        )

    def _ensure_grid_template(self):
        """One-time: grid structure + per-phase sample dof matrices.

        The sampling (X/Y/Phase and the fake points) is parameter-independent;
        only real points' GM changes with the fit parameters, and those are
        re-evaluated directly through the generated evaluator (which appends
        prf.param_values live)."""
        if self._grid_tmpl is not None:
            return
        from types import SimpleNamespace
        from pycalphad.gpu.gpu_calculate import get_grid_evaluator
        g0 = self._make_grid(dict(zip(self.param_names,
                                      np.zeros(len(self.param_names)))))
        gm = np.asarray(g0.GM)
        M = gm.shape[-1]
        nT = len(self._unique_T)
        gm2 = gm.reshape(-1, M)
        gx = np.asarray(g0.X).reshape((-1,) + np.asarray(g0.X).shape[-2:])
        gy = np.asarray(g0.Y).reshape((-1,) + np.asarray(g0.Y).shape[-2:])
        gp = np.asarray(g0.Phase).reshape(-1, M)
        row0 = gp[0]
        # per-phase dof matrices: rows are (T-combo major, column minor);
        # Y sampling is identical across T combos.
        phase_cols, phase_dofs = {}, {}
        for ph in self.phases:
            cols = np.flatnonzero(row0 == ph)
            if cols.size == 0:
                continue
            pdof = self.prf[ph].phase_dof
            y0 = gy[0][cols, :pdof]                      # (ncols, pdof)
            rows = np.empty((nT * cols.size, 3 + pdof), dtype=np.float64)
            rows[:, 0] = 1.0
            rows[:, 1] = self._P
            rows[:, 2] = np.repeat(self._unique_T, cols.size)
            rows[:, 3:] = np.tile(y0, (nT, 1))
            phase_cols[ph] = cols
            phase_dofs[ph] = np.ascontiguousarray(rows)
        fake_cols = np.flatnonzero(row0 == '_FAKE_')
        self._grid_tmpl = SimpleNamespace(
            M=M, nT=nT, X=gx, Y=gy, Phase=gp, GM_fake=gm2[:, fake_cols].copy(),
            fake_cols=fake_cols, phase_cols=phase_cols, phase_dofs=phase_dofs)
        self._grid_eval = get_grid_evaluator(
            'cpp', self.species, self.phases, self.models, self.prf,
            verbose=self.verbose)

    def _ext_blocks(self, key, W, Y_ptr, X_ptr, PID_ptr, GM_ptr, M, tmpl_block):
        """Compact external-pointer DeviceGrid blocks (grid_ext_mode=1): each
        (walker, T) block is ~100 bytes of header + pointers; Y/X/PhaseID are
        shared walker-invariant buffers, GM points into the per-walker energy
        buffer. Built once per (group, W) — zero per-step block traffic."""
        cache = getattr(self, '_ext_block_cache', None)
        if cache is None:
            self._ext_block_cache = cache = {}
        hit = cache.get((key, W))
        if hit is not None:
            return hit
        nT = len(self._unique_T)
        n_ph = int(tmpl_block['num_mappable_phases_in_grid'])
        dtype = [('num_grid_points_total', 'i4'), ('phase_dof_stride_Y', 'i4'),
                 ('num_components_stride_X', 'i4'), ('actual_y_data_size', 'i4'),
                 ('actual_x_data_size', 'i4'), ('actual_gm_data_size', 'i4'),
                 ('actual_phase_id_data_size', 'i4'), ('ext_mode', 'i4'),
                 ('Y_ext', 'u8'), ('X_ext', 'u8'), ('GM_ext', 'u8'), ('PID_ext', 'u8'),
                 ('phase_grid_indices_start', f'{n_ph}i4'),
                 ('phase_grid_indices_stop', f'{n_ph}i4'),
                 ('num_mappable_phases_in_grid', 'i4')]
        pad = (-np.dtype(dtype).itemsize) % 8
        if pad:
            dtype.append(('_tail_pad', f'{pad}u1'))
        blocks = np.zeros(W * nT, dtype=dtype)
        blocks['num_grid_points_total'] = M
        blocks['phase_dof_stride_Y'] = int(tmpl_block['phase_dof_stride_Y'])
        blocks['num_components_stride_X'] = int(tmpl_block['num_components_stride_X'])
        blocks['actual_y_data_size'] = 1
        blocks['actual_x_data_size'] = 1
        blocks['actual_gm_data_size'] = 1
        blocks['actual_phase_id_data_size'] = 2
        blocks['ext_mode'] = 1
        blocks['Y_ext'] = Y_ptr
        blocks['X_ext'] = X_ptr
        blocks['PID_ext'] = PID_ptr
        blocks['GM_ext'] = GM_ptr + np.arange(W * nT, dtype=np.uint64) * (M * 8)
        blocks['phase_grid_indices_start'] = tmpl_block['phase_grid_indices_start']
        blocks['phase_grid_indices_stop'] = tmpl_block['phase_grid_indices_stop']
        blocks['num_mappable_phases_in_grid'] = n_ph
        cache[(key, W)] = blocks
        return blocks

    def _blocks_from_template(self, key, grid_src, GM_rows):
        """DeviceGrid blocks for stacked walker grids: struct-pack ONE combo
        set through the reference packer, then tile and overwrite the GM
        field vectorized (X/Y/PhaseID are walker-invariant)."""
        from pycalphad.gpu.gpu_equilibrium import \
            _prepare_grid_data_for_gpu_from_calculate_result
        ds = self.solver.dynamic_sizes
        tmpl = getattr(self, '_block_tmpls', None)
        if tmpl is None:
            self._block_tmpls = tmpl = {}
        entry = tmpl.get(key)
        if entry is None:
            from types import SimpleNamespace
            nT = len(self._unique_T)
            one = SimpleNamespace(GM=np.asarray(grid_src.GM)[:nT],
                                  X=np.asarray(grid_src.X)[:nT],
                                  Y=np.asarray(grid_src.Y)[:nT],
                                  Phase=np.asarray(grid_src.Phase)[:nT],
                                  attrs={})
            blocks, _ = _prepare_grid_data_for_gpu_from_calculate_result(
                one, self.solver.name_to_idx, int(ds['MAX_PHASES']),
                int(ds['MAX_DOF_PER_PHASE']), int(ds['MAX_COMPONENTS']), False)
            tmpl[key] = entry = blocks
        n_combos = GM_rows.shape[0]
        W = n_combos // entry.shape[0]
        big_cache = getattr(self, '_big_block_cache', None)
        if big_cache is None:
            self._big_block_cache = big_cache = {}
        big = big_cache.get((key, W))
        if big is None:
            big = np.tile(entry, W)
            big_cache[(key, W)] = big
        m = big['GM_ptr_data'].shape[1]
        big['GM_ptr_data'][:] = GM_rows[:, :m]
        return big

    def _legacy_tmpl_block(self, key):
        """One legacy-packed block of this group (header/stride/phase-range
        source for the compact ext blocks)."""
        cache = getattr(self, '_legacy_blk_cache', None)
        if cache is None:
            self._legacy_blk_cache = cache = {}
        blk = cache.get(key)
        if blk is None:
            from types import SimpleNamespace
            from pycalphad.gpu.gpu_equilibrium import \
                _prepare_grid_data_for_gpu_from_calculate_result
            self._ensure_grid_template()
            t = self._grid_tmpl
            ds = self.solver.dynamic_sizes
            sel = None if key == 'all' else self._grid_phase_masks[key]
            one = SimpleNamespace(
                GM=t.GM_fake[:1] if False else (np.zeros((1, t.M)) if sel is None
                                                else np.zeros((1, sel.size))),
                X=(t.X[:1] if sel is None else t.X[:1, sel]),
                Y=(t.Y[:1] if sel is None else t.Y[:1, sel]),
                Phase=(t.Phase[:1] if sel is None else t.Phase[:1, sel]),
                attrs={})
            blocks, _ = _prepare_grid_data_for_gpu_from_calculate_result(
                one, self.solver.name_to_idx, int(ds['MAX_PHASES']),
                int(ds['MAX_DOF_PER_PHASE']), int(ds['MAX_COMPONENTS']), False)
            blk = blocks[0]
            cache[key] = blk
        return blk

    def _group_buffers(self, key):
        """Persistent walker-invariant grid arrays for one launch group
        ('all' or a restricted phase): Y, X, PhaseID (+ device copies and
        per-phase dof matrices on CUDA). Built once."""
        cache = getattr(self, '_group_buf_cache', None)
        if cache is None:
            self._group_buf_cache = cache = {}
        buf = cache.get(key)
        if buf is not None:
            return buf
        self._ensure_grid_template()
        t = self._grid_tmpl
        if key == 'all':
            sel = None
            Y = np.ascontiguousarray(t.Y[0])
            X = np.ascontiguousarray(t.X[0])
            Phase = t.Phase[0]
        else:
            sel = self._grid_phase_masks[key]
            Y = np.ascontiguousarray(t.Y[0][sel])
            X = np.ascontiguousarray(t.X[0][sel])
            Phase = t.Phase[0][sel]
        pid = np.full(Phase.shape[0], -1, dtype=np.int32)
        for name, idx in self.solver.name_to_idx.items():
            pid[Phase == name] = idx
        buf = {'sel': sel, 'Y': Y, 'X': X, 'PID': pid, 'M': X.shape[0]}
        if self.backend == 'cuda':
            cp = self.solver._cp
            buf['dY'] = cp.asarray(Y)
            buf['dX'] = cp.asarray(X)
            buf['dPID'] = cp.asarray(pid)
        cache[key] = buf
        return buf

    def _walker_gm_device(self, params_matrix):
        """Per-walker grid energies computed ON DEVICE (grid_eval_params_kernel
        over the cached sample dofs); returns (device_gm (W*nT, M), host copy).
        Fake-point energies are parameter-independent and pre-filled."""
        cp = self.solver._cp
        t = self._grid_tmpl
        W = len(params_matrix)
        nT = t.nT
        cache = getattr(self, '_dev_gm_cache', None)
        if cache is None:
            self._dev_gm_cache = cache = {}
        entry = cache.get(W)
        if entry is None:
            d_gm = cp.empty((W * nT, t.M), dtype=cp.float64)
            # fake cols: same per combo; tile template fakes across walkers
            fake_host = np.tile(t.GM_fake, (W, 1))
            d_gm[:, t.fake_cols] = cp.asarray(fake_host)
            d_dofs = {ph: cp.asarray(rows) for ph, rows in t.phase_dofs.items()}
            kern = self.solver.module.get_function('grid_eval_params_kernel')
            cache[W] = entry = (d_gm, d_dofs, kern)
        d_gm, d_dofs, kern = entry
        d_params = cp.asarray(np.ascontiguousarray(params_matrix, dtype=np.float64))
        n_par = d_params.shape[1]
        tpb = 128
        for w in range(W):
            base = d_gm[w * nT:(w + 1) * nT]
            for ph, cols in t.phase_cols.items():
                rows = d_dofs[ph]
                n_rows = rows.shape[0]
                kern(((n_rows + tpb - 1) // tpb,), (tpb,),
                     (np.int32(self.solver.name_to_idx[ph]), rows,
                      d_params[w], np.int32(n_par),
                      np.int64(n_rows), np.int32(rows.shape[1]),
                      base, np.int32(cols.size), np.int32(int(cols[0])),
                      np.int64(t.M)))
        cp.cuda.runtime.deviceSynchronize()
        return d_gm, cp.asnumpy(d_gm)

    def _stacked_grids_fast(self, params_matrix):
        """Per-walker GM via the generated evaluator on cached sample dofs."""
        from types import SimpleNamespace
        self._ensure_grid_template()
        t = self._grid_tmpl
        W = len(params_matrix)
        GM = np.empty((W * t.nT, t.M), dtype=np.float64)
        pv = np.asarray(self.prf.param_values, dtype=np.float64).reshape(-1)
        for w, pvec in enumerate(params_matrix):
            pv_view = self.prf.param_values
            np.asarray(pv_view).reshape(-1)[:] = pvec  # in-place, evaluator reads live
            blk = GM[w * t.nT:(w + 1) * t.nT]
            blk[:, t.fake_cols] = t.GM_fake
            for ph, cols in t.phase_cols.items():
                out = np.empty(t.phase_dofs[ph].shape[0], dtype=np.float64)
                self._grid_eval(ph, t.phase_dofs[ph], out)
                blk[:, cols] = out.reshape(t.nT, cols.size)
        np.asarray(self.prf.param_values).reshape(-1)[:] = pv  # restore
        return SimpleNamespace(
            GM=GM,
            X=np.broadcast_to(t.X[0], (W * t.nT,) + t.X.shape[1:]),
            Y=np.broadcast_to(t.Y[0], (W * t.nT,) + t.Y.shape[1:]),
            Phase=np.broadcast_to(t.Phase[0], (W * t.nT, t.M)), attrs={})

    def _walker_major(self, raw, params_matrix, key=None):
        """Tile a point group walker-major with per-walker parameter rows and
        explicit (walker*nT + t) combo indices. The tiled point list is
        STATIC across MCMC steps (only the parameter rows change), so it is
        cached per (group, ensemble size) and params are poked in place."""
        T, P, X, mask = raw
        W = len(params_matrix)
        n = len(T)
        cache = getattr(self, '_wm_cache', None)
        if cache is None:
            self._wm_cache = cache = {}
        entry = cache.get((key, W)) if key is not None else None
        if entry is None:
            nT = len(self._unique_T)
            t_idx = np.searchsorted(self._unique_T, T)
            combo = np.repeat(np.arange(W), n) * nT + np.tile(t_idx, W)
            pts = PointList(T=np.tile(T, W), P=np.tile(P, W), N=1.0,
                            X=np.tile(X, (W, 1)), x_cond_mask=np.tile(mask, (W, 1)),
                            params=np.empty((W * n, len(self.param_names))))
            if key is not None:
                cache[(key, W)] = (pts, combo)
            entry = (pts, combo)
        pts, combo = entry
        pts.params[:] = np.repeat(np.asarray(params_matrix, dtype=np.float64), n, axis=0)
        return pts, combo, n

    def driving_forces_ensemble(self, params_matrix):
        """Batched driving forces for an ENSEMBLE of parameter vectors.

        One all-phase launch + one launch per restricted phase covers every
        (walker, vertex) pair. Returns a list of (driving_forces, weights)
        pairs, one per walker, each identical in structure to
        ``driving_forces(params_matrix[w])``.
        """
        params_matrix = np.atleast_2d(np.asarray(params_matrix, dtype=np.float64))
        if len(params_matrix) > self.walker_chunk:
            out = []
            for cs in range(0, len(params_matrix), self.walker_chunk):
                out.extend(self.driving_forces_ensemble(
                    params_matrix[cs:cs + self.walker_chunk]))
            return out
        W = len(params_matrix)
        if self.backend == 'cuda':
            self._ensure_grid_template()
            # device path: walker energies computed on device; ONE D2H of the
            # GM buffer serves the hull/sample-df host code; ext blocks carry
            # device pointers with zero per-step block re-upload.
            from types import SimpleNamespace
            d_gm, gm_host = self._walker_gm_device(params_matrix)
            t = self._grid_tmpl
            grid = SimpleNamespace(
                GM=gm_host,
                X=np.broadcast_to(t.X[0], (W * t.nT,) + t.X.shape[1:]),
                Y=np.broadcast_to(t.Y[0], (W * t.nT,) + t.Y.shape[1:]),
                Phase=np.broadcast_to(t.Phase[0], (W * t.nT, t.M)), attrs={})
            self._d_gm_current = d_gm
        else:
            grid = self._stacked_grids_fast(params_matrix)
            self._d_gm_current = None
            self._ensure_grid_template()

        def _blocks_for(key, gfilt_gm_host):
            buf = self._group_buffers(key)
            if self.backend == 'cuda':
                cp = self.solver._cp
                if key == 'all':
                    d_g = self._d_gm_current
                else:
                    dcache = getattr(self, '_dev_gmf_cache', None)
                    if dcache is None:
                        self._dev_gmf_cache = dcache = {}
                    d_g = self._d_gm_current[:, cp.asarray(buf['sel'])]
                    dcache[key] = d_g  # keep alive through the launch
                return self._ext_blocks(key, W, int(buf['dY'].data.ptr),
                                        int(buf['dX'].data.ptr),
                                        int(buf['dPID'].data.ptr),
                                        int(d_g.data.ptr), buf['M'],
                                        self._legacy_tmpl_block(key))
            # cpp: host pointers; GM host buffer must stay referenced
            self._host_gm_keepalive = getattr(self, '_host_gm_keepalive', {})
            self._host_gm_keepalive[key] = gfilt_gm_host
            return self._ext_blocks(key, W, buf['Y'].ctypes.data,
                                    buf['X'].ctypes.data, buf['PID'].ctypes.data,
                                    gfilt_gm_host.ctypes.data, buf['M'],
                                    self._legacy_tmpl_block(key))

        hyp_res, n_hyp = None, 0
        if self._hyp_pts_raw is not None:
            pts, combo, n_hyp = self._walker_major(self._hyp_pts_raw, params_matrix, key='hyp')
            t = self._grid_tmpl
            hull = device_point_hull(pts, self.solver, t.X[0], grid.GM, combo,
                                     t.Phase[0], t.Y[0], self.nonvacant)
            gm_all = np.ascontiguousarray(grid.GM)
            blocks = _blocks_for('all', gm_all)
            hyp_res = self.solver.solve(pts, hull, grid, self.spec_row0,
                                        self.state_variables, self.nonvacant,
                                        combo_idx=combo, grid_blocks=blocks)

        iso_res, n_iso = {}, {}
        for ph, raw in self._iso_pts_raw.items():
            pts, combo, n_ph = self._walker_major(raw, params_matrix, key=ph)
            pts.phase_restrict[:] = ph
            gfilt = self._filtered_grid(grid, ph)
            t = self._grid_tmpl
            sel = self._grid_phase_masks[ph]
            gm_f = np.ascontiguousarray(np.asarray(gfilt.GM))
            hull = device_point_hull(pts, self.solver,
                                     np.ascontiguousarray(t.X[0][sel]),
                                     gm_f, combo,
                                     t.Phase[0][sel],
                                     np.ascontiguousarray(t.Y[0][sel]),
                                     self.nonvacant)
            blocks = _blocks_for(ph, gm_f)
            iso_res[ph] = self.solver.solve(pts, hull, grid, self.spec_row0,
                                            self.state_variables, self.nonvacant,
                                            restrict_grid_views=gfilt,
                                            combo_idx=combo, grid_blocks=blocks)
            n_iso[ph] = n_ph

        if self._vec_ok:
            return self._assemble_all(hyp_res, iso_res, params_matrix, grid.GM)
        out = []
        for w, pvec in enumerate(params_matrix):
            params_dict = dict(zip(self.param_names, pvec))
            try:
                from espei.shadow_functions import update_phase_record_parameters
                update_phase_record_parameters(self.prf, pvec)
            except Exception:
                pass
            out.append(self._assemble(
                hyp_res, iso_res, pvec, params_dict,
                row_off_hyp=w * n_hyp,
                row_off_iso={ph: w * n for ph, n in n_iso.items()},
                grid_GM=grid.GM, combo_off=w * len(self._unique_T)))
        return out

    def log_prob_ensemble(self, params_matrix, data_weight=1.0):
        """(n_walkers,) ZPF log-likelihood vector for emcee vectorize=True."""
        from scipy.stats import norm
        results = self.driving_forces_ensemble(params_matrix)
        lps = np.empty(len(results))
        for w, (dfs, wts) in enumerate(results):
            d = np.concatenate([np.asarray(x, dtype=np.float64) for x in dfs])
            v = np.concatenate([np.asarray(x, dtype=np.float64) for x in wts])
            if np.any(np.isinf(d) | np.isnan(d)):
                lps[w] = -np.inf
            else:
                lps[w] = float(np.sum(norm.logpdf(d, loc=0, scale=1000 / data_weight / v)))
        return lps

    def likelihood(self, parameters, data_weight=1.0):
        """Batched equivalent of ``calculate_zpf_error``."""
        from scipy.stats import norm
        dfs, wts = self.driving_forces(parameters)
        dfs = np.concatenate([np.asarray(d, dtype=np.float64) for d in dfs])
        wts = np.concatenate([np.asarray(w, dtype=np.float64) for w in wts])
        if np.any(np.isinf(dfs) | np.isnan(dfs)):
            return -np.inf
        return float(np.sum(norm.logpdf(dfs, loc=0, scale=1000 / data_weight / wts)))
