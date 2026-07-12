"""Walker-ensemble batch solving machinery (generic).

Building blocks for evaluating MANY independent parameter vectors
("walkers") against a fixed set of condition points in a few solver
launches — the pattern used by MCMC parameter estimation (see ESPEI's
BatchedZPFResidual, which subclasses :class:`EnsemblePointBatcher`), but
independent of any particular likelihood: nothing here knows what a
tie-line or a driving force is.

Provides, per (system, statevar domain):
- a cached grid template (parameter-independent sampling; per-phase dof
  matrices for on-device re-evaluation of energies per parameter vector);
- walker-stacked grid energies on the C++ or CUDA backend;
- external-pointer DeviceGrid blocks (walker-invariant Y/X/PhaseID shared,
  per-walker GM pointers, zero per-step block re-upload);
- walker-major point-list tiling with per-walker parameter rows;
- a vectorized max-driving-force estimate over a phase's grid sample.
"""
import numpy as np

from pycalphad.gpu.point_solver import PointList, get_point_solver


class EnsemblePointBatcher:
    """System context + cached machinery for walker-batched point solving.

    Subclasses (e.g. ESPEI's BatchedZPFCalculator) plan their own condition
    points, then call :meth:`finalize_domain` once with the statevar domain
    the points span; per parameter-matrix evaluation they use the
    walker-grid / block / tiling methods below and launch through
    ``self.solver`` (a :class:`pycalphad.gpu.point_solver.PointBatchSolver`).
    """

    def __init__(self, database, species, phases, models, phase_record_factory,
                 param_names, backend='cpp', walker_chunk=64, pdens=60,
                 verbose=False, robust=True):
        self.dbf = database
        self.species = species
        self.phases = list(phases)
        self.models = dict(models)
        self.prf = phase_record_factory
        self.param_names = [str(p) for p in param_names]
        self.backend = backend
        self.walker_chunk = int(walker_chunk)
        self.pdens = pdens
        self.verbose = verbose
        self.nonvacant = sorted(
            {el.upper() for sp in species for el in sp.constituents} - {'VA'})
        self.ncomp = len(self.nonvacant)
        self.state_variables = phase_record_factory.state_variables
        self.solver = get_point_solver(species, self.phases, self.models,
                                       phase_record_factory, robust=robust,
                                       backend=('cuda' if backend in ('cuda', 'gpu')
                                                else 'cpp'),
                                       verbose=verbose)
        self.dof_of_model = {}
        for ph in self.phases:
            self.dof_of_model[self.solver.name_to_idx[ph]] =                 phase_record_factory[ph].phase_dof
        self._grid_tmpl = None
        self._grid_eval = None
        self._grid_phase_masks = None
        self._unique_T = None
        self._P = 101325.0

    def finalize_domain(self, unique_T, P=101325.0):
        """Fix the statevar domain the condition points span (unique
        temperatures; single pressure) — called once after planning."""
        self._unique_T = np.unique(np.asarray(unique_T, dtype=np.float64))
        self._P = float(P)

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

    def _stacked_grids(self, params_matrix):
        """One grid per walker, stacked along the statevar-combo axis.

        Sampling points are parameter-independent — only GM changes — so
        X/Y/Phase are tiled views of walker 0's arrays."""
        from types import SimpleNamespace
        gm_rows = []
        first = None
        for pvec in params_matrix:
            params_dict = dict(zip(self.param_names, pvec))
            pv = getattr(self.prf, 'param_values', None)
            if pv is not None and np.asarray(pv).size:
                np.asarray(pv).reshape(-1)[:] = np.asarray(pvec, dtype=np.float64)
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
                 # shape-tuple form: numpy rejects the '1i4' string form,
                 # which single-phase problems hit via n_ph == 1
                 ('phase_grid_indices_start', 'i4', (n_ph,)),
                 ('phase_grid_indices_stop', 'i4', (n_ph,)),
                 ('num_mappable_phases_in_grid', 'i4')]
        pad = (-np.dtype(dtype).itemsize) % 8
        if pad:
            dtype.append(('_tail_pad', 'u1', (pad,)))
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
