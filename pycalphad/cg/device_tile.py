"""A whole cg tile on a GPU: cg_tile_device.h's kernels over the tile's arrays in device memory, driven round by
round as cg_tile_host.h's driver does on the host (the reference: same steps, same results to rounding).

The arrays are CuPy arrays and the kernels a CuPy RawModule (CudaPricer.solve_tile); or, to check the driver and
the kernels without a GPU, NumPy arrays and the same kernel source compiled for the host, each launch running its
threads in turn (HostPricer with target 'cuda-on-host').

Per round: the LP (a block of threads per active condition, which split its scans over the store), the polish (one
thread per condition), then sub-rounds of job listing (count, prefix sum, fill), pricing (one thread per job,
sorted by phase), consumption and the next polish, until no jobs are listed; then the merge of the conditions' own
points into the store. The host sees only counts between kernels (to size the job buffers and the store) and the
done flags.
"""
import ctypes
import time

import numpy as np

# CgTile's fields as the driver binds them (cg_dev_bind is generated from these): ints, doubles, arrays
_INTS = ('B', 'C', 'K', 'Dm', 'nph', 'ngauge', 'max_late', 'use_polish', 'nshared', 'cap', 'nj', 'ncp', 'jcap',
         'cpcap')
_DOUBLES = ('RT', 'prune', 'prune_late', 'skip_margin', 'tol', 'seed_prune')
_ARRAYS = (
    ('di', 'const int'), ('dd', 'const double'), ('ip', 'const int'), ('dp', 'const double'), ('sv', 'const double'),
    ('gauge', 'const double'),
    ('Py', 'double'), ('Px', 'double'), ('PG', 'double'), ('Pph', 'int'), ('Pnext', 'int'), ('head', 'int'),
    ('Pcode', 'long long'),
    ('Oy', 'double'), ('Ox', 'double'), ('OG', 'double'), ('Oph', 'int'), ('Ocode', 'long long'), ('nown', 'int'),
    ('own_off', 'const long'),
    ('R', 'const double'), ('b', 'const double'), ('mf', 'const double'), ('fixed', 'const int'),
    ('seedY', 'const double'), ('seed_off', 'const int'),
    ('basis', 'int'), ('xB', 'double'), ('mu', 'double'), ('has_last', 'unsigned char'), ('nopol', 'unsigned char'),
    ('polished', 'unsigned char'), ('skipped', 'unsigned char'), ('added', 'unsigned char'),
    ('done', 'unsigned char'), ('last_tpd', 'double'), ('last_mu', 'double'),
    ('pol_y', 'double'), ('pol_x', 'double'), ('pol_n', 'double'), ('pol_mu', 'double'), ('pol_G', 'double'),
    ('pol_ph', 'int'), ('pol_na', 'int'),
    ('cert_kind', 'int'), ('cert_phase', 'int'), ('aug', 'int'), ('kind', 'int'), ('cert_tpd', 'double'),
    ('ycert', 'double'), ('lp_iters', 'long long'), ('priced', 'long long'),
    ('j_cond', 'int'), ('j_phase', 'int'), ('j_keep', 'int'), ('j_conv', 'int'), ('j_y0', 'double'),
    ('j_y', 'double'), ('j_x', 'double'), ('j_G', 'double'), ('j_tpd', 'double'),
    ('cp_phase', 'int'), ('cp_start', 'int'), ('cp_end', 'int'), ('cp_first', 'int'), ('cp_last', 'int'),
    ('status', 'int'), ('nout', 'int'), ('phs', 'int'), ('amt', 'double'), ('X', 'double'), ('Y', 'double'),
    ('G', 'double'), ('mu_out', 'double'))
_STORE = ('Py', 'Px', 'PG', 'Pph', 'Pnext', 'Pcode')
_JOBS = ('j_cond', 'j_phase', 'j_keep', 'j_conv', 'j_y0', 'j_y', 'j_x', 'j_G', 'j_tpd')
# cg_dev_step's steps (cg_tile_device.h)
CG_STEP_PRICE, CG_STEP_POLISH, CG_STEP_REPOLISH, CG_STEP_JOBS = range(4)


def bind_source():
    """cg_dev_bind: the CgTile at T from the arrays' addresses p (in _ARRAYS' order) and the scalars iv, dv."""
    lines = ['extern "C" __global__ void cg_dev_bind(int n, CgTile* T, const long long* p, const int* iv, '
             'const double* dv) {']
    lines += [f'    T->{f} = ({ct}*)p[{k}];' for k, (f, ct) in enumerate(_ARRAYS)]
    lines += [f'    T->{f} = iv[{k}];' for k, f in enumerate(_INTS)]
    lines += [f'    T->{f} = dv[{k}];' for k, f in enumerate(_DOUBLES)]
    return '\n'.join(lines + ['}']) + '\n'


class _HostLaunch:
    """Kernels of the host build (target 'cuda-on-host'): NAME_launch runs the threads in turn."""
    xp = np

    def __init__(self, lib):
        self.lib = lib

    def __call__(self, name, n, *args):
        conv = [ctypes.c_int(int(n))]
        for a in args:
            if isinstance(a, np.ndarray):
                conv.append(ctypes.c_void_p(a.ctypes.data))
            elif isinstance(a, (float, np.floating)):
                conv.append(ctypes.c_double(float(a)))
            else:
                conv.append(ctypes.c_int(int(a)))
        getattr(self.lib, name + '_launch')(*conv)

    @staticmethod
    def address(a):
        return a.ctypes.data

    @staticmethod
    def get(a):
        return np.array(a)

    def sync(self):
        pass


# kernels with a block of lanes per item (cg_tile_device.h): their first argument is the number of blocks
_LANE_KERNELS = ('cg_dev_lp',)


class _CudaLaunch:
    """Kernels of a CuPy RawModule, BLOCK threads per block (_LANE_KERNELS: one block of LANES threads per item)."""

    def __init__(self, module, block, lanes):
        import cupy as cp
        self.xp = cp
        self.module, self.block, self.lanes, self.functions = module, block, lanes, {}

    def __call__(self, name, n, *args):
        f = self.functions.get(name)
        if f is None:
            f = self.functions[name] = self.module.get_function(name)
        conv = [np.float64(a) if isinstance(a, (float, np.floating)) else
                a if isinstance(a, self.xp.ndarray) else np.int32(a) for a in args]
        n = int(n)
        if n > 0 and name in _LANE_KERNELS:
            f((n,), (self.lanes,), (np.int32(n), *conv))
        elif n > 0:
            f(((n + self.block - 1) // self.block,), (self.block,), (np.int32(n), *conv))

    @staticmethod
    def address(a):
        return a.data.ptr

    @staticmethod
    def get(a):
        return a.get()

    def sync(self):
        self.xp.cuda.Stream.null.synchronize()


class DeviceTile:
    """The driver; times (seconds per step: lp, polish, jobs, price, consume, repolish, merge, and the transfers) and
    counters (as cg_tile_solve_host's) of the last solve."""

    def __init__(self, launch):
        self.launch = launch
        self.xp = launch.xp
        out = self.xp.zeros(3, dtype=np.int64)
        launch('cg_dev_sizes', 1, out)
        self.sizes = launch.get(out).tolist()   # CgTile, CgWork, CgPolishWork (bytes)

    @staticmethod
    def on_host(pricer):
        if getattr(pricer, '_device_tile', None) is None:
            pricer._device_tile = DeviceTile(_HostLaunch(pricer.lib))
        return pricer._device_tile

    # ---- arrays

    def _bind(self):
        L, xp, a = self.launch, self.xp, self.a
        p = xp.asarray(np.array([L.address(a[f]) for f, _ in _ARRAYS], dtype=np.int64))
        iv = xp.asarray(np.array([self.s[f] for f in _INTS], dtype=np.int32))
        dv = xp.asarray(np.array([self.s[f] for f in _DOUBLES], dtype=float))
        L('cg_dev_bind', 1, self.T, p, iv, dv)
        self._keep = (p, iv, dv)

    def _grow(self, names, keep, rows):
        """The arrays names to rows rows (first dimension), their first keep rows kept."""
        xp = self.xp
        for f in names:
            old = self.a[f]
            new = xp.zeros((rows,) + old.shape[1:], dtype=old.dtype)
            new[:keep] = old[:keep]
            self.a[f] = new

    def _store_capacity(self, need):
        s = self.s
        if need <= s['cap']:
            return
        cap = max(need, 2 * s['cap'])
        self._grow(_STORE, s['nshared'], cap)
        s['cap'] = cap
        self._bind()

    def _job_capacity(self, need):
        s = self.s
        if need <= s['jcap']:
            return
        cap = max(need, 2 * s['jcap'])
        self._grow(_JOBS, 0, cap)
        s['jcap'] = cap
        self._bind()

    def _setup(self, t):
        xp = self.xp
        B, C, K, Dm, nph = t.B, t.C, t.K, t.Dm, t.nph
        s = self.s = {f: getattr(t, f) for f in _INTS[:8] + _DOUBLES}
        s.update(nshared=t.npts, cap=max(2 * t.npts, 1024), nj=0, ncp=0, jcap=1024, cpcap=B * nph)
        z = xp.zeros
        a = self.a = {f: xp.asarray(getattr(t, f)) for f in
                      ('di', 'dd', 'ip', 'dp', 'sv', 'gauge', 'R', 'b', 'mf', 'fixed', 'seedY', 'seed_off', 'head')}
        cap, n = s['cap'], t.npts
        for f, shape, dt in (('Py', (cap, Dm), float), ('Px', (cap, C), float), ('PG', (cap,), float),
                             ('Pph', (cap,), np.int32), ('Pnext', (cap,), np.int32), ('Pcode', (cap,), np.int64)):
            a[f] = z(shape, dtype=dt)
            if n:
                a[f][:n] = xp.asarray(getattr(t, f))
        self.own_off = np.concatenate([[0], np.cumsum([t.own_first] + [t.own_other] * (B - 1))]).astype(np.int64)
        rows = int(self.own_off[-1])
        a.update(Oy=z((rows, Dm)), Ox=z((rows, C)), OG=z(rows), Oph=z(rows, dtype=np.int32),
                 Ocode=z(rows, dtype=np.int64), nown=z(B, dtype=np.int32), own_off=xp.asarray(self.own_off))
        for f, shape, dt in (
                ('basis', (B, K), np.int32), ('xB', (B, K), float), ('mu', (B, C), float),
                ('has_last', (B, nph), np.uint8), ('nopol', B, np.uint8), ('polished', B, np.uint8),
                ('skipped', B, np.uint8), ('added', B, np.uint8), ('done', B, np.uint8),
                ('last_tpd', (B, nph), float), ('last_mu', (B, nph, C), float), ('pol_y', (B, C, Dm), float),
                ('pol_x', (B, C, C), float), ('pol_n', (B, C), float), ('pol_mu', (B, C), float),
                ('pol_G', (B, C), float), ('pol_ph', (B, C), np.int32), ('pol_na', B, np.int32),
                ('cert_kind', B, np.int32), ('cert_phase', B, np.int32), ('aug', B, np.int32),
                ('kind', B, np.int32), ('cert_tpd', B, float), ('ycert', (B, Dm), float),
                ('lp_iters', B, np.int64), ('priced', B, np.int64),
                ('cp_phase', B * nph, np.int32), ('cp_start', B * nph, np.int32), ('cp_end', B * nph, np.int32),
                ('cp_first', B, np.int32), ('cp_last', B, np.int32),
                ('status', B, np.int32), ('nout', B, np.int32), ('amt', (B, C), float), ('X', (B, C, C), float),
                ('G', (B, C), float), ('mu_out', (B, C), float)):
            a[f] = z(shape, dtype=dt)
        a['phs'] = xp.full((B, C), -1, dtype=np.int32)
        a['Y'] = xp.full((B, C, Dm), np.nan)
        for f, shape, dt in (('j_cond', 1024, np.int32), ('j_phase', 1024, np.int32), ('j_keep', 1024, np.int32),
                             ('j_conv', 1024, np.int32), ('j_y0', (1024, Dm), float), ('j_y', (1024, Dm), float),
                             ('j_x', (1024, C), float), ('j_G', 1024, float), ('j_tpd', 1024, float)):
            a[f] = z(shape, dtype=dt)
        # the first condition starts from the solver's state
        a['basis'][0] = xp.asarray(t.basis0)
        a['has_last'][0] = xp.asarray(t.has_last0)
        a['last_tpd'][0] = xp.asarray(t.last_tpd0)
        a['last_mu'][0] = xp.asarray(t.last_mu0)
        self.work = z(B * self.sizes[1], dtype=np.uint8)
        self.pwork = z(B * self.sizes[2], dtype=np.uint8)
        self.T = z(self.sizes[0], dtype=np.uint8)
        self._bind()

    # ---- steps

    def _step(self, label, step, n, items, rounds=0, max_rounds=0, use_seeds=0, fill=0, start=None, count=None,
              want=None):
        """cg_dev_step on n items (arrays not given: the CgTile's bytes, unused)."""
        T = self.T
        self._timed(label, 'cg_dev_step', n, step, T, items, rounds, max_rounds, use_seeds, fill,
                    T if start is None else start, T if count is None else count, T if want is None else want,
                    self.work, self.pwork)

    def _timed(self, step, name, n, *args):
        t0 = time.perf_counter()
        self.launch(name, n, *args)
        self.launch.sync()
        self.times[step] = self.times.get(step, 0.0) + time.perf_counter() - t0

    def _get(self, a):
        t0 = time.perf_counter()
        out = self.launch.get(a)
        self.times['transfers'] = self.times.get('transfers', 0.0) + time.perf_counter() - t0
        return out

    def _list_jobs(self, conds, want, rounds, use_seeds):
        """The jobs of the conditions that want them, in ranges in the order of conds: their number."""
        xp, n = self.xp, len(conds)
        count = xp.zeros(n, dtype=np.int32)
        self._step('jobs', CG_STEP_JOBS, n, conds, rounds, use_seeds=use_seeds, fill=0, count=count, want=want)
        c = self._get(count).astype(np.int64)
        total = int(c.sum())
        if total == 0:
            return 0
        self._job_capacity(total)
        start = xp.asarray((np.cumsum(c) - c).astype(np.int32))
        self._step('jobs', CG_STEP_JOBS, n, conds, rounds, use_seeds=use_seeds, fill=1, start=start, count=count,
                   want=want)
        return total

    def _price(self, nj):
        xp, a = self.xp, self.a
        t0 = time.perf_counter()
        key = a['j_phase'][:nj].astype(np.int64) * nj + xp.arange(nj, dtype=np.int64)
        order = xp.argsort(key).astype(np.int32)   # (each warp runs one phase's code)
        self.launch.sync()
        self.times['sort'] = self.times.get('sort', 0.0) + time.perf_counter() - t0
        self._step('price', CG_STEP_PRICE, nj, order)

    def _merge(self, members):
        """The members' own points into the store (cg_tile_merge's result); whether a member's slab was full."""
        xp, a, s = self.xp, self.a, self.s
        t0 = time.perf_counter()
        nown = self._get(a['nown'])[members].astype(np.int64)
        full = int((nown >= self.own_off[members + 1] - self.own_off[members]).sum())
        npts = int(nown.sum())
        if npts:
            self._store_capacity(s['nshared'] + npts)
            rows_h = np.repeat(self.own_off[members], nown) + (np.arange(npts) - np.repeat(np.cumsum(nown) - nown, nown))
            rows = xp.asarray(rows_h.astype(np.int64))
            ph = a['Oph'][rows].astype(np.int64)
            order = xp.argsort(ph * npts + xp.arange(npts, dtype=np.int64)).astype(np.int32)
            ph_sorted = ph[order]
            group = xp.searchsorted(ph_sorted, ph_sorted, side='left').astype(np.int32)
            pos = xp.empty(npts, dtype=np.int32); pos[order] = xp.arange(npts, dtype=np.int32)
            first = xp.empty(npts, dtype=np.int32); dest = xp.empty(npts, dtype=np.int32)
            nsh = xp.zeros(1, dtype=np.int32)
            self.launch('cg_dev_merge_mark', npts, self.T, rows, order, group, first)
            self.launch('cg_dev_merge_resolve', 1, self.T, npts, rows, order, pos, first, dest, nsh)
            self.launch('cg_dev_merge_copy', npts, self.T, rows, dest)
            s['nshared'] = int(self.launch.get(nsh)[0])
        a['nown'][xp.asarray(members)] = 0
        self.launch.sync()
        self.times['merge'] = self.times.get('merge', 0.0) + time.perf_counter() - t0
        return full

    def solve(self, t, max_rounds=60):
        """The tile of tile_inputs t: the arrays of Solver.solve_sequence."""
        xp = self.xp
        self.times = {}
        t_start = time.perf_counter()
        self._setup(t)
        self.times['setup'] = time.perf_counter() - t_start
        a, T = self.a, self.T
        rounds_total = subrounds = jobs_total = own_full = 0
        for st in range(int(t.stage.max()) + 1):
            members = np.nonzero(t.stage == st)[0]
            if not len(members):
                continue
            d_members = xp.asarray(members.astype(np.int32))
            if st > 0:
                self._timed('stage_init', 'cg_dev_stage_init', len(members), T, d_members, xp.asarray(t.init_from))
            for rounds in range(1, max_rounds + 1):
                rounds_total += 1
                act = members[self._get(a['done'])[members] == 0].astype(np.int32)
                if len(act):
                    d_act = xp.asarray(act)
                    want = xp.zeros(len(act), dtype=np.int32)
                    self._timed('lp', 'cg_dev_lp', len(act), T, d_act, want)
                    self._step('polish', CG_STEP_POLISH, len(act), d_act, want=want)
                    nj = self._list_jobs(d_act, want, rounds, int(t.use_seeds and st == 0 and rounds == 1))
                    while nj > 0:
                        subrounds += 1; jobs_total += nj
                        self._price(nj)
                        self._timed('consume', 'cg_dev_consume', len(act), T, d_act, rounds, max_rounds)
                        self._step('repolish', CG_STEP_REPOLISH, len(act), d_act, rounds, max_rounds, want=want)
                        nj = self._list_jobs(d_act, want, rounds, 0)
                own_full += self._merge(members)
                if not len(act) or self._get(a['done'])[members].all():
                    break
        out = tuple(self._get(a[f]) for f in ('status', 'nout', 'phs', 'amt', 'X', 'Y', 'G', 'mu_out'))
        self.counters = dict(priced=int(self._get(a['priced']).sum()), lp_iters=int(self._get(a['lp_iters']).sum()),
                             rounds=rounds_total, subrounds=subrounds, jobs=jobs_total, own_full=own_full,
                             store=self.s['nshared'])
        self.times['total'] = time.perf_counter() - t_start
        self.a = None; self.work = self.pwork = None   # (the device memory back)
        return out
