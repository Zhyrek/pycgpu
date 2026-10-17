"""Batched pricing outside Cython: cg_price.h (a port of CPhase.price) with the system's energy functions from the
GPU code generator, compiled once per system.

- HostPricer: a C++ shared library on this machine, called through its function address from TileSolver without
  the GIL (the reference for the GPU builds of the same source).
- CudaPricer: the same source as a CUDA kernel (or HIP, with a ROCm build of CuPy), one thread per job; TileSolver
  hands it each sub-round's jobs through a Python callback.

Only phases cg_price.h supports are priced there (linear formula moles, no constraints beyond the sublattice
balances, no pinned species); TileSolver prices the others itself.
"""
import ctypes
import hashlib
import os
import re
import threading
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
_LIBRARIES = {}   # source hash -> loaded library

# phase description layout (cg_price.h: CGI_*, CGD_*)
_CGI = ['D', 'nb', 'model', 'nsym', 'nder', 'doff', 'boff', 'bidx', 'blk', 'single', 'pinned', 'sym_a', 'sym_b',
        'sym_off', 'A', 'asum', 'elem_of', 'stoich']
_CGD = ['m_lo', 'pen_w', 'max_dz']

_ENTRY_ARGS = ('int n, const int* jobs, const int* job_phase, const int* job_cond, const double* mu, '
               'const double* y0, int ystride, double* y_out, double* x_out, double* G_out, double* tpd_out, '
               'int* conv, int* evals, const int* di, const double* dd, const int* ip, const double* dp, int C, '
               'double RT, const double* sv')
_PASS_ARGS = ('job_phase, job_cond, mu, y0, ystride, y_out, x_out, G_out, tpd_out, conv, evals, di, dd, ip, dp, C, '
              'RT, sv')

_HOST_ENTRY = (
    'extern "C" void cg_price_batch(' + _ENTRY_ARGS + ') {\n'
    '    static thread_local CgWork W;\n'
    '    for (int k = 0; k < n; ++k) cg_price_job(jobs[k], ' + _PASS_ARGS + ', W);\n'
    '}\n')


def _kernels(classes):
    """One kernel per size class (D site fractions, N derivative variables): the work arrays fit the class. (Also
    cg_price_kernel, sized for the largest phase: for comparisons.)"""
    out = ['extern "C" __global__ void cg_price_kernel(' + _ENTRY_ARGS + ') {\n'
           '    int k = blockIdx.x * blockDim.x + threadIdx.x;\n'
           '    if (k >= n) return;\n'
           '    CgWork W;\n'
           '    cg_price_job(jobs[k], ' + _PASS_ARGS + ', W);\n'
           '}\n']
    for D, N in classes:
        out.append(
            f'extern "C" __global__ void cg_price_kernel_{D}_{N}(' + _ENTRY_ARGS + ') {\n'
            '    int k = blockIdx.x * blockDim.x + threadIdx.x;\n'
            '    if (k >= n) return;\n'
            f'    CgWorkT<{D}, {N}> W;\n'
            '    cg_price_job(jobs[k], ' + _PASS_ARGS + ', W);\n'
            '}\n')
    return '\n'.join(out)


def _kernels_on_host(classes):
    """The class kernels run on the host (cpu_compat.h makes the thread indices variables), each job by its
    phase's class: checks their source and the size classes without a GPU."""
    body = ['extern "C" void cg_price_kernel_on_host(' + _ENTRY_ARGS + ') {',
            '    blockDim.x = 1; blockIdx.x = 0; threadIdx.x = 0;',
            '    for (int k = 0; k < n; ++k) {',
            '        const int* d = di + job_phase[jobs[k]] * CGI_COUNT;']
    for D, N in classes:
        body.append(f'        if (d[CGI_D] <= {D} && d[CGI_NDER] <= {N}) {{ '
                    f'cg_price_kernel_{D}_{N}(1, jobs + k, ' + _PASS_ARGS + '); continue; }')
    body += ['    }', '}']
    return '\n'.join(body) + '\n'


_RTC_PRELUDE = '''#if defined(__CUDACC_RTC__) || defined(__HIPCC_RTC__)
#ifndef NAN
#define NAN __longlong_as_double(0x7ff8000000000000LL)
#endif
#ifndef INFINITY
#define INFINITY __longlong_as_double(0x7ff0000000000000LL)
#endif
#else
#include <math.h>
#endif
'''


def _split_functions(source):
    """Generated model source -> {function name: its text} (each function starts a line with __device__)."""
    out = {}
    for chunk in re.split(r'(?m)^(?=__device__ )', source):
        m = re.match(r'__device__ \w+ (\w+)\(', chunk)
        if m:
            out[m.group(1)] = chunk
    return out


def _model_code(wks, names):
    """Energy, gradient and fused energy/gradient/Hessian functions of the named phases, the model index of each,
    and each model's derivative variables (state variables the energy depends on, then site fractions)."""
    from pycalphad.gpu.gpu_equilibrium import _generate_c_code_for_phase_models
    from pycalphad.gpu.gpu_codegen import get_ordered_symbols_for_diff
    funcs, _, _, index = _generate_c_code_for_phase_models(wks, include_hess=True, validate=False)
    fn = _split_functions(funcs)
    keep, deriv = [], {}
    for name in names:
        k = index[name]
        for suffix in ('formulaobj', 'formulagrad', 'fused'):
            keep.append(fn[f'pycgpu_model_{k}_{suffix}'])
        deriv[name] = [str(s) for s in get_ordered_symbols_for_diff(wks.models[name], wks)]
    return '\n'.join(keep), index, deriv


def describe(cores, names, index, deriv, elements):
    """Phase descriptions for cg_price.h (int and double tables and pools) and which phases it prices."""
    di = np.zeros((len(cores), len(_CGI)), dtype=np.int32)
    dd = np.zeros((len(cores), len(_CGD)))
    ip, dp, ok = [], [], np.zeros(len(cores), dtype=np.uint8)
    for p, (core, name) in enumerate(zip(cores, names)):
        d = core.device_description()
        if not d['supported'] or name not in deriv:
            continue
        D = d['D']
        nder = len(deriv[name])
        # the derivative variables end with the site fractions, in the phase record's order
        if nder < D or any('Y(' not in s for s in deriv[name][nder - D:]):
            continue
        ok[p] = 0 if d['stoich'] else 1   # (stoichiometric phases: described for the polish, never priced)
        row = {'D': D, 'nb': d['nb'], 'model': index[name], 'nsym': d['nsym'], 'nder': nder, 'doff': nder - D,
               'stoich': int(d['stoich'])}
        for key in ('boff', 'bidx', 'blk', 'single', 'pinned', 'sym_a', 'sym_b', 'sym_off', 'elem_of'):
            row[key] = len(ip); ip.extend(np.asarray(d[key], dtype=np.int64).ravel().tolist() or [0])
        for key in ('A', 'asum'):
            row[key] = len(dp); dp.extend(np.asarray(d[key], dtype=float).ravel().tolist())
        di[p] = [row[k] for k in _CGI]
        dd[p] = [d[k] for k in _CGD]
    return (ok, di.ravel(), dd.ravel(), np.array(ip or [0], dtype=np.int32), np.array(dp or [0.0]))


def _source(model_code, index, names, Dmax, C, ns, nmax, target='host', classes=()):
    """The pricing source: 'host' (a C++ library), 'cuda' (a CUDA/HIP kernel), 'cuda-on-host' (that kernel as C++,
    called in a loop over threads)."""
    disp = ['CG_DEVICE void cg_eval(int model, int hess, double* eg, double* H, const double* x) {',
            '    switch (model) {']
    for name in names:
        k = index[name]
        disp.append(f'    case {k}: if (hess) pycgpu_model_{k}_fused(eg, H, x); else {{ '
                    f'eg[0] = pycgpu_model_{k}_formulaobj(x); pycgpu_model_{k}_formulagrad(&eg[1], x); }} return;')
    disp += ['    }', '}']
    header = (_HERE / 'cg_price.h').read_text() + '\n' + (_HERE / 'cg_round.h').read_text()
    sizes = f'#define CG_DMAX {Dmax}\n#define CG_CMAX {C}\n#define CG_NS {ns}\n#define CG_NMAX {nmax}\n'
    if target == 'cuda':
        return '\n'.join([_RTC_PRELUDE, model_code, '#define CG_DEVICE __device__\n' + sizes, '\n'.join(disp),
                          header, _kernels(classes)])
    compat = (_HERE.parent / 'gpu' / 'cpu_compat.h').read_text()
    parts = [compat, '#include <cmath>', model_code, '#define CG_DEVICE\n' + sizes, '\n'.join(disp), header]
    if target == 'host':
        parts += [_HOST_ENTRY, (_HERE / 'cg_tile.h').read_text(), (_HERE / 'cg_tile_host.h').read_text()]
    else:
        parts += [_kernels(classes), _kernels_on_host(classes)]
    return '\n'.join(parts)


def _cache_dir():
    d = Path(os.environ.get('PYCALPHAD_CG_CACHE', Path.home() / '.cache' / 'pycalphad-cg'))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _compile(source):
    from pycalphad.gpu.cpu_backend import _compile_command
    import subprocess
    key = hashlib.sha256(source.encode()).hexdigest()[:24]
    if key in _LIBRARIES:
        return _LIBRARIES[key]
    d = _cache_dir()
    lib_path = d / f'cg_price_{key}.so'
    if not lib_path.exists():
        src_path = d / f'cg_price_{key}.cpp'
        src_path.write_text(source)
        tmp = d / f'cg_price_{key}.{os.getpid()}.{threading.get_ident()}.so'   # (other processes may build it too)
        r = subprocess.run(_compile_command(str(tmp), str(src_path), []), capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f'compiling the cg pricing library failed:\n{r.stderr[-4000:]}')
        os.replace(tmp, lib_path)
    lib = ctypes.CDLL(str(lib_path))
    _LIBRARIES[key] = lib
    return lib


class _Pricer:
    def _setup(self, wks, phases, elements):
        self.names = [ph.name for ph in phases]
        cores = [ph.core for ph in phases]
        self.cand = [n for n, c in zip(self.names, cores) if c.device_description()['supported']]
        self.model_code, self.index, self.deriv = _model_code(wks, self.cand)
        self.Dmax = max([ph.D for ph in phases if ph.name in self.cand] or [1])
        self.nmax = max([len(v) for v in self.deriv.values()] or [1])
        self.ns = len(wks.phase_record_factory.state_variables)
        self.elements = elements
        # size classes (site fractions, derivative variables), smallest first; each phase in the first that fits
        D_of = {ph.name: ph.D for ph in phases}
        self.classes = sorted({(D_of[n], len(self.deriv[n])) for n in self.cand})
        self.class_of = np.full(len(phases), -1, dtype=np.int64)
        for p, name in enumerate(self.names):
            if name in self.cand:
                self.class_of[p] = self.classes.index((D_of[name], len(self.deriv[name])))

    def source(self, target):
        return _source(self.model_code, self.index, self.cand, self.Dmax, len(self.elements), self.ns, self.nmax,
                       target, self.classes)

    def descriptions(self, cores):
        return describe(cores, self.names, self.index, self.deriv, self.elements)


class HostPricer(_Pricer):
    """The compiled pricer for one system on the host. target 'cuda-on-host' builds the GPU kernel's source for
    the host instead (to check it without a GPU; same arguments)."""

    def __init__(self, wks, phases, elements, target='host'):
        self._setup(wks, phases, elements)
        self.lib = _compile(self.source(target))
        self.function = self.lib.cg_price_batch if target == 'host' else self.lib.cg_price_kernel_on_host
        self.address = ctypes.cast(self.function, ctypes.c_void_p).value

    def attach(self, solver, cores, sv_values):
        """Price solver's supported jobs here, at the state of cores (RT, pinned species; the descriptions carry
        them) and state variables sv_values (in the phase records' order)."""
        ok, di, dd, ip, dp = self.descriptions(cores)
        solver.set_pricer(self.address, ok, di, dd, ip, dp, np.asarray(sv_values, dtype=float), cores[0].RT,
                          keep=self)
        return int(ok.sum())


class CudaPricer(_Pricer):
    """The pricer as a CUDA kernel through CuPy (HIP with a ROCm CuPy): one thread per job; each sub-round's jobs
    are compacted, copied over, priced and copied back. seconds / jobs / batches: what it did."""

    BLOCK = int(os.environ.get('PYCALPHAD_CG_BLOCK', 64))
    # one launch of the largest-size kernel over the jobs sorted by phase (T4: 2.3x faster than unsorted; the class
    # kernels, one per size class on its own stream, were slower there: its FP64 rate, not memory, is the limit)
    CLASS_KERNELS = bool(int(os.environ.get('PYCALPHAD_CG_CLASS_KERNELS', '0')))
    STACK = int(os.environ.get('PYCALPHAD_CG_STACK', 65536))   # bytes per thread (the work arrays live there)

    def __init__(self, wks, phases, elements):
        import cupy as cp
        from pycalphad.gpu.kernel_manager import cuda_raw_module, ensure_device_stack_limit
        self.cp = cp
        self._setup(wks, phases, elements)
        ensure_device_stack_limit(self.STACK)
        self.module = cuda_raw_module(self.source('cuda'), ('-std=c++14',))
        self.kernels = [self.module.get_function(f'cg_price_kernel_{D}_{N}') for D, N in self.classes]
        self.kernel = self.module.get_function('cg_price_kernel')
        self.seconds = 0.0; self.kernel_seconds = 0.0; self.jobs = 0; self.batches = 0

    def upload(self, cores, sv_values):
        cp = self.cp
        ok, di, dd, ip, dp = self.descriptions(cores)
        self.d_desc = [cp.asarray(a) for a in (di, dd, ip, dp)]
        self.d_sv = cp.asarray(np.asarray(sv_values, dtype=float))
        self.RT = float(cores[0].RT)
        return ok

    def attach(self, solver, cores, sv_values):
        ok = self.upload(cores, sv_values)
        solver.set_pricer_callback(self.price, ok)
        return int(ok.sum())

    def price(self, jobs, job_phase, job_cond, mu, y0, y, x, G, tpd, conv, evals):
        """Price jobs (indices into the job tables) on the device, writing the results into the tables, sorted by
        phase (each warp runs one phase's code); with CLASS_KERNELS one kernel per size class (work arrays of the
        class), each on its own stream."""
        cp = self.cp
        t0 = time.perf_counter()
        jobs = np.asarray(jobs)
        jp_h = job_phase[jobs]
        cls = self.class_of[jp_h]
        order = np.lexsort((jp_h, cls))
        jobs = jobs[order]; jp_h = jp_h[order]; cls = cls[order]
        n = len(jobs)
        C, ys = mu.shape[1], y0.shape[1]
        jp = cp.asarray(jp_h); jc = cp.asarray(job_cond[jobs])
        d_mu = cp.asarray(mu); d_y0 = cp.asarray(y0[jobs])
        d_y = cp.zeros((n, ys)); d_x = cp.zeros((n, C)); d_G = cp.zeros(n); d_t = cp.zeros(n)
        d_c = cp.zeros(n, dtype=cp.int32); d_e = cp.zeros(2 * n, dtype=cp.int32)
        ident = cp.arange(n, dtype=cp.int32)
        di, dd, ip, dp = self.d_desc
        args = (jp, jc, d_mu, d_y0, np.int32(ys), d_y, d_x, d_G, d_t, d_c, d_e, di, dd, ip, dp, np.int32(C),
                np.float64(self.RT), self.d_sv)
        start, stop = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        if self.CLASS_KERNELS:
            bounds = np.searchsorted(cls, np.arange(len(self.classes) + 1))
            streams = [cp.cuda.Stream(non_blocking=True) for _ in self.kernels]
            for c, kernel in enumerate(self.kernels):
                a, b = int(bounds[c]), int(bounds[c + 1])
                if b > a:
                    with streams[c]:
                        kernel(((b - a + self.BLOCK - 1) // self.BLOCK,), (self.BLOCK,),
                               (np.int32(b - a), ident[a:b]) + args)
            for st in streams:
                st.synchronize()
        else:
            self.kernel(((n + self.BLOCK - 1) // self.BLOCK,), (self.BLOCK,), (np.int32(n), ident) + args)
        stop.record(); stop.synchronize()
        self.kernel_seconds += cp.cuda.get_elapsed_time(start, stop) / 1e3
        y[jobs] = d_y.get(); x[jobs] = d_x.get(); G[jobs] = d_G.get(); tpd[jobs] = d_t.get(); conv[jobs] = d_c.get()
        evals.reshape(-1, 2)[jobs] = d_e.get().reshape(n, 2)
        self.seconds += time.perf_counter() - t0; self.jobs += n; self.batches += 1


def _scaled_rows(R, b, mu_fixed, fixed, RT):
    """Condition rows as the solver scales them (Solver._set_conditions_c): each row by its right-hand side (else
    its largest coefficient), b >= 0; fixed potentials in RT units."""
    R = np.array(R, dtype=float); b = np.array(b, dtype=float)
    for k in range(b.shape[-1]):
        sc = np.where(b[..., k] < 0, -1.0, 1.0)
        m = np.abs(b[..., k])
        mr = np.abs(R[..., k, :]).max(axis=-1)
        m = np.where(m > 0, m, np.where(mr > 0, mr, 1.0))
        R[..., k, :] *= (sc / m)[..., None]
        b[..., k] *= sc / m
    return R, b, np.asarray(mu_fixed, dtype=float) / RT, np.asarray(fixed, dtype=np.int32)


def _pack_seeds(seeds, phases, Dm):
    rows, off = [], [0]
    for ph, sd in zip(phases, seeds if seeds is not None else [None] * len(phases)):
        S = np.zeros((0, Dm)) if sd is None else np.asarray(sd, dtype=float).reshape(-1, ph.D)
        P = np.zeros((S.shape[0], Dm)); P[:, :ph.D] = S
        rows.append(P); off.append(off[-1] + S.shape[0])
    return np.ascontiguousarray(np.vstack(rows) if off[-1] else np.zeros((1, Dm))), np.array(off, dtype=np.int32)


def _host_tile_solve(self, solver, cores, sv_values, phases, R, b, mu_fixed, fixed, seeds, stage, init_from,
                     max_rounds=60):
    """A whole tile in compiled code (cg_tile.h's steps, cg_tile_host.h's driver) from the solver's store, basis and
    skip state: the arrays of Solver.solve_sequence, or None if a phase is not supported (the caller solves it)."""
    ok, di, dd, ip, dp = self.descriptions(cores)
    if any(not ok[p] and not ph.stoichiometric for p, ph in enumerate(phases)):
        return None
    if any(ph.stoichiometric and di.reshape(len(phases), -1)[p, 0] == 0 for p, ph in enumerate(phases)):
        return None   # (a stoichiometric phase the compiled code cannot evaluate)
    RT = float(cores[0].RT)
    B = len(b); C = len(self.elements)
    Rs, bs, mfs, fx = _scaled_rows(R, b, mu_fixed, fixed, RT)
    K = bs.shape[1]
    st = solver.export_points()
    Dm = solver.Dmax
    n = st['n']
    Py = np.zeros((max(n, 1), Dm)); Py[:n] = st['Py']
    seedY, seed_off = _pack_seeds(seeds, phases, Dm)
    skip = st['skip']
    has_last = np.array([s[0] for s in skip], dtype=np.uint8)
    last_tpd = np.array([s[1] for s in skip], dtype=float)
    last_mu = np.ascontiguousarray(np.array([s[2] for s in skip], dtype=float).reshape(len(phases), C))
    basis0 = np.ascontiguousarray(np.asarray(st['basis'][:K], dtype=np.int32)) if st['K'] == K else \
        np.array([-(1 + k) for k in range(K)], dtype=np.int32)
    gauge = solver.gauge_matrix().reshape(-1, C)
    status = np.zeros(B, dtype=np.int32); nout = np.zeros(B, dtype=np.int32)
    phs = np.full((B, C), -1, dtype=np.int32); amt = np.zeros((B, C)); X = np.zeros((B, C, C))
    Y = np.full((B, C, Dm), np.nan); G = np.zeros((B, C)); mu = np.zeros((B, C))
    counters = np.zeros(8, dtype=np.int64)
    P = ctypes.c_void_p
    def a(x, dtype):
        x = np.ascontiguousarray(x, dtype=dtype); keep.append(x); return x.ctypes.data_as(P)
    keep = []
    I, D_ = ctypes.c_int, ctypes.c_double
    rc = self.lib.cg_tile_solve_host(
        I(B), I(C), I(K), I(Dm), I(len(phases)), I(gauge.shape[0]), I(solver.max_late), I(int(solver.use_polish)),
        I(max_rounds), D_(RT), D_(solver.prune), D_(solver.prune_late), D_(solver.skip_margin), D_(solver.tol),
        D_(solver.seed_prune if np.isfinite(solver.seed_prune) else 1e300),
        a(di, np.int32), a(dd, float), a(ip, np.int32), a(dp, float), a(sv_values, float),
        a(gauge if gauge.size else np.zeros(C), float),
        I(n), a(Py, float), a(st['Px'] if n else np.zeros((1, C)), float), a(st['PG'] if n else np.zeros(1), float),
        a(st['Pph'] if n else np.zeros(1), np.int32), a(st['Pcode'] if n else np.zeros(1), np.int64),
        a(st['Pnext'] if n else np.zeros(1), np.int32), a(st['head'], np.int32),
        a(Rs, float), a(bs, float), a(mfs, float), a(fx, np.int32),
        a(seedY, float), a(seed_off, np.int32), I(int(seeds is not None)), I(int(seed_off[-1]) + 4096), I(512),
        a(stage, np.int32), a(init_from, np.int32),
        a(basis0, np.int32), a(has_last, np.uint8), a(last_tpd, float), a(last_mu, float),
        status.ctypes.data_as(P), nout.ctypes.data_as(P), phs.ctypes.data_as(P), amt.ctypes.data_as(P),
        X.ctypes.data_as(P), Y.ctypes.data_as(P), G.ctypes.data_as(P), mu.ctypes.data_as(P), counters.ctypes.data_as(P))
    if rc != 0:
        raise RuntimeError(f'cg_tile_solve_host failed ({rc})')
    self.tile_counters = dict(zip(('priced', 'lp_iters', 'rounds', 'subrounds', 'jobs', 'own_full', 'store'),
                                  counters[:7].tolist()))
    return status, nout, phs, amt, X, Y, G, mu


HostPricer.solve_tile = _host_tile_solve
