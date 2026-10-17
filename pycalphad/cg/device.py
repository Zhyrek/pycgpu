"""Batched pricing outside Cython: cg_price.h (a port of CPhase.price) with the system's energy functions from the
GPU code generator, compiled once per system. The host build (a C++ shared library, called through ctypes'
function address from TileSolver without the GIL) is the reference for the GPU builds of the same source.

Only phases cg_price.h supports are priced there (linear formula moles, no constraints beyond the sublattice
balances, no pinned species); TileSolver prices the others itself.
"""
import ctypes
import hashlib
import os
import re
import threading
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
_LIBRARIES = {}   # source hash -> loaded library

# phase description layout (cg_price.h: CGI_*, CGD_*)
_CGI = ['D', 'nb', 'model', 'nsym', 'nder', 'doff', 'boff', 'bidx', 'blk', 'single', 'pinned', 'sym_a', 'sym_b',
        'sym_off', 'A', 'asum']
_CGD = ['m_lo', 'pen_w', 'max_dz']


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
        ok[p] = 1
        row = {'D': D, 'nb': d['nb'], 'model': index[name], 'nsym': d['nsym'], 'nder': nder, 'doff': nder - D}
        for key in ('boff', 'bidx', 'blk', 'single', 'pinned', 'sym_a', 'sym_b', 'sym_off'):
            row[key] = len(ip); ip.extend(np.asarray(d[key], dtype=np.int64).ravel().tolist() or [0])
        for key in ('A', 'asum'):
            row[key] = len(dp); dp.extend(np.asarray(d[key], dtype=float).ravel().tolist())
        di[p] = [row[k] for k in _CGI]
        dd[p] = [d[k] for k in _CGD]
    return (ok, di.ravel(), dd.ravel(), np.array(ip or [0], dtype=np.int32), np.array(dp or [0.0]))


def _source(model_code, index, names, Dmax, C, ns, nmax):
    disp = ['CG_DEVICE void cg_eval(int model, int hess, double* eg, double* H, const double* x) {',
            '    switch (model) {']
    for name in names:
        k = index[name]
        disp.append(f'    case {k}: if (hess) pycgpu_model_{k}_fused(eg, H, x); else {{ '
                    f'eg[0] = pycgpu_model_{k}_formulaobj(x); pycgpu_model_{k}_formulagrad(&eg[1], x); }} return;')
    disp += ['    }', '}']
    header = (_HERE / 'cg_price.h').read_text()
    host = r'''
extern "C" void cg_price_batch(int n, const int* jobs, const int* job_phase, const int* job_cond, const double* mu,
                               const double* y0, int ystride, double* y_out, double* x_out, double* G_out,
                               double* tpd_out, int* conv, int* evals, const int* di, const double* dd, const int* ip,
                               const double* dp, int C, double RT, const double* sv) {
    static thread_local CgWork W;
    for (int k = 0; k < n; ++k)
        cg_price_job(jobs[k], job_phase, job_cond, mu, y0, ystride, y_out, x_out, G_out, tpd_out, conv, evals, di, dd,
                     ip, dp, C, RT, sv, W);
}
'''
    compat = (_HERE.parent / 'gpu' / 'cpu_compat.h').read_text()
    defines = (f'#define CG_DEVICE\n#define CG_DMAX {Dmax}\n#define CG_CMAX {C}\n#define CG_NS {ns}\n'
               f'#define CG_NMAX {nmax}\n')
    return '\n'.join([compat, '#include <cmath>', model_code, defines, '\n'.join(disp), header, host])


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


class HostPricer:
    """The compiled pricer for one system and its phase descriptions (rebuilt per state, as the descriptions
    carry RT and pinned species)."""

    def __init__(self, wks, phases, elements):
        self.names = [ph.name for ph in phases]
        cores = [ph.core for ph in phases]
        cand = [n for n, c in zip(self.names, cores) if c.device_description()['supported']]
        model_code, self.index, self.deriv = _model_code(wks, cand)
        Dmax = max([ph.D for ph in phases if ph.name in cand] or [1])
        nmax = max([len(v) for v in self.deriv.values()] or [1])
        ns = len(wks.phase_record_factory.state_variables)
        self.lib = _compile(_source(model_code, self.index, cand, Dmax, len(elements), ns, nmax))
        self.address = ctypes.cast(self.lib.cg_price_batch, ctypes.c_void_p).value
        self.elements = elements

    def attach(self, solver, cores, sv_values):
        """Price solver's supported jobs here, at the state of cores (RT, pinned species) and state variables
        sv_values (in the phase records' order)."""
        ok, di, dd, ip, dp = describe(cores, self.names, self.index, self.deriv, self.elements)
        solver.set_pricer(self.address, ok, di, dd, ip, dp, np.asarray(sv_values, dtype=float), cores[0].RT,
                          keep=self)
        return int(ok.sum())
