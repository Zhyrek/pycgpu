# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""
Column generation for phase equilibria.

Master problem: a linear program over stored points (a phase and its site fractions), with the equilibrium
conditions as rows (N, mole and mass fractions, linear combinations; fixed chemical potentials enter the costs).
It is solved by a small dense revised simplex, warm-started from the previous basis; its duals give the chemical
potentials. Pricing: for each phase, minimize the tangent-plane distance TPD(y) = GM(y) - mu.x(y) over its site
fractions, by Gauss-Newton in softmax coordinates. Every minimum below the tangent plane becomes a column. Once the
master problem has identified the phases, a Newton iteration on (mu, phase amounts) finishes the equilibrium
("polish"), and a final pricing pass at its chemical potentials certifies it.

All energies are pycalphad's own compiled PhaseRecord functions (per formula unit, inputs: state variables, site
fractions, parameters).
"""
import numpy as np
from libc.math cimport exp, log, fabs, sqrt, INFINITY, NAN, isfinite
from libc.string cimport memcpy
from libc.stdlib cimport malloc, free
from libcpp.string cimport string
from scipy.linalg.cython_lapack cimport dsyev, dgetrf, dgetrs, dpotrf, dpotrs, dgesv, dgelss
from pycalphad.core.phase_rec cimport PhaseRecord, FastFunction, FastFunctionFactory
from pycalphad.core.constants import MIN_SITE_FRACTION

ctypedef void (*fn_t)(double*, const double*, void*) noexcept nogil

cdef double Y_FLOOR = MIN_SITE_FRACTION
cdef double R_GAS = 8.3145


import threading
_COMPILE_LOCK = threading.Lock()

cdef class CPhase:
    """One phase: compiled energy and formula-mole functions, sublattice structure, linear equality constraints beyond
    the sublattice balances (charge balance, phase-local conditions), and work buffers for pricing.

    The tangent-plane distance per mole of atoms, in RT units, is f(y) = (G(y)/RT - mu.m(y)) / M(y), with G the
    energy and m the moles of each element per formula unit and M = sum(m). For most phases m is linear in y
    (m = A y); ionic liquids' site ratios vary with y, and their m, dm, d2m come from the PhaseRecord."""
    cdef public object name, phase_record
    cdef public int D, C, nb, ns, npar, ne, ne_fixed
    cdef public bint stoich, linear
    cdef public double RT, T
    cdef int[::1] boff, bidx, blk, elem_of, single, free, pinned
    cdef public double[:, ::1] A          # linear phases: moles of element i per formula unit = A[i] . y
    cdef public double[::1] asum          # linear phases: atoms per formula unit = asum . y
    cdef double[::1] nu, mu_cur           # mu^T A (linear phases); the chemical potentials being priced (RT units)
    cdef double[:, ::1] E                 # constraints E y = e (rows: the PhaseRecord's beyond the site balances,
    cdef double[::1] e, lam, hres         #   then phase-local conditions); multipliers; residuals
    cdef object keep, ctor_args
    cdef fn_t p_G, p_grad, p_hess
    cdef void* d_G
    cdef void* d_grad
    cdef void* d_hess
    cdef fn_t* p_m
    cdef fn_t* p_mg
    cdef fn_t* p_mh
    cdef void** d_m
    cdef void** d_mg
    cdef void** d_mh
    cdef double[::1] inp, graw, hraw, gsv, mval, mgraw, mhraw, gM, hM, gm
    cdef double G_fu, Mtot
    cdef public double[::1] out1          # [GM] per mole of atoms at the last eval1
    cdef double[::1] y, z, zt, yt, gz, gzt, gy, gyt, gl, d, Hz, Hy, M, w, work, V, coef, bt, sc, Hs, Bm, Wm, xs
    cdef double[::1] As, WA, Se, tmpe
    cdef int[::1] ipe
    cdef int lwork, _t_index
    cdef public double m_lo, pen_w, max_dz
    cdef public double last_tpd
    cdef double[::1] last_mu
    cdef public bint has_last
    cdef public long evals1, evals2, nprice, nfail, nescape, nsaddle
    cdef bint last_indef                  # the last factor() found the Hessian indefinite
    cdef int[::1] sym_a, sym_b            # species pairs (sym_a[k], sym_b[k]) of sublattice pairs with the same species
    cdef int nsym                         #   (sym_off: where each sublattice pair's species start)
    cdef int[::1] sym_off

    def __init__(self, PhaseRecord pr, int temperature_index, int[::1] sublattice_of, int[::1] elem_of,
                 double[:, ::1] A, bint linear, E, e):
        cdef FastFunctionFactory ff = pr.function_factory
        cdef FastFunction f
        cdef int i
        self.ctor_args = (pr, temperature_index, np.array(sublattice_of, dtype=np.int32), np.array(elem_of, dtype=np.int32),
                          np.array(A, dtype=float), linear, np.array(E, dtype=float), np.array(e, dtype=float))
        self.phase_record = pr
        self.name = pr.phase_name
        self.D = pr.phase_dof; self.C = A.shape[0]
        self.ns = pr.num_statevars; self.npar = pr.parameters.shape[0]
        D = self.D; C = self.C; n = self.ns + D
        # the energy now (screening start configurations needs it); its derivatives when the phase is first priced.
        # Functions are called directly: FastFunction.call is not declared noexcept, which would take the GIL after
        # every call
        fG = <FastFunction>ff.get_func(string(b'G'))
        self.p_G = <fn_t>fG.f_ptr; self.d_G = fG.func_data
        self.p_grad = NULL; self.p_hess = NULL
        keep = [fG]
        self.linear = linear
        self.p_m = <fn_t*>malloc(C * sizeof(fn_t)); self.p_mg = <fn_t*>malloc(C * sizeof(fn_t))
        self.p_mh = <fn_t*>malloc(C * sizeof(fn_t))
        self.d_m = <void**>malloc(C * sizeof(void*)); self.d_mg = <void**>malloc(C * sizeof(void*))
        self.d_mh = <void**>malloc(C * sizeof(void*))
        if not linear:
            for i, el in enumerate(pr.nonvacant_elements):
                f = ff.get_mole_formula_func(str(el)); keep.append(f); self.p_m[i] = <fn_t>f.f_ptr; self.d_m[i] = f.func_data
                f = ff.get_mole_formula_grad(str(el)); keep.append(f); self.p_mg[i] = <fn_t>f.f_ptr; self.d_mg[i] = f.func_data
                f = ff.get_mole_formula_hess(str(el)); keep.append(f); self.p_mh[i] = <fn_t>f.f_ptr; self.d_mh[i] = f.func_data
        self.keep = keep
        subl = np.asarray(sublattice_of, dtype=np.int32)
        self.nb = int(subl.max()) + 1
        counts = np.bincount(subl, minlength=self.nb)
        self.boff = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
        self.bidx = np.argsort(subl, kind='stable').astype(np.int32)
        self.blk = subl.copy()
        self.elem_of = np.asarray(elem_of, dtype=np.int32).copy()
        self.single = (counts == 1).astype(np.int32)
        self.stoich = bool(np.all(counts == 1))
        self.pinned = np.zeros(D, dtype=np.int32)
        self.A = np.array(A, dtype=float); self.asum = np.asarray(A).sum(axis=0)
        EE = np.array(E, dtype=float).reshape(-1, D)
        self.ne_fixed = EE.shape[0]
        self.set_constraints(EE, np.asarray(e, dtype=float).reshape(-1))
        self.inp = np.zeros(n + self.npar); self.graw = np.zeros(n); self.hraw = np.zeros(n * n); self.gsv = np.zeros(self.ns)
        self.mval = np.zeros(C); self.mgraw = np.zeros(C * n); self.mhraw = np.zeros(C * n * n)
        self.gm = np.zeros(C * D); self.gM = np.zeros(D); self.hM = np.zeros(D * D)
        self.out1 = np.zeros(1)
        self.nu = np.zeros(D); self.mu_cur = np.zeros(C)
        self.y = np.zeros(D); self.z = np.zeros(D); self.zt = np.zeros(D); self.yt = np.zeros(D)
        self.gz = np.zeros(D); self.gzt = np.zeros(D); self.gy = np.zeros(D); self.gyt = np.zeros(D); self.gl = np.zeros(D)
        self.d = np.zeros(D); self.w = np.zeros(D); self.coef = np.zeros(D)
        self.Hz = np.zeros(D * D); self.Hy = np.zeros(D * D); self.M = np.zeros(D * D)
        self.V = np.zeros(D * D); self.Hs = np.zeros(D * D)
        self.bt = np.zeros(self.nb); self.sc = np.zeros(D); self.free = np.zeros(D, dtype=np.int32)
        self.Bm = np.zeros(D * C); self.Wm = np.zeros(D * C); self.xs = np.zeros(C)
        self.lwork = max(1, 64 * D); self.work = np.zeros(self.lwork)
        self.last_mu = np.zeros(C); self.has_last = False; self.last_tpd = 0.0
        self.m_lo = 0.0; self.pen_w = 0.0; self.max_dz = 30.0
        self.evals1 = 0; self.evals2 = 0; self.nprice = 0; self.nfail = 0; self.nescape = 0; self.nsaddle = 0
        self._t_index = temperature_index
        # sublattices with the same species (in the same order): configurations equal on two of them can be
        # stationary in the ordering direction without being a minimum (see escape_saddle)
        species = [[str(pr.variables[j].species) for j in self.bidx[self.boff[b]:self.boff[b + 1]]]
                   for b in range(self.nb)]
        sa, sb, off = [], [], [0]
        for b1 in range(self.nb):
            for b2 in range(b1 + 1, self.nb):
                if len(species[b1]) > 1 and species[b1] == species[b2]:
                    sa.extend(self.bidx[self.boff[b1]:self.boff[b1 + 1]]); sb.extend(self.bidx[self.boff[b2]:self.boff[b2 + 1]])
                    off.append(len(sa))
        self.sym_a = np.array(sa, dtype=np.int32).reshape(-1); self.sym_b = np.array(sb, dtype=np.int32).reshape(-1)
        self.sym_off = np.array(off, dtype=np.int32); self.nsym = len(off) - 1

    def _load_derivatives(self):
        cdef FastFunctionFactory ff = (<PhaseRecord>self.phase_record).function_factory
        cdef FastFunction fg, fh
        # one compilation at a time: the factory's look-up and compile is not safe against another thread doing the
        # same (worker threads pricing copies of one phase reach this together), and LLVM compilations must not run
        # concurrently
        with _COMPILE_LOCK:
            fg = <FastFunction>ff.get_grad(string(b'G'))
            fh = <FastFunction>ff.get_hess(string(b'G'))
        self.keep.extend([fg, fh])
        self.p_grad = <fn_t>fg.f_ptr; self.d_grad = fg.func_data
        self.p_hess = <fn_t>fh.f_ptr; self.d_hess = fh.func_data

    cdef inline void ensure_derivatives(self) noexcept nogil:
        if self.p_grad == NULL:
            with gil:
                self._load_derivatives()

    def __dealloc__(self):
        free(self.p_m); free(self.p_mg); free(self.p_mh); free(self.d_m); free(self.d_mg); free(self.d_mh)

    def clone(self):
        """The same phase with its own work buffers and state (for another thread); the compiled functions are
        shared."""
        P = CPhase(*self.ctor_args)
        P.set_bounds(self.m_lo)
        return P

    def device_description(self):
        """What cg_price.h needs about this phase: sublattice structure, formula moles (linear phases), bounds; and
        whether that port can price it (linear formula moles, no constraints beyond the sublattice balances, no
        pinned species)."""
        return {'D': self.D, 'nb': self.nb, 'boff': np.asarray(self.boff).copy(), 'bidx': np.asarray(self.bidx).copy(),
                'blk': np.asarray(self.blk).copy(), 'single': np.asarray(self.single).copy(),
                'pinned': np.asarray(self.pinned).copy(), 'nsym': self.nsym,
                'sym_a': np.asarray(self.sym_a).copy(), 'sym_b': np.asarray(self.sym_b).copy(),
                'sym_off': np.asarray(self.sym_off).copy(), 'A': np.asarray(self.A).copy(),
                'asum': np.asarray(self.asum).copy(), 'm_lo': self.m_lo, 'pen_w': self.pen_w,
                'max_dz': self.max_dz, 'RT': self.RT,
                'supported': bool(self.linear and self.ne == 0 and not np.asarray(self.pinned).any() and not self.stoich)}

    def set_constraints(self, E, e):
        """Linear equality constraints E y = e beyond the site-fraction balances."""
        cdef int D = self.D
        EE = np.array(E, dtype=float).reshape(-1, D)
        ee = np.array(e, dtype=float).reshape(-1)
        if EE.shape[0]:
            # scale each row by its right-hand side (else its largest coefficient), so that a trace value such as
            # y_j = 1e-10 is met to relative precision
            rowmax = np.abs(EE).max(axis=1)
            ee = np.where(np.abs(ee) > 1e-13 * rowmax, ee, 0.0)   # (a computed zero, e.g. charge balance)
            scale = np.where(np.abs(ee) > 0, np.abs(ee), rowmax)
            scale = np.where(scale > 0, scale, 1.0)
            EE = EE / scale[:, None]; ee = ee / scale
        e = ee
        self.ne = EE.shape[0]
        k = max(self.ne, 1)
        self.E = EE if self.ne else np.zeros((1, D)); self.e = np.array(e, dtype=float).reshape(-1) if self.ne else np.zeros(1)
        self.lam = np.zeros(k); self.hres = np.zeros(k)
        self.As = np.zeros(k * D); self.WA = np.zeros(k * D); self.Se = np.zeros(k * k); self.tmpe = np.zeros(k * max(self.C, 1))
        self.ipe = np.zeros(k, dtype=np.int32)

    def set_local_constraints(self, E, e):
        """Phase-local conditions as constraints E y = e, after the phase's own (pass empty arrays to clear)."""
        base_E = np.asarray(self.E)[:self.ne_fixed]; base_e = np.asarray(self.e)[:self.ne_fixed]
        EE = np.vstack([base_E, np.array(E, dtype=float).reshape(-1, self.D)])
        self.set_constraints(EE, np.concatenate([base_e, np.array(e, dtype=float).reshape(-1)]))
        self.has_last = False

    def set_pinned(self, mask):
        """Species held at the site-fraction floor (phase-local site-fraction conditions at 0, or at 1 for another
        species of the sublattice)."""
        self.pinned = np.asarray(mask, dtype=np.int32).copy()
        self.has_last = False

    def set_bounds(self, double m_lo):
        """Atoms per formula unit below which the energy is penalized (phases with a pure-vacancy end member)."""
        self.m_lo = m_lo
        self.pen_w = 1e4 / m_lo ** 3 if m_lo > 0 else 0.0

    def set_state(self, statevars, parameters):
        """State-variable values (in the PhaseRecord's order) and parameter values for the following evaluations."""
        cdef int k
        sv = np.asarray(statevars, dtype=float)
        for k in range(self.ns): self.inp[k] = sv[k]
        par = np.asarray(parameters, dtype=float)
        for k in range(self.npar): self.inp[self.ns + self.D + k] = par[k]
        self.T = sv[self._t_index]; self.RT = R_GAS * self.T
        self.has_last = False

    cdef void moles(self, double* y) noexcept nogil:
        """m (mval), dm/dy (gm), M (Mtot), dM/dy (gM) at y; inp must hold y."""
        cdef int D = self.D, C = self.C, ns = self.ns, n = self.ns + self.D, i, j
        cdef double s
        if self.linear:
            self.Mtot = 0.0
            for j in range(D): self.Mtot += self.asum[j] * y[j]
            for i in range(C):
                s = 0.0
                for j in range(D): s += self.A[i, j] * y[j]
                self.mval[i] = s
            return
        self.Mtot = 0.0
        for j in range(D): self.gM[j] = 0.0
        for i in range(C):
            self.p_m[i](&self.mval[i], &self.inp[0], self.d_m[i])
            self.p_mg[i](&self.mgraw[i * n], &self.inp[0], self.d_mg[i])
            self.Mtot += self.mval[i]
            for j in range(D):
                self.gm[i * D + j] = self.mgraw[i * n + ns + j]
                self.gM[j] += self.mgraw[i * n + ns + j]

    cdef inline double gm_ij(self, int i, int j) noexcept nogil:
        return self.A[i, j] if self.linear else self.gm[i * self.D + j]

    cdef inline double gM_j(self, int j) noexcept nogil:
        return self.asum[j] if self.linear else self.gM[j]

    cdef void eval0(self, double* y) noexcept nogil:
        """Energy and formula moles only; GM per mole of atoms in out1[0]."""
        memcpy(&self.inp[self.ns], y, self.D * sizeof(double))
        self.p_G(&self.G_fu, &self.inp[0], self.d_G)
        self.moles(y)
        self.out1[0] = self.G_fu / self.Mtot

    cdef void eval1(self, double* y) noexcept nogil:
        """Energy and its gradient (graw), formula moles (moles()); GM per mole of atoms in out1[0]."""
        cdef int D = self.D, ns = self.ns, j
        self.evals1 += 1
        self.ensure_derivatives()
        memcpy(&self.inp[ns], y, D * sizeof(double))
        self.p_G(&self.G_fu, &self.inp[0], self.d_G)
        self.p_grad(&self.graw[0], &self.inp[0], self.d_grad)
        self.moles(y)
        self.out1[0] = self.G_fu / self.Mtot
        for j in range(ns): self.gsv[j] = self.graw[j] / self.Mtot

    cdef void eval2(self, double* y) noexcept nogil:
        """eval1, the energy Hessian (hraw) and, for non-linear formula moles, d2m (mhraw) and d2M (hM)."""
        cdef int D = self.D, C = self.C, ns = self.ns, n = self.ns + self.D, i, a, b
        self.eval1(y)
        self.evals2 += 1
        self.p_hess(&self.hraw[0], &self.inp[0], self.d_hess)
        if not self.linear:
            for a in range(D * D): self.hM[a] = 0.0
            for i in range(C):
                self.p_mh[i](&self.mhraw[i * n * n], &self.inp[0], self.d_mh[i])
                for a in range(D):
                    for b in range(D):
                        self.hM[a * D + b] += self.mhraw[i * n * n + (ns + a) * n + ns + b]

    cdef void y_of_z(self, double* z, double* y) noexcept nogil:
        cdef int b, k, j
        cdef double m, s
        for b in range(self.nb):
            m = -INFINITY
            for k in range(self.boff[b], self.boff[b + 1]):
                j = self.bidx[k]
                if z[j] > m and not self.pinned[j]: m = z[j]
            s = 0.0
            for k in range(self.boff[b], self.boff[b + 1]):
                j = self.bidx[k]
                if self.pinned[j]:
                    y[j] = 0.0
                else:
                    y[j] = exp(z[j] - m); s += y[j]
            for k in range(self.boff[b], self.boff[b + 1]):
                j = self.bidx[k]
                y[j] = y[j] / s if s > 0 else 0.0
                if y[j] < Y_FLOOR: y[j] = Y_FLOOR

    cdef inline void set_mu(self, double* mu) noexcept nogil:
        cdef int i, j
        cdef double s
        for i in range(self.C): self.mu_cur[i] = mu[i]
        if self.linear:
            for j in range(self.D):
                s = 0.0
                for i in range(self.C): s += mu[i] * self.A[i, j]
                self.nu[j] = s

    cdef inline void x_of_y(self, double* y, double* x) noexcept nogil:
        cdef int i
        if not self.linear:
            memcpy(&self.inp[self.ns], y, self.D * sizeof(double))
        self.moles(y)
        for i in range(self.C): x[i] = self.mval[i] / self.Mtot

    cdef double tpd_parts(self, double* y, double* gy) noexcept nogil:
        """After eval1 at y: TPD/RT (without the vacancy penalty) and its gradient in y."""
        cdef int D = self.D, C = self.C, ns = self.ns, i, j
        cdef double u, f, s, M = self.Mtot
        u = self.G_fu / self.RT
        for i in range(C): u -= self.mu_cur[i] * self.mval[i]
        f = u / M
        for j in range(D):
            if self.linear:
                s = self.graw[ns + j] / self.RT - self.nu[j]
            else:
                s = self.graw[ns + j] / self.RT
                for i in range(C): s -= self.mu_cur[i] * self.gm[i * D + j]
            gy[j] = (s - f * self.gM_j(j)) / M
        return f

    cdef double fgz(self, double* z, double* y, double* gz, double* gy) noexcept nogil:
        """TPD/RT at z for the chemical potentials set by set_mu; fills y, gy (d/dy), gz (d/dz); the constraint
        residuals in hres."""
        cdef int D = self.D, j, b, k
        cdef double f, s, M
        self.y_of_z(z, y)
        self.eval1(y)
        f = self.tpd_parts(y, gy)
        M = self.Mtot
        if M < self.m_lo:
            # per mole of atoms the energy is unbounded below as the formula unit empties of atoms: penalize
            f += self.pen_w * (self.m_lo - M) ** 3
            for j in range(D): gy[j] -= 3 * self.pen_w * (self.m_lo - M) ** 2 * self.gM_j(j)
        for b in range(self.nb):
            s = 0.0
            for k in range(self.boff[b], self.boff[b + 1]):
                j = self.bidx[k]; s += y[j] * gy[j]
            for k in range(self.boff[b], self.boff[b + 1]):
                j = self.bidx[k]; gz[j] = y[j] * (gy[j] - s)
        for k in range(self.ne):
            s = -self.e[k]
            for j in range(D): s += self.E[k, j] * y[j]
            self.hres[k] = s
        return f

    cdef inline double gznorm(self) noexcept nogil:
        cdef double s = 0.0
        cdef int j
        for j in range(self.D): s += self.gz[j] * self.gz[j]
        return sqrt(s)

    cdef inline double hnorm(self) noexcept nogil:
        cdef double s = 0.0
        cdef int k
        for k in range(self.ne): s += fabs(self.hres[k])
        return s

    cdef void lagrangian_gradient(self) noexcept nogil:
        """gl = gy + E^T lam (gy itself without constraints)."""
        cdef int j, k
        for j in range(self.D):
            self.gl[j] = self.gy[j]
            for k in range(self.ne): self.gl[j] += self.lam[k] * self.E[k, j]

    cdef double kkt(self) noexcept nogil:
        """Stationarity of the current point, in potentials: with r_j = dL/dy_j minus its y-weighted sublattice mean
        (L the Lagrangian, RT units; at a stationary point every species' r_j is zero), max |r_j| over species above
        the site-fraction floor, and max(-r_j) over species at the floor (which may stay there only if growing would
        raise the TPD); with constraints, also their residuals. (The softmax gradient y_j r_j vanishes as y_j -> 0
        whatever r_j, so it cannot measure trace species.)"""
        cdef int b, k, j
        cdef double s, r, out = 0.0
        self.lagrangian_gradient()
        for b in range(self.nb):
            if self.single[b]: continue
            s = 0.0
            for k in range(self.boff[b], self.boff[b + 1]):
                j = self.bidx[k]; s += self.y[j] * self.gl[j]
            for k in range(self.boff[b], self.boff[b + 1]):
                j = self.bidx[k]; r = self.gl[j] - s
                if self.pinned[j]:
                    continue
                if self.y[j] > 1.5 * Y_FLOOR:
                    if fabs(r) > out: out = fabs(r)
                elif -r > out:
                    out = -r
        for k in range(self.ne):
            if 1e3 * fabs(self.hres[k]) > out: out = 1e3 * fabs(self.hres[k])
        return out if out == out else INFINITY

    cdef int build_hessian(self) noexcept nogil:
        """At the current point (self.y, self.gl, self.mu_cur set): the Gauss-Newton Hessian of the TPD over the free
        variables, Jacobi-scaled by self.sc, in self.V and self.Hs (m x m); with constraints, their scaled Jacobian
        in self.As (ne x m). Returns m."""
        cdef int D = self.D, C = self.C, ns = self.ns, n = self.ns + self.D, i, j, k, a, b, m, p, q
        cdef double s, f, M, hu
        self.eval2(&self.y[0])
        M = self.Mtot
        f = self.tpd_parts(&self.y[0], &self.gyt[0])   # (gyt: gradient without the penalty)
        # Hy = (d2u - df dM^T - dM df^T - f d2M) / M, u = G/RT - mu.m
        for a in range(D):
            for b in range(a, D):
                hu = self.hraw[(ns + a) * n + ns + b] / self.RT
                if not self.linear:
                    for i in range(C): hu -= self.mu_cur[i] * self.mhraw[i * n * n + (ns + a) * n + ns + b]
                s = hu - self.gyt[a] * self.gM_j(b) - self.gM_j(a) * self.gyt[b]
                if not self.linear:
                    s -= f * self.hM[a * D + b]
                s /= M
                if M < self.m_lo:
                    s += 6 * self.pen_w * (self.m_lo - M) * self.gM_j(a) * self.gM_j(b)
                self.Hy[a * D + b] = s; self.Hy[b * D + a] = s
        # Hz = J Hy J, J the softmax Jacobian, block diagonal: (J v)_a = y_a (v_a - sum_{l in block(a)} y_l v_l).
        # (Gauss-Newton: the softmax curvature term vanishes at a stationary point; keeping it makes trace species
        # approach their minimum one e-fold per iteration.)
        for a in range(D):
            for b in range(self.nb):
                s = 0.0
                for k in range(self.boff[b], self.boff[b + 1]):
                    j = self.bidx[k]; s += self.y[j] * self.Hy[a * D + j]
                self.bt[b] = s
            for j in range(D):
                self.M[a * D + j] = self.y[j] * (self.Hy[a * D + j] - self.bt[self.blk[j]])
        for j in range(D):
            for b in range(self.nb):
                s = 0.0
                for k in range(self.boff[b], self.boff[b + 1]):
                    a = self.bidx[k]; s += self.y[a] * self.M[a * D + j]
                self.bt[b] = s
            for a in range(D):
                self.Hz[a * D + j] = self.y[a] * (self.M[a * D + j] - self.bt[self.blk[a]])
        # gauge: z is defined up to a constant per sublattice, so fix the largest species of each sublattice (and drop
        # single-species sublattices); species at the floor that would raise the TPD by growing are held there
        m = 0
        for b in range(self.nb):
            if self.single[b]: continue
            i = self.bidx[self.boff[b]]
            s = 0.0
            for k in range(self.boff[b], self.boff[b + 1]):
                j = self.bidx[k]; s += self.y[j] * self.gl[j]
                if self.y[j] > self.y[i]: i = j
            for k in range(self.boff[b], self.boff[b + 1]):
                j = self.bidx[k]
                if j != i and not self.pinned[j] and not (self.y[j] <= 1.5 * Y_FLOOR and self.gl[j] > s):
                    self.free[m] = j; m += 1
        if m == 0:
            return 0
        for p in range(m):
            s = fabs(self.Hz[self.free[p] * D + self.free[p]])
            self.sc[p] = 1.0 / sqrt(s) if s > 1e-300 else 1.0
        for p in range(m):
            for q in range(m):
                self.V[p * m + q] = self.sc[p] * self.Hz[self.free[p] * D + self.free[q]] * self.sc[q]
        memcpy(&self.Hs[0], &self.V[0], m * m * sizeof(double))
        # constraint Jacobian in the scaled free variables: (E J)_{k,l} = y_l (E_kl - sum_{j in block(l)} E_kj y_j)
        for k in range(self.ne):
            for p in range(m):
                a = self.free[p]; b = self.blk[a]
                s = 0.0
                for q in range(self.boff[b], self.boff[b + 1]):
                    j = self.bidx[q]; s += self.E[k, j] * self.y[j]
                self.As[k * m + p] = self.sc[p] * self.y[a] * (self.E[k, a] - s)
        return m

    cdef int factor(self, int m) noexcept nogil:
        """Cholesky of the scaled Hessian in V (m x m); if indefinite, of its eigenvalue-modified version (|w|,
        floored). Returns 0, or -1 on failure."""
        cdef int info, k, p, q, lw = self.lwork
        cdef double s, aw
        cdef char jobz = b'V', uplo = b'U'
        dpotrf(&uplo, &m, &self.V[0], &m, &info)
        self.last_indef = info != 0
        if info == 0:
            return 0
        memcpy(&self.V[0], &self.Hs[0], m * m * sizeof(double))
        dsyev(&jobz, &uplo, &m, &self.V[0], &m, &self.w[0], &self.work[0], &lw, &info)
        if info != 0:
            return -1
        # rebuild V |w| V^T in Hs's storage, then factor it
        for p in range(m):
            for q in range(m):
                s = 0.0
                for k in range(m):
                    aw = fabs(self.w[k])
                    if aw < 1e-8: aw = 1e-8
                    s += self.V[k * m + p] * aw * self.V[k * m + q]
                self.Hy[p * m + q] = s
        memcpy(&self.V[0], &self.Hy[0], m * m * sizeof(double))
        dpotrf(&uplo, &m, &self.V[0], &m, &info)
        return 0 if info == 0 else -1

    cdef bint symmetric(self) noexcept nogil:
        """Whether two sublattices with the same species have the same site fractions at the current point."""
        cdef int k, q
        for k in range(self.nsym):
            for q in range(self.sym_off[k], self.sym_off[k + 1]):
                if fabs(self.y[self.sym_a[q]] - self.y[self.sym_b[q]]) > 1e-7:
                    break
            else:
                return True
        return False

    cdef int escape_saddle(self, double* f) noexcept nogil:
        """At a stationary point (unconstrained phases): if the Hessian has a direction of negative curvature, move
        along it to a lower point and return 1, else 0. A symmetric configuration (e.g. a disordered B2 or L1_2,
        equal sublattices) is stationary in the ordering directions whether or not it is a minimum, and Newton with
        the modified Hessian never leaves it; the ordered minimum below it is found from the lower point."""
        cdef int D = self.D, m, p, info, k, sgn, lw = self.lwork
        cdef double s, t, ft, best = f[0]
        cdef char jobz = b'V', uplo = b'U'
        self.nsaddle += 1
        m = self.build_hessian()
        if m == 0:
            return 0
        dpotrf(&uplo, &m, &self.V[0], &m, &info)
        if info == 0:
            return 0
        memcpy(&self.V[0], &self.Hs[0], m * m * sizeof(double))
        dsyev(&jobz, &uplo, &m, &self.V[0], &m, &self.w[0], &self.work[0], &lw, &info)
        if info != 0 or self.w[0] > -1e-6:
            return 0
        # the eigenvector of the most negative eigenvalue, in z; steps up to 0.5 in log site fraction, both ways
        for p in range(D): self.d[p] = 0.0
        s = 0.0
        for p in range(m):
            self.d[self.free[p]] = self.sc[p] * self.V[p]
            if fabs(self.d[self.free[p]]) > s: s = fabs(self.d[self.free[p]])
        if not s > 0:
            return 0
        for p in range(D): self.d[p] *= 0.5 / s
        t = 1.0
        for k in range(12):
            for sgn in range(2):
                for p in range(D): self.zt[p] = self.z[p] + (t if sgn == 0 else -t) * self.d[p]
                ft = self.fgz(&self.zt[0], &self.yt[0], &self.gzt[0], &self.gyt[0])
                if isfinite(ft) and ft < best - 1e-12 and self.hnorm() <= 1e-9:
                    self.nescape += 1
                    memcpy(&self.z[0], &self.zt[0], D * sizeof(double)); memcpy(&self.y[0], &self.yt[0], D * sizeof(double))
                    memcpy(&self.gz[0], &self.gzt[0], D * sizeof(double)); memcpy(&self.gy[0], &self.gyt[0], D * sizeof(double))
                    f[0] = ft
                    return 1
            t *= 0.5
        # (no lower point along it: restore the residuals of the current point)
        f[0] = self.fgz(&self.z[0], &self.y[0], &self.gz[0], &self.gy[0])
        return 0

    cdef int newton_direction(self, int m) noexcept nogil:
        """Direction in self.d from the factored scaled Hessian (V): unconstrained, or the KKT step with the
        constraints (multipliers in self.lam). Returns 0, or -1 on failure."""
        cdef int D = self.D, ne = self.ne, p, k, l, info, nrhs = 1
        cdef double s
        cdef char uplo = b'U'
        for p in range(m):
            self.coef[p] = -self.sc[p] * self.gz[self.free[p]]
        dpotrs(&uplo, &m, &nrhs, &self.V[0], &m, &self.coef[0], &m, &info)   # v = -H^-1 g
        if ne > 0:
            # W = H^-1 As^T; (As W) lam = As v + h; d = v - W lam
            for k in range(ne):
                for p in range(m): self.WA[k * m + p] = self.As[k * m + p]
            dpotrs(&uplo, &m, &ne, &self.V[0], &m, &self.WA[0], &m, &info)
            for k in range(ne):
                for l in range(ne):
                    s = 0.0
                    for p in range(m): s += self.As[k * m + p] * self.WA[l * m + p]
                    self.Se[l * ne + k] = s
                s = self.hres[k]
                for p in range(m): s += self.As[k * m + p] * self.coef[p]
                self.lam[k] = s
            dgesv(&ne, &nrhs, &self.Se[0], &ne, &self.ipe[0], &self.lam[0], &ne, &info)
            if info != 0:
                return -1
            for p in range(m):
                s = 0.0
                for k in range(ne): s += self.WA[k * m + p] * self.lam[k]
                self.coef[p] -= s
            # (multipliers of the scaled constraint rows are those of E y = e in the RT-scaled TPD)
        for p in range(D): self.d[p] = 0.0
        for p in range(m):
            self.d[self.free[p]] = self.sc[p] * self.coef[p]
        return 0

    cdef int price(self, double* mu, double* y0, double* y_out, double* x_out, double* G_out, double* tpd_out) noexcept nogil:
        """Minimize TPD (mu in RT units) from y0, subject to the constraints. Returns 1 if it reached a stationary
        point, else 0. tpd_out is in RT units; G_out (J/mol of atoms) is NaN outside the admissible domain."""
        cdef int D = self.D, C = self.C, it, ls, i, j, a, m, escapes
        cdef double f, ft, gnorm, slope, step, s, rho = 0.0, phi, phit, h0
        cdef bint accepted
        for j in range(D):
            s = y0[j]
            if s < Y_FLOOR: s = Y_FLOOR
            self.z[j] = log(s)
        self.set_mu(mu)
        self.nprice += 1
        self.last_indef = False   # (nothing carried over from the previous minimization: jobs are independent)
        for j in range(self.ne): self.lam[j] = 0.0
        f = self.fgz(&self.z[0], &self.y[0], &self.gz[0], &self.gy[0])
        escapes = 0
        for it in range(60):
            if self.kkt() < 1e-9:
                # (only symmetric points are stationary in a direction of negative curvature by construction: elsewhere
                # Newton with the modified Hessian moves off a saddle; and where the Hessian of the last Newton step
                # was positive definite, it is so at the stationary point next to it)
                if (self.ne == 0 and escapes < 3 and self.nsym > 0 and (it == 0 or self.last_indef) and self.symmetric()
                        and self.escape_saddle(&f)):
                    escapes += 1
                    continue
                break
            m = self.build_hessian()
            if m == 0:
                break
            if self.factor(m) != 0:
                break
            if self.newton_direction(m) != 0:
                break
            s = 0.0
            for a in range(D):
                if fabs(self.d[a]) > s: s = fabs(self.d[a])
            if s > self.max_dz:
                for a in range(D): self.d[a] *= self.max_dz / s
            slope = 0.0
            for a in range(D): slope += self.gz[a] * self.d[a]
            # merit f + rho |h|_1, rho above the multipliers
            for j in range(self.ne):
                if 2 * fabs(self.lam[j]) + 1e-3 > rho: rho = 2 * fabs(self.lam[j]) + 1e-3
            h0 = self.hnorm()
            phi = f + rho * h0
            slope -= rho * h0
            step = 1.0; accepted = False
            for ls in range(30):
                for j in range(D): self.zt[j] = self.z[j] + step * self.d[j]
                ft = self.fgz(&self.zt[0], &self.yt[0], &self.gzt[0], &self.gyt[0])
                phit = ft + rho * self.hnorm()
                if isfinite(ft) and phit <= phi + 1e-4 * step * (slope if slope < 0 else 0.0) + 1e-12:
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                # restore the residuals of the current point
                f = self.fgz(&self.z[0], &self.y[0], &self.gz[0], &self.gy[0])
                break
            memcpy(&self.z[0], &self.zt[0], D * sizeof(double)); memcpy(&self.y[0], &self.yt[0], D * sizeof(double))
            memcpy(&self.gz[0], &self.gzt[0], D * sizeof(double)); memcpy(&self.gy[0], &self.gyt[0], D * sizeof(double))
            f = ft
        # steepest-descent clean-up if Newton stalled (unconstrained phases only)
        gnorm = self.gznorm()
        it = 0
        while self.ne == 0 and self.kkt() >= 1e-7 and it < 200 and isfinite(gnorm):
            step = 1.0 / (gnorm if gnorm > 1.0 else 1.0); accepted = False
            for ls in range(40):
                for j in range(D): self.zt[j] = self.z[j] - step * self.gz[j]
                ft = self.fgz(&self.zt[0], &self.yt[0], &self.gzt[0], &self.gyt[0])
                if isfinite(ft) and ft <= f - 1e-4 * step * gnorm * gnorm:
                    accepted = True; break
                step *= 0.5
            if not accepted: break
            memcpy(&self.z[0], &self.zt[0], D * sizeof(double)); memcpy(&self.y[0], &self.yt[0], D * sizeof(double))
            memcpy(&self.gz[0], &self.gzt[0], D * sizeof(double)); memcpy(&self.gy[0], &self.gyt[0], D * sizeof(double))
            f = ft
            gnorm = self.gznorm()
            it += 1
        self.eval1(&self.y[0])
        memcpy(y_out, &self.y[0], D * sizeof(double))
        for i in range(C): x_out[i] = self.mval[i] / self.Mtot
        G_out[0] = self.out1[0]
        s = self.out1[0] / self.RT
        for i in range(C): s -= mu[i] * x_out[i]
        tpd_out[0] = s
        if self.Mtot < self.m_lo or self.hnorm() > 1e-9:
            G_out[0] = NAN   # outside the admissible domain, or the constraints are not met: never stored
        if self.kkt() >= 1e-7:
            self.nfail += 1
            return 0
        return 1

    cdef int sens(self, double* S) noexcept nogil:
        return self.sens_full(S, NULL)

    cdef int sens_full(self, double* S, double* Xsv) noexcept nogil:
        """After price(): S = dx/dmu (C x C, mu in RT units) at the minimum, B P B^T with B = dx/dz over the free
        variables and P = H^-1 - H^-1 A^T (A H^-1 A^T)^-1 A H^-1 (A the constraint Jacobian; P = H^-1 without
        constraints). If Xsv is given, also dx/d(state variable) at fixed mu in J/mol (C x ns, row-major):
        -B P J d2GM/dsv dy / RT. Returns 0, or -1 if the Hessian there is not positive definite."""
        cdef int D = self.D, C = self.C, ne = self.ne, ns = self.ns, n = self.ns + self.D, m, p, q, i, j, k, l, b, t, info
        cdef double M, s, u
        cdef int nrhs = 1
        cdef char uplo = b'U', N = b'N'
        for i in range(C * C): S[i] = 0.0
        if Xsv != NULL:
            for i in range(C * ns): Xsv[i] = 0.0
        if self.stoich:
            return 0
        self.lagrangian_gradient()
        m = self.build_hessian()
        if m == 0:
            return 0
        dpotrf(&uplo, &m, &self.V[0], &m, &info)
        if info != 0:
            return -1
        M = self.Mtot
        for i in range(C): self.xs[i] = self.mval[i] / M
        # B[i, p] = sc_p y_k (Xy_ik - sum_{j in block(k)} Xy_ij y_j), k = free[p], Xy_ij = (dm_ij - x_i dM_j) / M
        for i in range(C):
            for p in range(m):
                k = self.free[p]; b = self.blk[k]
                s = 0.0
                for q in range(self.boff[b], self.boff[b + 1]):
                    j = self.bidx[q]; s += (self.gm_ij(i, j) - self.xs[i] * self.gM_j(j)) / M * self.y[j]
                self.Bm[i * m + p] = self.sc[p] * self.y[k] * ((self.gm_ij(i, k) - self.xs[i] * self.gM_j(k)) / M - s)
        memcpy(&self.Wm[0], &self.Bm[0], m * C * sizeof(double))
        dpotrs(&uplo, &m, &C, &self.V[0], &m, &self.Wm[0], &m, &info)   # Wm = H^-1 B^T
        if ne > 0:
            # WA = H^-1 A^T; Se = A H^-1 A^T, LU-factored in Se/ipe
            for k in range(ne):
                for p in range(m): self.WA[k * m + p] = self.As[k * m + p]
            dpotrs(&uplo, &m, &ne, &self.V[0], &m, &self.WA[0], &m, &info)
            for k in range(ne):
                for l in range(ne):
                    s = 0.0
                    for p in range(m): s += self.As[k * m + p] * self.WA[l * m + p]
                    self.Se[l * ne + k] = s
            dgetrf(&ne, &ne, &self.Se[0], &ne, &self.ipe[0], &info)
            if info != 0:
                return -1
            # project each column of Wm: w -= WA Se^-1 (A w)
            for j in range(C):
                for k in range(ne):
                    s = 0.0
                    for p in range(m): s += self.As[k * m + p] * self.Wm[j * m + p]
                    self.tmpe[k] = s
                dgetrs(&N, &ne, &nrhs, &self.Se[0], &ne, &self.ipe[0], &self.tmpe[0], &ne, &info)
                for p in range(m):
                    s = 0.0
                    for k in range(ne): s += self.WA[k * m + p] * self.tmpe[k]
                    self.Wm[j * m + p] -= s
        for i in range(C):
            for j in range(C):
                s = 0.0
                for p in range(m): s += self.Bm[i * m + p] * self.Wm[j * m + p]
                S[i * C + j] = s
        if Xsv != NULL:
            for t in range(ns):
                # g_j = d2GM/dsv_t dy_j (J): (d2G/dsv dy_j - dGM/dsv dM_j) / M; c_p = sc_p (J g)_{free p}
                for j in range(D):
                    self.gyt[j] = (self.hraw[t * n + ns + j] - self.gsv[t] * self.gM_j(j)) / M
                for p in range(m):
                    k = self.free[p]; b = self.blk[k]
                    s = 0.0
                    for q in range(self.boff[b], self.boff[b + 1]):
                        j = self.bidx[q]; s += self.y[j] * self.gyt[j]
                    self.coef[p] = self.sc[p] * self.y[k] * (self.gyt[k] - s)
                dpotrs(&uplo, &m, &nrhs, &self.V[0], &m, &self.coef[0], &m, &info)   # v = H^-1 c
                if ne > 0:
                    for k in range(ne):
                        s = 0.0
                        for p in range(m): s += self.As[k * m + p] * self.coef[p]
                        self.tmpe[k] = s
                    dgetrs(&N, &ne, &nrhs, &self.Se[0], &ne, &self.ipe[0], &self.tmpe[0], &ne, &info)
                    for p in range(m):
                        s = 0.0
                        for k in range(ne): s += self.WA[k * m + p] * self.tmpe[k]
                        self.coef[p] -= s
                for i in range(C):
                    s = 0.0
                    for p in range(m): s += self.Bm[i * m + p] * self.coef[p]
                    Xsv[i * ns + t] = -s / self.RT
        return 0

    cdef void composition_start(self, double* comp, double* y) noexcept nogil:
        """Each mixing sublattice set to the given element composition (where allowed)."""
        cdef int b, k, j, e
        cdef double s
        for j in range(self.D):
            e = self.elem_of[j]
            y[j] = 1e-6
            if e >= 0 and comp[e] > 1e-6:
                y[j] = comp[e]
        for b in range(self.nb):
            if self.single[b]:
                y[self.bidx[self.boff[b]]] = 1.0
                continue
            s = 0.0
            for k in range(self.boff[b], self.boff[b + 1]): s += y[self.bidx[k]]
            for k in range(self.boff[b], self.boff[b + 1]): y[self.bidx[k]] /= s

    cdef long pattern(self, double* y) noexcept nogil:
        """Ordering pattern: the dominant species on each sublattice."""
        cdef long code = 0, mult = 1
        cdef int b, k, best
        cdef double m
        for b in range(self.nb):
            best = 0; m = -1.0
            for k in range(self.boff[b], self.boff[b + 1]):
                if y[self.bidx[k]] > m:
                    m = y[self.bidx[k]]; best = k - self.boff[b]
            code += best * mult
            mult *= (self.boff[b + 1] - self.boff[b])
        return code

    def evaluate(self, y):
        """(GM, x) at y."""
        cdef double[::1] yy = np.ascontiguousarray(y, dtype=float)
        cdef double[::1] x = np.zeros(self.C)
        cdef int i
        self.eval0(&yy[0])
        for i in range(self.C): x[i] = self.mval[i] / self.Mtot
        return self.out1[0], np.asarray(x)

    def set_statevars(self, statevars):
        """New state-variable values (parameters unchanged)."""
        cdef int k
        sv = np.asarray(statevars, dtype=float)
        for k in range(self.ns): self.inp[k] = sv[k]
        self.T = sv[self._t_index]; self.RT = R_GAS * self.T

    def statevars(self):
        return np.asarray(self.inp[:self.ns]).copy()

    def local_point(self, mu_J, y0):
        """The local TPD minimum from y0 at chemical potentials mu (J/mol) and the current state variables, with the
        derivatives a Newton step on the equilibrium conditions needs: (converged, y, x, GM, TPD (J/mol),
        dx/dmu (C x C, per J/mol), dx/dsv (C x ns), dGM/dsv (ns)). By the envelope theorem dTPD/dmu = -x and
        dTPD/dsv = dGM/dsv at fixed y."""
        cdef double[::1] mu = np.asarray(mu_J, dtype=float) / self.RT
        cdef double[::1] yy0 = np.ascontiguousarray(y0, dtype=float)
        cdef double[::1] y = np.zeros(self.D), x = np.zeros(self.C)
        cdef double[::1] S = np.zeros(self.C * self.C), Xsv = np.zeros(self.C * self.ns)
        cdef double G = 0.0, tpd = 0.0
        cdef int ok
        conv = self.price(&mu[0], &yy0[0], &y[0], &x[0], &G, &tpd)
        ok = self.sens_full(&S[0], &Xsv[0])
        return (bool(conv) and ok == 0, np.asarray(y).copy(), np.asarray(x).copy(), G, tpd * self.RT,
                np.asarray(S).reshape(self.C, self.C) / self.RT, np.asarray(Xsv).reshape(self.C, self.ns).copy(),
                np.asarray(self.gsv).copy())

    def bench(self, y, int n):
        """Seconds per value, value+gradient and Hessian evaluation at y."""
        import time
        cdef double[::1] yy = np.ascontiguousarray(y, dtype=float)
        cdef int k
        self.ensure_derivatives()
        t = time.perf_counter()
        for k in range(n): self.eval0(&yy[0])
        t0 = (time.perf_counter() - t) / n
        t = time.perf_counter()
        for k in range(n): self.eval1(&yy[0])
        t1 = (time.perf_counter() - t) / n
        t = time.perf_counter()
        for k in range(n): self.eval2(&yy[0])
        return t0, t1, (time.perf_counter() - t) / n

    def energy_many(self, Y):
        """Energy per formula unit at each row of site fractions (no derivatives)."""
        cdef double[:, ::1] YY = np.ascontiguousarray(Y, dtype=float)
        cdef double[::1] out = np.zeros(YY.shape[0])
        cdef int r
        for r in range(YY.shape[0]):
            self.eval0(&YY[r, 0])
            out[r] = self.G_fu
        return np.asarray(out)

    def py_price(self, mu_J, y0):
        """Pricing from Python (mu in J/mol): (tpd J/mol, y, x, GM, converged)."""
        cdef double[::1] mu = np.asarray(mu_J, dtype=float) / self.RT
        cdef double[::1] yy0 = np.ascontiguousarray(y0, dtype=float)
        cdef double[::1] y = np.zeros(self.D), x = np.zeros(self.C)
        cdef double G = 0.0, tpd = 0.0
        conv = self.price(&mu[0], &yy0[0], &y[0], &x[0], &G, &tpd)
        return tpd * self.RT, np.asarray(y), np.asarray(x), G, bool(conv)


cdef class Solver:
    """Column generation over a fixed list of phases. Stored points persist across calls (warm starts); the
    caller clears them when the state variables change."""
    cdef public list phases
    cdef public int C, K, Dmax, npts, cap, nph
    cdef double[:, ::1] Py, Px, R
    cdef double[::1] PG, b, mf
    cdef int[::1] Pph, Pnext, head, basis, ipiv, fixed
    cdef long[::1] Pcode
    cdef public double RT
    cdef double[::1] mu, dual, xB, LU, tmp, cB, ybuf, xbuf, ybest, xbest, ystart, colbuf, ycert
    cdef public long lp_iters, priced, skipped, npolish, npolish_try, nunpolished
    cdef public bint use_polish, last_polished
    cdef public int polish_fail
    cdef public object finish_reason, cert_log, gauge
    cdef public int cert_kind, cert_phase
    cdef public double cert_tpd
    cdef void** pptr                     # the phases, for calls without the GIL
    cdef double[:, ::1] seedY, Rbuf
    cdef double[::1] bbuf, mfbuf
    cdef int[::1] fixedbuf
    cdef int[::1] seed_off
    cdef int ngauge
    cdef double[:, ::1] gauge_mv
    cdef double[::1] pol_y, pol_x, pol_S, pol_J, pol_F, pol_mu, pol_n, pol_G, pol_t, pol_RS, pol_J2, pol_F2, pol_sv, pol_work
    cdef int[::1] pol_ph, pol_ip
    cdef int pol_na
    cdef bint seed_added
    cdef public double skip_margin, prune, tol, prune_late, seed_prune
    cdef public int max_late

    def __init__(self, phases, int C, double skip_margin=1.0, double prune=1.0, double tol=1e-6,
                 double prune_late=0.25, int max_late=3, double seed_prune=INFINITY):
        self.phases = list(phases); self.nph = len(phases); self.C = C; self.K = C
        self.pptr = <void**>malloc(self.nph * sizeof(void*))
        for i, ph in enumerate(self.phases): self.pptr[i] = <void*>ph
        self.seedY = np.zeros((1, 1)); self.seed_off = np.zeros(self.nph + 1, dtype=np.int32)
        self.cert_kind = 0; self.cert_phase = -1; self.cert_tpd = 0.0
        self.Dmax = max(p.D for p in phases)
        self.skip_margin = skip_margin; self.prune = prune; self.tol = tol
        self.prune_late = prune_late; self.max_late = max_late; self.seed_prune = seed_prune
        self.use_polish = True
        self.gauge = None; self.ngauge = 0; self.gauge_mv = np.zeros((1, C))
        self.lp_iters = 0; self.priced = 0; self.skipped = 0; self.npolish = 0; self.npolish_try = 0; self.nunpolished = 0
        self.pol_y = np.zeros(C * self.Dmax); self.pol_x = np.zeros(C * C); self.pol_S = np.zeros(C * C * C)
        self.pol_J = np.zeros(4 * C * C); self.pol_F = np.zeros(2 * C); self.pol_mu = np.zeros(C)
        self.pol_n = np.zeros(C); self.pol_G = np.zeros(C); self.pol_t = np.zeros(C); self.pol_RS = np.zeros(C * C)
        self.pol_ph = np.zeros(C, dtype=np.int32); self.pol_ip = np.zeros(2 * C, dtype=np.int32)
        self.pol_J2 = np.zeros(4 * C * C); self.pol_F2 = np.zeros(2 * C); self.pol_sv = np.zeros(2 * C); self.pol_work = np.zeros(64 * C + 64)
        self.cap = 256
        self.Py = np.zeros((self.cap, self.Dmax)); self.Px = np.zeros((self.cap, C)); self.PG = np.zeros(self.cap)
        self.Pph = np.zeros(self.cap, dtype=np.int32); self.Pnext = np.full(self.cap, -1, dtype=np.int32)
        self.Pcode = np.zeros(self.cap, dtype=np.int64)
        self.mu = np.zeros(C); self.dual = np.zeros(C); self.xB = np.zeros(C); self.LU = np.zeros(C * C)
        self.tmp = np.zeros(C); self.cB = np.zeros(C); self.ipiv = np.zeros(C, dtype=np.int32)
        self.colbuf = np.zeros(C)
        self.ybuf = np.zeros(self.Dmax); self.xbuf = np.zeros(C); self.ybest = np.zeros(self.Dmax)
        self.xbest = np.zeros(C); self.ystart = np.zeros(self.Dmax); self.ycert = np.zeros(self.Dmax)
        self.R = np.eye(C); self.b = np.zeros(C); self.mf = np.zeros(C); self.fixed = np.zeros(C, dtype=np.int32)
        self.Rbuf = np.zeros((C, C)); self.bbuf = np.zeros(C); self.mfbuf = np.zeros(C); self.fixedbuf = np.zeros(C, dtype=np.int32)
        self.reset_points()

    def __dealloc__(self):
        free(self.pptr)

    def reset_points(self):
        """Forget all points (e.g. after a change of the state variables)."""
        self.npts = 0
        self.head = np.full(self.nph, -1, dtype=np.int32)
        self.basis = np.array([-(1 + k) for k in range(self.C)], dtype=np.int32)   # (first K entries used)
        self.RT = (<CPhase>self.phases[0]).RT
        for P in self.phases:
            (<CPhase>P).has_last = False

    def points_phase(self, int p):
        """Diagnostics: the phase index of stored point p."""
        return self.Pph[p]

    def lp_state(self):
        """Diagnostics: the LP basis (points, or -(1 + k) for artificial columns), their amounts, the chemical
        potentials (J/mol) and the reduced costs (RT units) of the basis variables."""
        basis = np.asarray(self.basis[:self.K]).copy(); xB = np.asarray(self.xB[:self.K]).copy()
        rc = []
        for var in basis:
            if var < 0:
                rc.append(1e3 - self.dual[-(1 + var)])
            else:
                rc.append(self.PG[var] / self.RT - float(np.dot(np.asarray(self.mu), np.asarray(self.Px[var]))))
        return basis, xB, np.asarray(self.mu) * self.RT, np.array(rc)

    def refresh_points(self):
        """Re-evaluate the stored points' energies after a change of the state variables (their site fractions and
        compositions stay), keeping them and the LP basis as a warm start; the skip rule's state is cleared."""
        cdef int p
        cdef void* pp
        with nogil:
            for p in range(self.npts):
                pp = self.pptr[self.Pph[p]]
                (<CPhase>pp).eval0(&self.Py[p, 0])
                self.PG[p] = (<CPhase>pp).out1[0] if isfinite((<CPhase>pp).out1[0]) else 1e10   # (never chosen)
        self.RT = (<CPhase>self.phases[0]).RT
        for P in self.phases:
            (<CPhase>P).has_last = False

    def set_conditions(self, R, b, mu_fixed, fixed):
        """Condition rows: sum_p n_p R x_p = b (K rows); chemical potentials mu_fixed (J/mol) for the components
        flagged in `fixed` (K + number fixed = C)."""
        cdef int k, K = np.asarray(R).shape[0]
        RR = np.array(R, dtype=float, ndmin=2); bb = np.array(b, dtype=float)
        neg = bb < 0
        RR[neg] *= -1; bb[neg] *= -1   # artificial columns need b >= 0
        # equilibrate: a trace composition row (b ~ 1e-10) would otherwise make the basis matrix nearly singular
        # (the chemical potentials R^T duals do not depend on the row scaling)
        scale = np.where(bb > 0, bb, np.abs(RR).max(axis=1))
        scale = np.where(scale > 0, scale, 1.0)
        RR /= scale[:, None]; bb /= scale
        if K != self.K:
            self.K = K
            self.basis = np.array([-(1 + k) for k in range(self.C)], dtype=np.int32)
        self.R = RR; self.b = bb
        self.RT = (<CPhase>self.phases[0]).RT
        self.mf = np.asarray(mu_fixed, dtype=float) / self.RT
        self.fixed = np.asarray(fixed, dtype=np.int32)

    def export_points(self):
        """The point store and LP basis (to start other solvers over clones of the same phases from it)."""
        cdef int n = self.npts
        return {'n': n, 'Py': np.asarray(self.Py[:n]).copy(), 'Px': np.asarray(self.Px[:n]).copy(),
                'PG': np.asarray(self.PG[:n]).copy(), 'Pph': np.asarray(self.Pph[:n]).copy(),
                'Pnext': np.asarray(self.Pnext[:n]).copy(), 'Pcode': np.asarray(self.Pcode[:n]).copy(),
                'head': np.asarray(self.head).copy(), 'basis': np.asarray(self.basis).copy(), 'K': self.K,
                'skip': [((<CPhase>P).has_last, (<CPhase>P).last_tpd, np.asarray((<CPhase>P).last_mu).copy())
                         for P in self.phases]}

    def import_points(self, d):
        """Replace the point store (and the LP basis, for the same number of condition rows) by an exported one."""
        cdef int n = d['n']
        while self.cap < n:
            self._grow()
        np.asarray(self.Py)[:n] = d['Py']; np.asarray(self.Px)[:n] = d['Px']; np.asarray(self.PG)[:n] = d['PG']
        np.asarray(self.Pph)[:n] = d['Pph']; np.asarray(self.Pnext)[:n] = d['Pnext']; np.asarray(self.Pcode)[:n] = d['Pcode']
        self.npts = n
        self.head = np.array(d['head'], dtype=np.int32)
        if d['K'] == self.K:
            self.basis = np.array(d['basis'], dtype=np.int32)
        # the skip rule's state (valid at other potentials too: it bounds the change by max|delta mu|)
        for P, (has_last, last_tpd, last_mu) in zip(self.phases, d['skip']):
            (<CPhase>P).has_last = has_last; (<CPhase>P).last_tpd = last_tpd
            np.asarray((<CPhase>P).last_mu)[:] = last_mu

    def points(self, int ph):
        """Stored points of phase ph: (y, x, GM) arrays."""
        idx = []
        p = self.head[ph]
        while p >= 0:
            idx.append(p); p = self.Pnext[p]
        D = (<CPhase>self.phases[ph]).D
        return np.asarray(self.Py)[idx, :D].copy(), np.asarray(self.Px)[idx].copy(), np.asarray(self.PG)[idx].copy()

    def _grow(self):
        cap = self.cap * 2
        def g(a, shape, fill, dtype):
            out = np.full(shape, fill, dtype=dtype); out[:self.cap] = np.asarray(a); return out
        self.Py = g(self.Py, (cap, self.Dmax), 0.0, float); self.Px = g(self.Px, (cap, self.C), 0.0, float)
        self.PG = g(self.PG, (cap,), 0.0, float)
        self.Pph = g(self.Pph, (cap,), 0, np.int32); self.Pnext = g(self.Pnext, (cap,), -1, np.int32)
        self.Pcode = g(self.Pcode, (cap,), 0, np.int64)
        self.cap = cap

    cdef int add_point(self, int ph, double* y, double* x, double G) noexcept nogil:
        cdef void* pp = self.pptr[ph]
        cdef int j, i, p, D = (<CPhase>pp).D
        cdef double m
        if not isfinite(G):
            return -1
        for i in range(self.C):
            if not isfinite(x[i]): return -1
        p = self.head[ph]
        while p >= 0:
            m = 0.0
            for j in range(D):
                if fabs(self.Py[p, j] - y[j]) > m: m = fabs(self.Py[p, j] - y[j])
            if m < 1e-7:
                return -1
            p = self.Pnext[p]
        if self.npts >= self.cap:
            with gil:
                self._grow()
        p = self.npts
        for j in range(D): self.Py[p, j] = y[j]
        for i in range(self.C): self.Px[p, i] = x[i]
        self.PG[p] = G; self.Pph[p] = ph; self.Pcode[p] = (<CPhase>pp).pattern(y)
        self.Pnext[p] = self.head[ph]; self.head[ph] = p
        self.npts += 1
        return p

    def add_points(self, int ph, starts):
        """Evaluate and store site-fraction rows of phase ph."""
        cdef CPhase P = <CPhase>self.phases[ph]
        cdef double[:, ::1] Y = np.ascontiguousarray(starts, dtype=float)
        cdef int r, j
        cdef double M
        for r in range(Y.shape[0]):
            P.eval0(&Y[r, 0])
            if P.Mtot < P.m_lo or P.ne > 0: continue   # (constrained phases: only points from pricing are stored)
            for j in range(self.C): self.xbuf[j] = P.mval[j] / P.Mtot
            self.add_point(ph, &Y[r, 0], &self.xbuf[0], P.out1[0])

    cdef inline double cost(self, int j) noexcept nogil:
        cdef double c
        cdef int i
        if j < 0:
            return 1e3
        c = self.PG[j] / self.RT
        for i in range(self.C): c -= self.mf[i] * self.Px[j, i]
        return c

    cdef inline void column(self, int j, double* out) noexcept nogil:
        cdef int k, i
        cdef double s
        for k in range(self.K):
            if j < 0:
                out[k] = 1.0 if k == -(j + 1) else 0.0
            else:
                s = 0.0
                for i in range(self.C): s += self.R[k, i] * self.Px[j, i]
                out[k] = s

    cdef int lp_solve(self, double* obj) noexcept nogil:
        """Revised primal simplex (artificial identity columns at cost 1e3), warm-started from the previous basis.
        Leaves the chemical potentials (RT units) in self.mu: R^T duals + the fixed ones. Returns 0 on success."""
        cdef int K = self.K, C = self.C, it, i, k, j, q, leave, info, nrhs = 1, n = self.K
        cdef double best, t, tmin, s, piv
        cdef char N = b'N', T = b'T'
        cdef bint bland = False
        cdef int resets = 0
        for it in range(20000):
            self.lp_iters += 1
            for k in range(K):
                self.column(self.basis[k], &self.LU[k * K])
            dgetrf(&n, &n, &self.LU[0], &n, &self.ipiv[0], &info)
            if info != 0:
                # (a singular basis: restart from the artificial one, a few times at most)
                if resets >= 3: return 1
                for k in range(K): self.basis[k] = -(1 + k)
                resets += 1; bland = False
                continue
            for i in range(K): self.xB[i] = self.b[i]
            dgetrs(&N, &n, &nrhs, &self.LU[0], &n, &self.ipiv[0], &self.xB[0], &n, &info)
            j = 0
            for i in range(K):
                if self.xB[i] < -1e-9: j = 1
            if j:
                # the warm basis is infeasible for these conditions: restart from the artificial basis
                if resets >= 3: return 2
                for k in range(K): self.basis[k] = -(1 + k)
                resets += 1; bland = False
                continue
            for k in range(K): self.cB[k] = self.cost(self.basis[k])
            memcpy(&self.dual[0], &self.cB[0], K * sizeof(double))
            dgetrs(&T, &n, &nrhs, &self.LU[0], &n, &self.ipiv[0], &self.dual[0], &n, &info)
            for i in range(C):
                s = self.mf[i]
                for k in range(K): s += self.R[k, i] * self.dual[k]
                self.mu[i] = s
            # entering column: most negative reduced cost (first negative, Bland's rule, after many iterations);
            # reduced costs within rounding of zero (relative to the costs' size) do not count
            if it > 50 * K + 100: bland = True
            s = 1.0
            for k in range(K):
                if fabs(self.cB[k]) > s: s = fabs(self.cB[k])
            # (basic variables are skipped: in an ill-conditioned basis their reduced costs are not exactly zero, and
            # one could re-enter in place of itself indefinitely)
            q = 0; best = -1e-10 * s; leave = -2
            for k in range(K):
                s = 1e3 - self.dual[k]
                if s < best and not self.is_basic(-(1 + k)):
                    best = s; q = -(1 + k); leave = -1
                    if bland: break
            if leave == -2 or not bland:
                for j in range(self.npts):
                    s = self.PG[j] / self.RT
                    for i in range(C): s -= self.mu[i] * self.Px[j, i]
                    if s < best and not self.is_basic(j):
                        best = s; q = j; leave = -1
                        if bland: break
            if leave == -2:
                obj[0] = 0.0
                for k in range(K): obj[0] += self.cB[k] * self.xB[k]
                return 0
            self.column(q, &self.tmp[0])
            dgetrs(&N, &n, &nrhs, &self.LU[0], &n, &self.ipiv[0], &self.tmp[0], &n, &info)
            # pivots must not be small relative to the column, else the next basis is (nearly) singular
            piv = 1e-11
            for i in range(K):
                if 1e-9 * fabs(self.tmp[i]) > piv: piv = 1e-9 * fabs(self.tmp[i])
            leave = -1; tmin = INFINITY
            for i in range(K):
                if self.tmp[i] > piv:
                    t = (self.xB[i] if self.xB[i] > 0 else 0.0) / self.tmp[i]
                    if t < tmin:
                        tmin = t; leave = i
            if leave < 0:
                return 3
            if bland:
                # Bland's rule also for the leaving variable: among the (near-)ties, the first variable in the order
                # in which entering ones are scanned (artificial columns 0..K-1, then points), else degenerate
                # pivots can cycle
                for i in range(K):
                    if self.tmp[i] > piv:
                        t = (self.xB[i] if self.xB[i] > 0 else 0.0) / self.tmp[i]
                        if t <= tmin + 1e-12 * (1.0 + tmin) and self.bland_rank(self.basis[i]) < self.bland_rank(self.basis[leave]):
                            leave = i
            self.basis[leave] = q
        return 4

    cdef inline bint is_basic(self, int var) noexcept nogil:
        cdef int k
        for k in range(self.K):
            if self.basis[k] == var: return True
        return False

    cdef inline int bland_rank(self, int var) noexcept nogil:
        """Position of an LP variable (artificial column -(1 + k), or a point) in the entering scan's order."""
        return -(1 + var) if var < 0 else self.K + var

    cdef int _polish_start(self) noexcept nogil:
        """The current LP solution's points as the polish's starting state (pol_*, mu). Returns their number, or -1
        if an artificial column is still in use."""
        cdef int C = self.C, K = self.K, Dm = self.Dmax, k, p, j, c, na = 0
        for c in range(C): self.pol_mu[c] = self.mu[c]
        for k in range(K):
            if self.xB[k] > 1e-14:   # (basis columns at zero amount make the system singular)
                p = self.basis[k]
                if p < 0:
                    if self.xB[k] > 1e-9: return -1
                    continue   # (an artificial column at a negligible amount: a redundant row)
                self.pol_ph[na] = self.Pph[p]; self.pol_n[na] = self.xB[k]
                for j in range((<CPhase>self.pptr[self.Pph[p]]).D): self.pol_y[na * Dm + j] = self.Py[p, j]
                na += 1
        return na

    cdef int polish(self, int max_it=40) noexcept nogil:
        cdef int na = self._polish_start()
        if na <= 0:
            return 0
        if self._polish_core(na, max_it, False) == 1:
            return 1
        if self.polish_fail != 4:
            return 0
        # an amount went negative: the LP basis holds a column the converged state does not need, e.g. one that only
        # made up for two columns of one phase on either side of its minimum (mixed by the LP, merged by the
        # polish). Left unpolished, such an LP result has a spurious extra composition set and a trace phase, within
        # the certification tolerance in energy. Again from the same start, dropping points whose amounts reach
        # zero; the pricing round that follows certifies the result (or finds what a dropped point was needed for).
        na = self._polish_start()
        if na <= 0:
            return 0
        return self._polish_core(na, max_it, True)

    cdef int _polish_core(self, int na, int max_it, bint allow_drop) noexcept nogil:
        """Newton on the equilibrium conditions from the starting state in pol_* (see _polish_start). Unknowns: mu and
        the amounts n. Equations: TPD_a(mu) = 0 (each point at its local TPD minimum), the condition rows
        sum_a n_a R x_a(mu) = b, and the fixed potentials. By the envelope theorem dTPD_a/dmu = -x_a; dx_a/dmu comes
        from the phase Hessian at the minimum. Adds the converged points as columns and leaves their potentials in
        self.mu; returns 1 if it converged, 0 if it gave up. With allow_drop, a point whose amount reaches zero is
        dropped (the step is cut there), as when a condition changes continuously."""
        cdef int C = self.C, K = self.K, Dm = self.Dmax, k, p, a, bb, c, i, j, it, w, N, info, nrhs = 1, row, adrop
        cdef void* pp
        cdef double G = 0.0, t = 0.0, res, last_res = INFINITY, sc, s, alpha, rcond
        cdef int rank, lw
        cdef bint merged
        cdef double* py = &self.pol_y[0]
        cdef double* px = &self.pol_x[0]
        cdef double* pS = &self.pol_S[0]
        cdef double* pJ = &self.pol_J[0]
        cdef double* pF = &self.pol_F[0]
        if allow_drop:
            # at most K points can have independent amounts (phase rule; e.g. one row, N, with all but one chemical
            # potential fixed): keep the K largest
            while na > K:
                adrop = 0
                for a in range(1, na):
                    if self.pol_n[a] < self.pol_n[adrop]: adrop = a
                for a in range(adrop, na - 1):
                    self.pol_ph[a] = self.pol_ph[a + 1]; self.pol_n[a] = self.pol_n[a + 1]
                    memcpy(&py[a * Dm], &py[(a + 1) * Dm], Dm * sizeof(double))
                na -= 1
        for it in range(max_it):
            for a in range(na):
                pp = self.pptr[self.pol_ph[a]]
                if not (<CPhase>pp).price(&self.pol_mu[0], &py[a * Dm], &self.ybuf[0], &px[a * C], &G, &t):
                    self.polish_fail = 1; return 0
                if not isfinite(G): return 0
                memcpy(&py[a * Dm], &self.ybuf[0], (<CPhase>pp).D * sizeof(double))
                self.pol_G[a] = G; self.pol_t[a] = t
                if (<CPhase>pp).sens(&pS[a * C * C]) < 0:
                    self.polish_fail = 2; return 0
            # merge composition sets that met, and ordering variants of one state (site fractions permuted among
            # equivalent sublattices: priced at the same potentials they have the same composition and energy, and
            # as separate sets they make the system singular)
            w = 0
            for a in range(na):
                merged = False
                pp = self.pptr[self.pol_ph[a]]
                for bb in range(w):
                    if self.pol_ph[bb] == self.pol_ph[a]:
                        s = 0.0
                        for j in range((<CPhase>pp).D):
                            if fabs(py[bb * Dm + j] - py[a * Dm + j]) > s: s = fabs(py[bb * Dm + j] - py[a * Dm + j])
                        if s >= 1e-6 and fabs(self.pol_G[bb] - self.pol_G[a]) < 1e-9 * (1.0 + fabs(self.pol_G[a])):
                            s = 0.0
                            for i in range(C):
                                if fabs(px[bb * C + i] - px[a * C + i]) > s: s = fabs(px[bb * C + i] - px[a * C + i])
                            s *= 1e3   # (compositions within 1e-9)
                        if s < 1e-6:
                            self.pol_n[bb] += self.pol_n[a]; merged = True; break
                if not merged:
                    if w != a:
                        self.pol_ph[w] = self.pol_ph[a]; self.pol_n[w] = self.pol_n[a]; self.pol_G[w] = self.pol_G[a]
                        self.pol_t[w] = self.pol_t[a]
                        memcpy(&py[w * Dm], &py[a * Dm], Dm * sizeof(double))
                        memcpy(&px[w * C], &px[a * C], C * sizeof(double))
                        memcpy(&pS[w * C * C], &pS[a * C * C], C * C * sizeof(double))
                    w += 1
            na = w
            # residuals: TPD of each point, the condition rows, the fixed potentials
            res = 0.0
            for a in range(na):
                pF[a] = self.pol_t[a]
                if fabs(pF[a]) > res: res = fabs(pF[a])
            for k in range(K):
                s = -self.b[k]
                for a in range(na):
                    for i in range(C): s += self.pol_n[a] * self.R[k, i] * px[a * C + i]
                pF[na + k] = s
                if fabs(s) > res: res = fabs(s)   # (rows are scaled to unit right-hand sides)
            row = na + K
            for i in range(C):
                if self.fixed[i]:
                    pF[row] = self.pol_mu[i] - self.mf[i]
                    if fabs(pF[row]) > res: res = fabs(pF[row])
                    row += 1
            # (rows are compared relative to their right-hand sides, so trace amounts converge too)
            if res < 1e-11 or (res < 1e-8 and res > 0.1 * last_res):
                break
            last_res = res
            # Jacobian (column-major), unknowns [mu (C), n (na)]
            N = na + C
            for i in range(N * N): pJ[i] = 0.0
            for a in range(na):
                for i in range(C):
                    pJ[i * N + a] = -px[a * C + i]
                for k in range(K):
                    s = 0.0
                    for i in range(C): s += self.R[k, i] * px[a * C + i]
                    pJ[(C + a) * N + na + k] = s
            for k in range(K):
                for j in range(C):
                    s = 0.0
                    for a in range(na):
                        for i in range(C): s += self.pol_n[a] * self.R[k, i] * pS[a * C * C + i * C + j]
                    pJ[j * N + na + k] = s
            row = na + K
            for i in range(C):
                if self.fixed[i]:
                    pJ[i * N + row] = 1.0
                    row += 1
            for i in range(N): pF[i] = -pF[i]
            memcpy(&self.pol_J2[0], pJ, N * N * sizeof(double)); memcpy(&self.pol_F2[0], pF, N * sizeof(double))
            dgesv(&N, &nrhs, pJ, &N, &self.pol_ip[0], pF, &N, &info)
            if info != 0:
                # dependent rows (e.g. compositions on a charge-neutral manifold): least squares
                rcond = 1e-10
                lw = self.pol_work.shape[0]
                dgelss(&N, &N, &nrhs, &self.pol_J2[0], &N, &self.pol_F2[0], &N, &self.pol_sv[0], &rcond, &rank,
                       &self.pol_work[0], &lw, &info)
                if info != 0:
                    self.polish_fail = 3; return 0
                memcpy(pF, &self.pol_F2[0], N * sizeof(double))
            # no step along gauge directions (the Jacobian is singular there; a near-singular solve can return a huge
            # component along them)
            for k in range(self.ngauge):
                s = 0.0
                for i in range(C): s += self.gauge_mv[k, i] * pF[i]
                for i in range(C): pF[i] -= s * self.gauge_mv[k, i]
            s = 0.0
            for i in range(C):
                if fabs(pF[i]) > s: s = fabs(pF[i])
            sc = 5.0 / s if s > 5.0 else 1.0   # at most 5 RT per step in any potential
            adrop = -1
            if allow_drop:
                # cut the step where the first amount reaches zero, and drop that point
                for a in range(na):
                    if self.pol_n[a] + sc * pF[C + a] < 0 and pF[C + a] < 0:
                        alpha = self.pol_n[a] / -pF[C + a]
                        if alpha < sc:
                            sc = alpha; adrop = a
            for i in range(C): self.pol_mu[i] += sc * pF[i]
            for a in range(na):
                self.pol_n[a] += sc * pF[C + a]
                if self.pol_n[a] < -1e-9 and not allow_drop:
                    self.polish_fail = 4; return 0
            if adrop >= 0:
                if na == 1:
                    self.polish_fail = 5; return 0
                for a in range(adrop, na - 1):
                    self.pol_ph[a] = self.pol_ph[a + 1]; self.pol_n[a] = self.pol_n[a + 1]
                    memcpy(&py[a * Dm], &py[(a + 1) * Dm], Dm * sizeof(double))
                na -= 1
                last_res = INFINITY
        else:
            self.polish_fail = 6; return 0
        self.pol_na = na
        for a in range(na):
            if self.pol_n[a] < 0: self.pol_n[a] = 0.0
            self.add_point(self.pol_ph[a], &py[a * Dm], &px[a * C], self.pol_G[a])
        # certify at these potentials (the LP's own duals are not unique when a single point sits at the composition)
        for c in range(C): self.mu[c] = self.pol_mu[c]
        self.npolish += 1
        return 1

    cdef void try_start(self, int ph, double* y0, double* mu, double* best_tpd, int* all_conv, bint keep) noexcept nogil:
        cdef double G = 0.0, tpd = 0.0
        cdef int conv
        conv = (<CPhase>self.pptr[ph]).price(mu, y0, &self.ybuf[0], &self.xbuf[0], &G, &tpd)
        if not conv:
            all_conv[0] = 0
        if not isfinite(G):
            tpd = INFINITY
        self.priced += 1
        # every distinct minimum below the tangent plane becomes a column (so one branch cannot hide another);
        # seed minima are kept anyway, as warm starts for later conditions
        if keep or tpd < -self.tol:
            if self.add_point(ph, &self.ybuf[0], &self.xbuf[0], G) >= 0 and tpd < -self.tol:
                self.seed_added = True
        if tpd < best_tpd[0]:
            best_tpd[0] = tpd
            memcpy(&self.ybest[0], &self.ybuf[0], (<CPhase>self.pptr[ph]).D * sizeof(double))

    cdef void _price_from_points(self, int ph, int k, int rounds, int ncodes, long* codes, int* code_pt,
                                 double* code_tpd, double* mu, double* best_tpd, int* all_conv) noexcept nogil:
        """Starts from stored points: the best one (k), the phase's points in the LP solution, the best of the other
        ordering patterns, and in the first round a disordered start at k's composition."""
        cdef int si, sj, p
        cdef double t
        self.try_start(ph, &self.Py[k, 0], mu, best_tpd, all_conv, False)
        for si in range(ncodes):
            for sj in range(si + 1, ncodes):
                if code_tpd[sj] < code_tpd[si]:
                    t = code_tpd[si]; code_tpd[si] = code_tpd[sj]; code_tpd[sj] = t
                    p = code_pt[si]; code_pt[si] = code_pt[sj]; code_pt[sj] = p
        for si in range(self.K):
            p = self.basis[si]
            if p >= 0 and p != k and self.Pph[p] == ph and self.xB[si] > 1e-12:
                self.try_start(ph, &self.Py[p, 0], mu, best_tpd, all_conv, False)
        for si in range(min(ncodes, 6 if rounds == 1 else self.max_late)):
            if code_pt[si] != k and self.Pcode[code_pt[si]] != self.Pcode[k]:
                self.try_start(ph, &self.Py[code_pt[si], 0], mu, best_tpd, all_conv, False)
        if rounds == 1:
            (<CPhase>self.pptr[ph]).composition_start(&self.Px[k, 0], &self.ystart[0])
            self.try_start(ph, &self.ystart[0], mu, best_tpd, all_conv, False)

    cdef bint _pricing_round(self, int rounds, bint use_seeds, bint polished, bint* no_polish) noexcept nogil:
        """Price every phase at self.mu (with the packed seeds if use_seeds); returns whether anything new below the
        tangent plane was found."""
        cdef int C = self.C, K = self.K, ph, p, k, i, si, sj, ncodes, all_conv
        cdef double best_tpd, maxdmu, t
        cdef void* pp
        cdef bint added = False
        cdef long codes[64]
        cdef int code_pt[64]
        cdef double code_tpd[64]
        cdef double* mu = &self.mu[0]
        self.seed_added = False
        self.cert_kind = 0
        for ph in range(self.nph):
            pp = self.pptr[ph]
            if (<CPhase>pp).stoich:
                continue
            # starts: the phase's stored point with the lowest TPD; the best point of each other ordering pattern
            # (dominant species per sublattice) within `prune` of the tangent plane (6 at most) in the first round,
            # within `prune_late` (max_late at most) later; the phase's points in the LP solution; in the first round
            # also a disordered start at the composition of the best point, and the seeds
            k = -1; best_tpd = INFINITY; ncodes = 0
            p = self.head[ph]
            while p >= 0:
                t = self.PG[p] / self.RT
                for i in range(C): t -= mu[i] * self.Px[p, i]
                if t < best_tpd:
                    best_tpd = t; k = p
                if t < (self.prune if rounds == 1 else self.prune_late):
                    for si in range(ncodes):
                        if codes[si] == self.Pcode[p]:
                            if t < code_tpd[si]:
                                code_tpd[si] = t; code_pt[si] = p
                            break
                    else:
                        if ncodes < 64:
                            codes[ncodes] = self.Pcode[p]; code_tpd[ncodes] = t; code_pt[ncodes] = p; ncodes += 1
                p = self.Pnext[p]
            # skip rule: a phase's minimum TPD can fall by at most max|delta mu| (mole fractions are in [0, 1] and sum
            # to 1); the minimum recorded at the last pricing is only the lowest of the basins priced then, so the
            # phase's stored points (minima of its other basins, relaxed under earlier conditions) must also be well
            # above the plane
            if (<CPhase>pp).has_last and best_tpd > self.skip_margin:
                maxdmu = 0.0
                for i in range(C):
                    t = fabs(mu[i] - (<CPhase>pp).last_mu[i])
                    if t > maxdmu: maxdmu = t
                if (<CPhase>pp).last_tpd - maxdmu > self.skip_margin:
                    self.skipped += 1
                    continue
            if k < 0 and not (rounds == 1 and use_seeds):
                continue
            best_tpd = INFINITY; all_conv = 1
            if k >= 0:
                self._price_from_points(ph, k, rounds, ncodes, codes, code_pt, code_tpd, mu, &best_tpd, &all_conv)
            if rounds == 1 and use_seeds:
                # (constrained phases have no stored points until their seeds are priced)
                for si in range(self.seed_off[ph], self.seed_off[ph + 1]):
                    # seeds already far above the tangent plane are not worth a local minimization
                    (<CPhase>pp).eval0(&self.seedY[si, 0])
                    for i in range(C): self.xbuf[i] = (<CPhase>pp).mval[i] / (<CPhase>pp).Mtot
                    t = (<CPhase>pp).out1[0] / self.RT
                    for i in range(C): t -= mu[i] * self.xbuf[i]
                    if not t < self.seed_prune:
                        continue
                    self.try_start(ph, &self.seedY[si, 0], mu, &best_tpd, &all_conv, True)
            if polished and best_tpd < -self.tol:
                if self.cert_kind == 0 or best_tpd < self.cert_tpd:
                    # (the most negative minimum, for _polish_augment)
                    memcpy(&self.ycert[0], &self.ybest[0], (<CPhase>pp).D * sizeof(double))
                    self.cert_kind = 1; self.cert_phase = ph; self.cert_tpd = best_tpd
                # certification failed (the point may already be a column the LP cannot use yet): continue with the
                # LP's own potentials
                added = True; no_polish[0] = True
            if all_conv:  # only a converged minimum may justify skipping this phase later
                (<CPhase>pp).has_last = True; (<CPhase>pp).last_tpd = best_tpd
                for i in range(C): (<CPhase>pp).last_mu[i] = mu[i]
            else:
                (<CPhase>pp).has_last = False
        if polished and not added:
            # the stored points (stoichiometric phases among them) were checked by the LP, not at these potentials
            for p in range(self.npts):
                t = self.PG[p] / self.RT
                for i in range(C): t -= mu[i] * self.Px[p, i]
                if t < -self.tol:
                    self.cert_kind = 2; self.cert_phase = self.Pph[p]; self.cert_tpd = t
                    memcpy(&self.ycert[0], &self.Py[p, 0], (<CPhase>self.pptr[self.Pph[p]]).D * sizeof(double))
                    added = True; no_polish[0] = True
                    break
        if self.seed_added and not added:
            self.cert_kind = 3
        return added or self.seed_added

    def set_gauge(self, W):
        """Orthonormal rows w with w.x = 0 for every admissible composition of every phase (e.g. the elements' charges
        when every phase is charge neutral and each element has one valence): the chemical potentials are defined
        only up to multiples of them, and an artificial column left in the LP basis pins those at arbitrary, huge
        values. Such components are removed after every LP solve (no reduced cost changes)."""
        self.gauge = np.array(W, dtype=float).reshape(-1, self.C)
        self.ngauge = self.gauge.shape[0]
        self.gauge_mv = self.gauge if self.ngauge else np.zeros((1, self.C))

    cdef void gauge_fix(self) noexcept nogil:
        cdef int k, i
        cdef double s
        for k in range(self.ngauge):
            s = 0.0
            for i in range(self.C): s += self.gauge_mv[k, i] * self.mu[i]
            for i in range(self.C): self.mu[i] -= s * self.gauge_mv[k, i]

    def _pack_seeds(self, seeds):
        """Seeds (per phase, rows of site fractions) into one array, for pricing without the GIL."""
        cdef int i, r, j, row = 0
        cdef double[:, ::1] Si
        rows = [np.ascontiguousarray(np.asarray(sd, dtype=float).reshape(-1, (<CPhase>self.phases[i]).D))
                for i, sd in enumerate(seeds)]
        total = sum(x.shape[0] for x in rows)
        cdef int[::1] off = np.zeros(self.nph + 1, dtype=np.int32)
        cdef double[:, ::1] Y = np.zeros((max(total, 1), self.Dmax))
        for i in range(self.nph):
            Si = rows[i]
            for r in range(Si.shape[0]):
                for j in range(Si.shape[1]): Y[row, j] = Si[r, j]
                row += 1
            off[i + 1] = row
        self.seedY = Y; self.seed_off = off

    def cert_description(self):
        """What made the last certification fail (diagnostics)."""
        names = [(<CPhase>p).name for p in self.phases]
        return {0: None, 1: ('phase', names[self.cert_phase] if self.cert_phase >= 0 else None, self.cert_tpd),
                2: ('stored point', names[self.cert_phase] if self.cert_phase >= 0 else None, self.cert_tpd),
                3: ('seed',)}.get(self.cert_kind)

    def _polished_result(self, int rounds):
        cdef int a, C = self.C, na = self.pol_na
        phs = [self.pol_ph[a] for a in range(na)]
        Y = [np.asarray(self.pol_y[a * self.Dmax:a * self.Dmax + (<CPhase>self.phases[phs[a]]).D]).copy() for a in range(na)]
        X = [np.asarray(self.pol_x[a * C:(a + 1) * C]).copy() for a in range(na)]
        return (0, phs, Y, X, [self.pol_n[a] for a in range(na)], [self.pol_G[a] for a in range(na)],
                np.asarray(self.mu).copy() * self.RT, rounds)

    def _lp_result(self, int rounds):
        cdef int k, p
        phs = []; Y = []; X = []; n = []; G = []
        for k in range(self.K):
            if self.xB[k] > 1e-300:
                p = self.basis[k]
                if p < 0:
                    # an artificial column still carries part of the prescribed amounts (rows are scaled to unit
                    # right-hand sides, so trace amounts count fully); at a negligible amount it marks a redundant row
                    if self.xB[k] > 1e-9:
                        return (5, None, None, None, None, None, None, rounds)
                    continue
                phs.append(self.Pph[p]); Y.append(np.asarray(self.Py[p, :(<CPhase>self.phases[self.Pph[p]]).D]).copy())
                X.append(np.asarray(self.Px[p]).copy()); n.append(self.xB[k]); G.append(self.PG[p])
        return (0, phs, Y, X, n, G, np.asarray(self.mu).copy() * self.RT, rounds)

    cdef void _set_conditions_c(self, double* R, double* b, double* mu_fixed, int* fixed, int K) noexcept nogil:
        """set_conditions without the GIL (into the solver's own buffers)."""
        cdef int k, i
        cdef double sc, m
        if K != self.K:
            self.K = K
            for k in range(self.C): self.basis[k] = -(1 + k)
        for k in range(K):
            sc = 1.0
            if b[k] < 0: sc = -1.0
            m = fabs(b[k])
            if m <= 0:
                for i in range(self.C):
                    if fabs(R[k * self.C + i]) > m: m = fabs(R[k * self.C + i])
                if m <= 0: m = 1.0
            for i in range(self.C): self.Rbuf[k, i] = sc * R[k * self.C + i] / m
            self.bbuf[k] = sc * b[k] / m
        for i in range(self.C):
            self.mfbuf[i] = mu_fixed[i] / self.RT
            self.fixedbuf[i] = fixed[i]
        self.R = self.Rbuf; self.b = self.bbuf; self.mf = self.mfbuf; self.fixed = self.fixedbuf

    cdef bint _polish_augment(self) noexcept nogil:
        """Newton from the last polished state plus the point that failed its certification (ycert, of cert_phase)
        at zero amount; returns whether it converged."""
        cdef int na = self.pol_na, Dm = self.Dmax
        if na <= 0 or na + 1 > self.K:
            return False
        self.pol_ph[na] = self.cert_phase; self.pol_n[na] = 0.0
        memcpy(&self.pol_y[na * Dm], &self.ycert[0], (<CPhase>self.pptr[self.cert_phase]).D * sizeof(double))
        return self._polish_core(na + 1, 40, True) == 1

    cdef bint _round_step(self, int rounds, bint use_seeds, bint* no_polish, bint* added_out) noexcept nogil:
        """One round after the LP: polish (unless the last certification failed), price (certifying a polished
        state), and Newton again with any minimum found below a polished tangent plane. Returns whether the state
        is polished; added_out: whether anything new was found (if not, the condition is solved)."""
        cdef int aug = 0, C = self.C
        cdef bint polished = False, skipped, added
        skipped = no_polish[0]
        if self.use_polish and not no_polish[0]:
            # finish the current phase set by Newton; the pricing below then certifies it (or finds more)
            self.npolish_try += 1
            polished = self.polish() == 1
        no_polish[0] = False
        added = self._pricing_round(rounds, use_seeds, polished, no_polish)
        # a minimum below the polished tangent plane: Newton again with it as a further composition set (from
        # zero amount). The LP alone can stall here: when the polished set is a single point at the overall
        # composition, the LP is degenerate and can keep choosing other dual solutions under which the new
        # point does not help, without the neighbouring points the true phase set needs.
        while added and polished and (self.cert_kind == 1 or self.cert_kind == 2) and aug < C:
            aug += 1
            if not self._polish_augment():
                polished = False
                break
            no_polish[0] = False
            added = self._pricing_round(rounds, False, True, no_polish)
        if not added and not polished and skipped and self.use_polish:
            # nothing new at the LP's potentials, but the polish was skipped this round (after a failed
            # certification): finish by Newton and certify that, rather than end on the LP's mixture of points
            # (which can hold spurious composition sets within the tolerance)
            self.npolish_try += 1
            if self.polish() == 1:
                polished = True
                added = self._pricing_round(rounds, False, True, no_polish)
                if added:
                    polished = False
        added_out[0] = added
        return polished

    cdef int _solve_c(self, bint use_seeds, int max_rounds, bint* polished_out, int* rounds_out) noexcept nogil:
        """The column-generation rounds (see solve()); returns the status."""
        cdef int rounds = 0, st = 0
        cdef double obj = 0.0
        cdef bint added = False, polished = False, no_polish = False
        for rounds in range(1, max_rounds + 1):
            st = self.lp_solve(&obj)
            if st != 0:
                break
            self.gauge_fix()
            polished = self._round_step(rounds, use_seeds and rounds == 1, &no_polish, &added)
            if not added:
                break
        if st == 0 and added:
            st = 6   # the round limit without a certified result
        polished_out[0] = polished; rounds_out[0] = rounds
        self.last_polished = polished
        return st

    def solve_sequence(self, R, b, mu_fixed, fixed, seeds=None, int max_rounds=60):
        """Solve a sequence of conditions (rows R[t] (K x C), b[t], fixed potentials mu_fixed[t] (J/mol) for the
        components flagged in fixed[t]) in order, warm-started, in one stretch without the GIL; seeds as in solve(),
        for the first. Returns arrays: status, number of points, and per point their phase indices, amounts,
        compositions, site fractions (padded), GM, and the chemical potentials (J/mol)."""
        cdef double[:, :, ::1] Rv = np.ascontiguousarray(R, dtype=float)
        cdef double[:, ::1] bv = np.ascontiguousarray(b, dtype=float)
        cdef double[:, ::1] mfv = np.ascontiguousarray(mu_fixed, dtype=float)
        cdef int[:, ::1] fxv = np.ascontiguousarray(fixed, dtype=np.int32)
        cdef int n = Rv.shape[0], K = Rv.shape[1], C = self.C, Dm = self.Dmax, t, a, k, p, i, j, st, na, rounds
        cdef bint polished, use_seeds = seeds is not None
        status = np.zeros(n, dtype=np.int32); nout = np.zeros(n, dtype=np.int32)
        phs = np.full((n, C), -1, dtype=np.int32); amt = np.zeros((n, C)); X = np.zeros((n, C, C))
        Y = np.full((n, C, Dm), np.nan); G = np.zeros((n, C)); mu = np.zeros((n, C))
        cdef int[::1] status_v = status, nout_v = nout
        cdef int[:, ::1] phs_v = phs
        cdef double[:, ::1] amt_v = amt, G_v = G, mu_v = mu
        cdef double[:, :, ::1] X_v = X, Y_v = Y
        self.RT = (<CPhase>self.phases[0]).RT
        if use_seeds:
            self._pack_seeds(seeds)
        with nogil:
            for t in range(n):
                self._set_conditions_c(&Rv[t, 0, 0], &bv[t, 0], &mfv[t, 0], &fxv[t, 0], K)
                st = self._solve_c(use_seeds and t == 0, max_rounds, &polished, &rounds)
                na = 0
                if st == 0 and polished:
                    for a in range(self.pol_na):
                        phs_v[t, na] = self.pol_ph[a]; amt_v[t, na] = self.pol_n[a]; G_v[t, na] = self.pol_G[a]
                        for i in range(C): X_v[t, na, i] = self.pol_x[a * C + i]
                        for j in range((<CPhase>self.pptr[self.pol_ph[a]]).D): Y_v[t, na, j] = self.pol_y[a * Dm + j]
                        na += 1
                elif st == 0:
                    self.nunpolished += 1
                    for k in range(self.K):
                        if self.xB[k] > 1e-300:
                            p = self.basis[k]
                            if p < 0:
                                if self.xB[k] > 1e-9:
                                    st = 5
                                    break
                                continue
                            phs_v[t, na] = self.Pph[p]; amt_v[t, na] = self.xB[k]; G_v[t, na] = self.PG[p]
                            for i in range(C): X_v[t, na, i] = self.Px[p, i]
                            for j in range((<CPhase>self.pptr[self.Pph[p]]).D): Y_v[t, na, j] = self.Py[p, j]
                            na += 1
                status_v[t] = st; nout_v[t] = na
                for i in range(C): mu_v[t, i] = self.mu[i] * self.RT
        return status, nout, phs, amt, X, Y, G, mu

    def solve(self, seeds=None, int max_rounds=60):
        """Column generation for the current conditions. seeds: per phase, an array of start configurations to price
        in the first round (the first conditions at given state variables), or None.
        Returns (status, phases, Y rows, X rows, amounts, GM rows, mu (J/mol), rounds); status 0 on success. The
        rounds run without the GIL, so solvers over separate phase objects can run in parallel threads."""
        cdef int rounds = 0, st = 0
        cdef bint polished = False, use_seeds = seeds is not None
        self.RT = (<CPhase>self.phases[0]).RT
        if use_seeds:
            self._pack_seeds(seeds)
        with nogil:
            st = self._solve_c(use_seeds, max_rounds, &polished, &rounds)
        if st != 0:
            return (st, None, None, None, None, None, None, rounds)
        self.last_polished = polished
        self.cert_log = self.cert_description()
        if polished:
            return self._polished_result(rounds)
        return self._lp_result(rounds)

    def polish_from_state(self, R, b, mu_fixed, fixed):
        """Newton from the last polished state under new conditions (which become the current ones), dropping points
        whose amounts reach zero. Returns 1 if it converged. For following a condition continuously."""
        if self.pol_na <= 0:
            return 0
        self.set_conditions(R, b, mu_fixed, fixed)
        return self._polish_core(self.pol_na, 40, True)

    def certify(self):
        """After a converged polish: one pricing round at its potentials. Returns the result, or None if anything
        below the tangent plane was found."""
        cdef bint no_polish = False
        if self._pricing_round(2, False, True, &no_polish):
            self.cert_log = self.cert_description()
            return None
        return self._polished_result(0)

    def finish(self, R, b, mu_fixed, fixed):
        """From the current state (the polished one if the last solve ended with one, else the LP solution), Newton
        under other condition rows and fixed potentials (e.g. a fixed chemical potential, after a solve with a trial
        mole fraction in its place), then one certifying pricing round at the resulting potentials. With fewer rows
        than composition sets, each choice of as many sets is tried (largest amounts first). Returns the result, or
        None if no choice converges and certifies. The previous conditions are restored."""
        import itertools
        cdef int na, ok, a, j, Dm = self.Dmax, C = self.C
        cdef bint no_polish = False, added
        cdef int K0 = self.K
        cdef double[:, ::1] R0 = self.R
        cdef double[::1] b0 = self.b, mf0 = self.mf
        cdef int[::1] fixed0 = self.fixed
        na = self.pol_na if self.last_polished else self._polish_start()
        if na <= 0:
            return None
        start_ph = np.asarray(self.pol_ph[:na]).copy(); start_n = np.asarray(self.pol_n[:na]).copy()
        start_y = np.asarray(self.pol_y[:na * Dm]).copy(); start_mu = np.asarray(self.pol_mu).copy()
        RR = np.array(R, dtype=float, ndmin=2); bb = np.array(b, dtype=float)
        neg = bb < 0
        RR[neg] *= -1; bb[neg] *= -1
        scale = np.where(bb > 0, bb, np.abs(RR).max(axis=1))
        scale = np.where(scale > 0, scale, 1.0)
        RR /= scale[:, None]; bb /= scale
        self.K = RR.shape[0]; self.R = RR; self.b = bb
        self.mf = np.asarray(mu_fixed, dtype=float) / self.RT; self.fixed = np.asarray(fixed, dtype=np.int32)
        order = list(np.argsort(-start_n))
        choices = [tuple(sorted(c)) for c in itertools.combinations(order, min(self.K, na))] if na > self.K else [tuple(range(na))]
        try:
            self.finish_reason = None
            for choice in choices[:16]:
                for a, src in enumerate(choice):
                    self.pol_ph[a] = start_ph[src]; self.pol_n[a] = start_n[src]
                    for j in range(Dm): self.pol_y[a * Dm + j] = start_y[src * Dm + j]
                for j in range(C): self.pol_mu[j] = start_mu[j]
                ok = self._polish_core(len(choice), 40, True)
                if not ok:
                    self.finish_reason = ('polish', self.polish_fail, len(choice))
                    continue
                no_polish = False
                added = self._pricing_round(2, False, True, &no_polish)
                if added:
                    self.finish_reason = ('certify', self.cert_description())
                    continue
                return self._polished_result(0)
            return None
        finally:
            self.K = K0; self.R = R0; self.b = b0; self.mf = mf0; self.fixed = fixed0


ctypedef void (*pricer_fn)(int n, const int* jobs, const int* job_phase, const int* job_cond, const double* mu,
                          const double* y0, int ystride, double* y_out, double* x_out, double* G_out, double* tpd_out,
                          int* conv, int* evals, const int* di, const double* dd, const int* ip, const double* dp,
                          int C, double RT, const double* sv) noexcept nogil

cdef int STAGED = 1 << 29   # LP basis entries >= STAGED name a point staged in the current round (STAGED + its index)


cdef class TileSolver(Solver):
    """Column generation for a tile of conditions in lockstep rounds over one shared point store (the scheme a
    batched, GPU-style solver would follow). The first condition is solved on its own, with the seeds; then in each
    round every unfinished condition takes one step of Solver's rounds (LP from its own warm basis, polish, pricing
    with its own skip state). The points a condition adds during a round are its own until the round ends, as if the
    conditions of a round ran concurrently; then they join the shared store, duplicates dropped, for the next round.
    Records the work done per round (stats)."""
    cdef int[:, ::1] t_basis
    cdef double[:, ::1] t_mu, t_ltpd
    cdef double[:, :, ::1] t_lmu
    cdef unsigned char[:, ::1] t_has
    cdef unsigned char[::1] t_nopol
    cdef double[:, ::1] Sy, Sx
    cdef double[::1] SG
    cdef int[::1] Sph, Smap, head0
    cdef int nst, scap
    cdef public list stats
    # batched rounds (solve_tile_batched): per-condition state kept between the sub-rounds of a round
    cdef double[:, ::1] t_xB, t_pol_y, t_pol_x, t_pol_n, t_pol_mu, t_pol_G, t_ycert
    cdef int[:, ::1] t_pol_ph
    cdef int[::1] t_pol_na, t_cert_kind, t_cert_phase, t_aug, t_kind
    cdef double[::1] t_cert_tpd
    cdef unsigned char[::1] t_polished, t_skipped, t_added
    # a condition's own points of the round (staged): a chain per condition, in insertion order
    cdef int[::1] Shead, Stail, Snext, Sown, Sidx
    cdef long[::1] Scode
    cdef int[::1] own_s       # the staged index of each store entry above the shared ones while a condition is loaded
    # pricing jobs: (condition, phase, start) and their results; per (condition, phase) the range of its jobs
    cdef int nj, jcap, ncp, cpcap
    cdef int[::1] j_cond, j_phase, j_conv, cp_phase, cp_start, cp_end, cp_first, cp_last
    cdef double[:, ::1] j_y0, j_y, j_x
    cdef double[::1] j_G, j_tpd
    cdef public list job_batches
    # an external batched pricer (cg_price.h compiled for this system): a C function, the phases it prices, and
    # its phase descriptions
    cdef pricer_fn pricer
    cdef unsigned char[::1] pricer_ok
    cdef int[::1] pr_di, pr_ip, pr_jobs, pr_evals
    cdef double[::1] pr_dd, pr_dp, pr_sv
    cdef double pr_RT
    cdef public object pricer_keep
    cdef public long device_jobs
    cdef public object pricer_cb, cb_error   # a Python batched pricer (e.g. a GPU), and an error it raised

    def __init__(self, phases, int C, **kw):
        Solver.__init__(self, phases, C, **kw)
        self.scap = 256
        self.Sy = np.zeros((self.scap, self.Dmax)); self.Sx = np.zeros((self.scap, C)); self.SG = np.zeros(self.scap)
        self.Sph = np.zeros(self.scap, dtype=np.int32); self.Smap = np.zeros(self.scap, dtype=np.int32)
        self.head0 = np.zeros(self.nph, dtype=np.int32)
        self.stats = []
        self.Scode = np.zeros(self.scap, dtype=np.int64); self.Snext = np.full(self.scap, -1, dtype=np.int32)
        self.Sown = np.zeros(self.scap, dtype=np.int32); self.Sidx = np.zeros(self.scap, dtype=np.int32)
        self.own_s = np.zeros(64, dtype=np.int32)
        self.jcap = 0; self.cpcap = 0; self.nj = 0; self.ncp = 0
        self.job_batches = []
        self.pricer = NULL; self.device_jobs = 0; self.pricer_cb = None; self.cb_error = None

    def _grow_staged(self):
        cap = 2 * self.scap
        def g(a, shape, dtype):
            out = np.zeros(shape, dtype=dtype); out[:self.scap] = np.asarray(a); return out
        self.Sy = g(self.Sy, (cap, self.Dmax), float); self.Sx = g(self.Sx, (cap, self.C), float)
        self.SG = g(self.SG, (cap,), float); self.Sph = g(self.Sph, (cap,), np.int32)
        self.Smap = g(self.Smap, (cap,), np.int32)
        self.Scode = g(self.Scode, (cap,), np.int64); self.Snext = g(self.Snext, (cap,), np.int32)
        self.Sown = g(self.Sown, (cap,), np.int32); self.Sidx = g(self.Sidx, (cap,), np.int32)
        self.scap = cap

    cdef int find_point(self, int ph, double* y) noexcept nogil:
        """A stored point of phase ph at site fractions y (as add_point's duplicate test), or -1."""
        cdef int p = self.head[ph], j, D = (<CPhase>self.pptr[ph]).D
        cdef double m
        while p >= 0:
            m = 0.0
            for j in range(D):
                if fabs(self.Py[p, j] - y[j]) > m: m = fabs(self.Py[p, j] - y[j])
            if m < 1e-7:
                return p
            p = self.Pnext[p]
        return -1

    cdef void _load(self, int i) noexcept nogil:
        cdef int ph, c
        cdef void* pp
        for c in range(self.K): self.basis[c] = self.t_basis[i, c]
        for ph in range(self.nph):
            pp = self.pptr[ph]
            (<CPhase>pp).has_last = self.t_has[i, ph]; (<CPhase>pp).last_tpd = self.t_ltpd[i, ph]
            for c in range(self.C): (<CPhase>pp).last_mu[c] = self.t_lmu[i, ph, c]

    cdef void _save(self, int i) noexcept nogil:
        cdef int ph, c
        cdef void* pp
        for c in range(self.K): self.t_basis[i, c] = self.basis[c]
        for c in range(self.C): self.t_mu[i, c] = self.mu[c]
        for ph in range(self.nph):
            pp = self.pptr[ph]
            self.t_has[i, ph] = (<CPhase>pp).has_last; self.t_ltpd[i, ph] = (<CPhase>pp).last_tpd
            for c in range(self.C): self.t_lmu[i, ph, c] = (<CPhase>pp).last_mu[c]

    cdef void _stage(self, int i, int n0) noexcept nogil:
        """Move the points condition i added (store entries n0..npts-1) to the staging area, and point its basis at
        them there."""
        cdef int p, s, k, j
        for p in range(n0, self.npts):
            if self.nst >= self.scap:
                with gil:
                    self._grow_staged()
            s = self.nst
            for j in range(self.Dmax): self.Sy[s, j] = self.Py[p, j]
            for j in range(self.C): self.Sx[s, j] = self.Px[p, j]
            self.SG[s] = self.PG[p]; self.Sph[s] = self.Pph[p]
            for k in range(self.K):
                if self.t_basis[i, k] == p:
                    self.t_basis[i, k] = STAGED + s
            self.nst += 1
        self.npts = n0
        for k in range(self.nph): self.head[k] = self.head0[k]

    cdef int _merge(self, int B) noexcept nogil:
        """The staged points into the shared store (duplicates of stored or earlier staged points dropped); the
        bases follow. Returns the number of new points."""
        cdef int s, p, i, k, n0 = self.npts
        for s in range(self.nst):
            p = self.find_point(self.Sph[s], &self.Sy[s, 0])
            if p < 0:
                p = self.add_point(self.Sph[s], &self.Sy[s, 0], &self.Sx[s, 0], self.SG[s])
            self.Smap[s] = p
        for i in range(B):
            for k in range(self.K):
                if self.t_basis[i, k] >= STAGED:
                    p = self.Smap[self.t_basis[i, k] - STAGED]
                    # (a point add_point refused, e.g. non-finite: the artificial column for that row instead)
                    self.t_basis[i, k] = p if p >= 0 else -(1 + k)
        self.nst = 0
        return self.npts - n0

    cdef void _record(self, int t, int st, bint polished, int[::1] status_v, int[::1] nout_v, int[:, ::1] phs_v,
                      double[:, ::1] amt_v, double[:, :, ::1] X_v, double[:, :, ::1] Y_v, double[:, ::1] G_v,
                      double[:, ::1] mu_v) noexcept nogil:
        """The result of condition t (as solve_sequence writes it)."""
        cdef int a, i, j, k, p, na = 0, C = self.C, Dm = self.Dmax
        if st == 0 and polished:
            for a in range(self.pol_na):
                phs_v[t, na] = self.pol_ph[a]; amt_v[t, na] = self.pol_n[a]; G_v[t, na] = self.pol_G[a]
                for i in range(C): X_v[t, na, i] = self.pol_x[a * C + i]
                for j in range((<CPhase>self.pptr[self.pol_ph[a]]).D): Y_v[t, na, j] = self.pol_y[a * Dm + j]
                na += 1
        elif st == 0:
            self.nunpolished += 1
            for k in range(self.K):
                if self.xB[k] > 1e-300:
                    p = self.basis[k]
                    if p < 0:
                        if self.xB[k] > 1e-9:
                            st = 5
                            break
                        continue
                    phs_v[t, na] = self.Pph[p]; amt_v[t, na] = self.xB[k]; G_v[t, na] = self.PG[p]
                    for i in range(C): X_v[t, na, i] = self.Px[p, i]
                    for j in range((<CPhase>self.pptr[self.Pph[p]]).D): Y_v[t, na, j] = self.Py[p, j]
                    na += 1
        status_v[t] = st; nout_v[t] = na
        for i in range(C): mu_v[t, i] = self.mu[i] * self.RT

    def _work(self):
        return (sum((<CPhase>P).nprice for P in self.phases), sum((<CPhase>P).evals1 for P in self.phases),
                sum((<CPhase>P).evals2 for P in self.phases), self.lp_iters)

    cdef void _copy_state(self, int src, int dst) noexcept nogil:
        cdef int k, ph, c
        for k in range(self.t_basis.shape[1]): self.t_basis[dst, k] = self.t_basis[src, k]
        for c in range(self.C): self.t_mu[dst, c] = self.t_mu[src, c]
        for ph in range(self.nph):
            self.t_has[dst, ph] = self.t_has[src, ph]; self.t_ltpd[dst, ph] = self.t_ltpd[src, ph]
            for c in range(self.C): self.t_lmu[dst, ph, c] = self.t_lmu[src, ph, c]
        self.t_nopol[dst] = 0

    def solve_tile(self, R, b, mu_fixed, fixed, seeds=None, int max_rounds=60, stage=None, init_from=None):
        """Solve the conditions R[t], b[t], mu_fixed[t], fixed[t] (as solve_sequence) in lockstep rounds; returns the
        same arrays as solve_sequence. Condition 0 is solved first, alone, with the seeds. The others are solved in
        stages (stage[t] >= 1, in increasing order; default: all in stage 1), each condition starting from the final
        state (LP basis, potentials, skip state) of condition init_from[t] of an earlier stage (default: 0).
        self.stats: one dict per round (stage, conditions active, minimizations, energy gradient and Hessian
        evaluations, LP iterations, points staged and new)."""
        cdef double[:, :, ::1] Rv = np.ascontiguousarray(R, dtype=float)
        cdef double[:, ::1] bv = np.ascontiguousarray(b, dtype=float)
        cdef double[:, ::1] mfv = np.ascontiguousarray(mu_fixed, dtype=float)
        cdef int[:, ::1] fxv = np.ascontiguousarray(fixed, dtype=np.int32)
        cdef int B = Rv.shape[0], K = Rv.shape[1], C = self.C, Dm = self.Dmax, t, a, k, st, rounds, n0, nact, m
        cdef int nstaged, nmem
        cdef bint polished, added, no_polish, use_seeds = seeds is not None
        cdef double obj = 0.0
        stage = np.ones(B, dtype=np.int32) if stage is None else np.asarray(stage, dtype=np.int32)
        init_from = np.zeros(B, dtype=np.int32) if init_from is None else np.asarray(init_from, dtype=np.int32)
        if B and (stage[0] != 0 or (stage[1:] < 1).any()):
            raise ValueError('condition 0 alone is stage 0; the others need stages >= 1')
        if B and ((stage[init_from[1:]] >= stage[1:]).any()):
            raise ValueError('a condition must start from one of an earlier stage')
        status = np.zeros(B, dtype=np.int32); nout = np.zeros(B, dtype=np.int32)
        phs = np.full((B, C), -1, dtype=np.int32); amt = np.zeros((B, C)); X = np.zeros((B, C, C))
        Y = np.full((B, C, Dm), np.nan); G = np.zeros((B, C)); mu = np.zeros((B, C))
        cdef int[::1] status_v = status, nout_v = nout
        cdef int[:, ::1] phs_v = phs
        cdef double[:, ::1] amt_v = amt, G_v = G, mu_v = mu
        cdef double[:, :, ::1] X_v = X, Y_v = Y
        done = np.zeros(B, dtype=np.uint8)
        cdef unsigned char[::1] done_v = done
        cdef int[::1] members_v, init_v = init_from
        self.stats = []
        if B == 0:
            return status, nout, phs, amt, X, Y, G, mu
        self.RT = (<CPhase>self.phases[0]).RT
        if use_seeds:
            self._pack_seeds(seeds)
        self.t_basis = np.zeros((B, max(K, 1)), dtype=np.int32); self.t_mu = np.zeros((B, C))
        self.t_has = np.zeros((B, self.nph), dtype=np.uint8); self.t_ltpd = np.zeros((B, self.nph))
        self.t_lmu = np.zeros((B, self.nph, C)); self.t_nopol = np.zeros(B, dtype=np.uint8)
        w0 = self._work()
        with nogil:
            self._set_conditions_c(&Rv[0, 0, 0], &bv[0, 0], &mfv[0, 0], &fxv[0, 0], K)
            st = self._solve_c(use_seeds, max_rounds, &polished, &rounds)
            self._record(0, st, polished, status_v, nout_v, phs_v, amt_v, X_v, Y_v, G_v, mu_v)
            self._save(0)
        done_v[0] = 1
        w1 = self._work()
        self.stats.append({'stage': 0, 'round': 0, 'active': 1, 'rounds_first': rounds, 'priced': w1[0] - w0[0],
                           'evals1': w1[1] - w0[1], 'evals2': w1[2] - w0[2], 'lp_iters': w1[3] - w0[3],
                           'staged': 0, 'new': 0, 'store': self.npts})
        self.nst = 0
        for s_ in sorted(set(stage[1:].tolist())):
            members = np.nonzero(stage == s_)[0].astype(np.int32)
            members_v = members
            nmem = members.shape[0]
            with nogil:
                for m in range(nmem):
                    self._copy_state(init_v[members_v[m]], members_v[m])
            for rounds in range(1, max_rounds + 1):
                w0 = self._work()
                nact = 0; nstaged = 0
                with nogil:
                    for k in range(self.nph): self.head0[k] = self.head[k]
                    n0 = self.npts
                    for m in range(nmem):
                        t = members_v[m]
                        if done_v[t]:
                            continue
                        nact += 1
                        self._set_conditions_c(&Rv[t, 0, 0], &bv[t, 0], &mfv[t, 0], &fxv[t, 0], K)
                        self._load(t)
                        st = self.lp_solve(&obj)
                        if st != 0:
                            self._record(t, st, False, status_v, nout_v, phs_v, amt_v, X_v, Y_v, G_v, mu_v)
                            done_v[t] = 1
                            self._save(t); self._stage(t, n0)
                            continue
                        self.gauge_fix()
                        no_polish = self.t_nopol[t]
                        polished = self._round_step(rounds, False, &no_polish, &added)
                        self.t_nopol[t] = no_polish
                        if not added or rounds == max_rounds:
                            self._record(t, 6 if added else 0, polished, status_v, nout_v, phs_v, amt_v, X_v, Y_v,
                                         G_v, mu_v)
                            done_v[t] = 1
                        self._save(t)
                        nstaged += self.npts - n0
                        self._stage(t, n0)
                    new = self._merge(B)
                w1 = self._work()
                self.stats.append({'stage': int(s_), 'round': rounds, 'active': nact, 'priced': w1[0] - w0[0],
                                   'evals1': w1[1] - w0[1], 'evals2': w1[2] - w0[2], 'lp_iters': w1[3] - w0[3],
                                   'staged': nstaged, 'new': new, 'store': self.npts})
                if nact == 0 or done[members].all():
                    break
        return status, nout, phs, amt, X, Y, G, mu

    # ---- batched rounds: pricing as jobs (generate for every condition, execute all together, consume)

    def _grow_jobs(self):
        cdef int cap = max(1024, 2 * self.jcap), old = self.jcap
        def g(view, shape, dtype):
            out = np.zeros(shape, dtype=dtype)
            if old:   # (an unset memoryview must not be touched)
                out[:old] = np.asarray(view)
            return out
        if old:
            self.j_cond = g(self.j_cond, (cap,), np.int32); self.j_phase = g(self.j_phase, (cap,), np.int32)
            self.j_conv = g(self.j_conv, (cap,), np.int32)
            self.j_y0 = g(self.j_y0, (cap, self.Dmax), float); self.j_y = g(self.j_y, (cap, self.Dmax), float)
            self.j_x = g(self.j_x, (cap, self.C), float)
            self.j_G = g(self.j_G, (cap,), float); self.j_tpd = g(self.j_tpd, (cap,), float)
        else:
            self.j_cond = g(None, (cap,), np.int32); self.j_phase = g(None, (cap,), np.int32)
            self.j_conv = g(None, (cap,), np.int32)
            self.j_y0 = g(None, (cap, self.Dmax), float); self.j_y = g(None, (cap, self.Dmax), float)
            self.j_x = g(None, (cap, self.C), float)
            self.j_G = g(None, (cap,), float); self.j_tpd = g(None, (cap,), float)
        self.jcap = cap

    def _grow_cp(self):
        cap, old = max(256, 2 * self.cpcap), self.cpcap
        def g(view):
            out = np.zeros(cap, dtype=np.int32)
            if old:
                out[:old] = np.asarray(view)
            return out
        if old:
            self.cp_phase = g(self.cp_phase); self.cp_start = g(self.cp_start); self.cp_end = g(self.cp_end)
        else:
            self.cp_phase = g(None); self.cp_start = g(None); self.cp_end = g(None)
        self.cpcap = cap

    def _grow_own(self):
        out = np.zeros(2 * self.own_s.shape[0], dtype=np.int32); out[:self.own_s.shape[0]] = np.asarray(self.own_s)
        self.own_s = out

    cdef void _push_job(self, int t, int ph, double* y) noexcept nogil:
        cdef int i, j, D = (<CPhase>self.pptr[ph]).D
        if self.nj >= self.jcap:
            with gil:
                self._grow_jobs()
        j = self.nj
        self.j_cond[j] = t; self.j_phase[j] = ph
        for i in range(D): self.j_y0[j, i] = y[i]
        self.nj += 1

    cdef void _gen_jobs(self, int t, int rounds) noexcept nogil:
        """The pricing jobs of condition t at self.mu: the starts _pricing_round would price, phase by phase
        (skip rule, stored points, ordering patterns, LP basis points, composition start in a first round)."""
        cdef int C = self.C, K = self.K, ph, p, k, i, si, sj, ncodes, cp
        cdef double best_tpd, maxdmu, tt
        cdef void* pp
        cdef long codes[64]
        cdef int code_pt[64]
        cdef double code_tpd[64]
        cdef double* mu = &self.mu[0]
        self.cp_first[t] = self.ncp
        for ph in range(self.nph):
            pp = self.pptr[ph]
            if (<CPhase>pp).stoich:
                continue
            k = -1; best_tpd = INFINITY; ncodes = 0
            p = self.head[ph]
            while p >= 0:
                tt = self.PG[p] / self.RT
                for i in range(C): tt -= mu[i] * self.Px[p, i]
                if tt < best_tpd:
                    best_tpd = tt; k = p
                if tt < (self.prune if rounds == 1 else self.prune_late):
                    for si in range(ncodes):
                        if codes[si] == self.Pcode[p]:
                            if tt < code_tpd[si]:
                                code_tpd[si] = tt; code_pt[si] = p
                            break
                    else:
                        if ncodes < 64:
                            codes[ncodes] = self.Pcode[p]; code_tpd[ncodes] = tt; code_pt[ncodes] = p; ncodes += 1
                p = self.Pnext[p]
            if (<CPhase>pp).has_last and best_tpd > self.skip_margin:
                maxdmu = 0.0
                for i in range(C):
                    tt = fabs(mu[i] - (<CPhase>pp).last_mu[i])
                    if tt > maxdmu: maxdmu = tt
                if (<CPhase>pp).last_tpd - maxdmu > self.skip_margin:
                    self.skipped += 1
                    continue
            if k < 0:
                continue
            if self.ncp >= self.cpcap:
                with gil:
                    self._grow_cp()
            cp = self.ncp
            self.cp_phase[cp] = ph; self.cp_start[cp] = self.nj
            self._push_job(t, ph, &self.Py[k, 0])
            for si in range(ncodes):
                for sj in range(si + 1, ncodes):
                    if code_tpd[sj] < code_tpd[si]:
                        tt = code_tpd[si]; code_tpd[si] = code_tpd[sj]; code_tpd[sj] = tt
                        p = code_pt[si]; code_pt[si] = code_pt[sj]; code_pt[sj] = p
            for si in range(K):
                p = self.basis[si]
                if p >= 0 and p != k and self.Pph[p] == ph and self.xB[si] > 1e-12:
                    self._push_job(t, ph, &self.Py[p, 0])
            for si in range(min(ncodes, 6 if rounds == 1 else self.max_late)):
                if code_pt[si] != k and self.Pcode[code_pt[si]] != self.Pcode[k]:
                    self._push_job(t, ph, &self.Py[code_pt[si], 0])
            if rounds == 1:
                (<CPhase>pp).composition_start(&self.Px[k, 0], &self.ystart[0])
                self._push_job(t, ph, &self.ystart[0])
            self.cp_end[cp] = self.nj
            self.ncp += 1
        self.cp_last[t] = self.ncp
        for i in range(C): self.t_mu[t, i] = mu[i]

    def set_pricer(self, size_t address, ok, di, dd, ip, dp, sv, double RT, keep=None):
        """Price the jobs of the phases flagged in ok with the batched C function at address (cg_price_batch of
        cg_price.h; descriptions di, dd, pools ip, dp, state variables sv); address 0: price everything here."""
        if address == 0:
            self.pricer = NULL; self.pricer_keep = None
            return
        self.pricer = <pricer_fn>address
        self.pricer_ok = np.ascontiguousarray(ok, dtype=np.uint8)
        self.pr_di = np.ascontiguousarray(di, dtype=np.int32); self.pr_dd = np.ascontiguousarray(dd, dtype=float)
        self.pr_ip = np.ascontiguousarray(ip, dtype=np.int32); self.pr_dp = np.ascontiguousarray(dp, dtype=float)
        self.pr_sv = np.ascontiguousarray(sv, dtype=float); self.pr_RT = RT
        self.pr_jobs = np.zeros(max(self.jcap, 1024), dtype=np.int32)
        self.pr_evals = np.zeros(2 * max(self.jcap, 1024), dtype=np.int32)
        self.pricer_keep = keep

    def set_pricer_callback(self, cb, ok):
        """Price the jobs of the phases flagged in ok by calling cb(jobs, job_phase, job_cond, mu, y0, y, x, G, tpd,
        conv, evals) once per sub-round: jobs lists the job indices to price; the other arrays are the job tables
        (indexed by job; mu by condition, RT units), into which cb writes y, x, G, tpd, conv and evals (2 per job).
        None: price everything here."""
        self.pricer_cb = cb; self.cb_error = None
        if cb is None:
            return
        self.pricer = NULL
        self.pricer_ok = np.ascontiguousarray(ok, dtype=np.uint8)
        self.pr_jobs = np.zeros(max(self.jcap, 1024), dtype=np.int32)
        self.pr_evals = np.zeros(2 * max(self.jcap, 1024), dtype=np.int32)

    def _call_pricer(self, int n):
        try:
            self.pricer_cb(np.asarray(self.pr_jobs)[:n], np.asarray(self.j_phase)[:self.nj],
                           np.asarray(self.j_cond)[:self.nj], np.asarray(self.t_mu), np.asarray(self.j_y0)[:self.nj],
                           np.asarray(self.j_y)[:self.nj], np.asarray(self.j_x)[:self.nj],
                           np.asarray(self.j_G)[:self.nj], np.asarray(self.j_tpd)[:self.nj],
                           np.asarray(self.j_conv)[:self.nj], np.asarray(self.pr_evals)[:2 * self.nj])
        except BaseException as e:
            self.cb_error = e
            for k in range(n):
                j = self.pr_jobs[k]
                self.j_conv[j] = 0; self.j_G[j] = np.nan; self.j_tpd[j] = np.inf

    def _grow_pricer_buffers(self):
        self.pr_jobs = np.zeros(self.jcap, dtype=np.int32); self.pr_evals = np.zeros(2 * self.jcap, dtype=np.int32)

    cdef void _execute_jobs(self) noexcept nogil:
        """Every pricing job: a local TPD minimization of its phase from its start at its condition's potentials.
        (Independent of each other: the part a batched device kernel does, see set_pricer.)"""
        cdef int j, n = 0, k
        cdef void* pp
        cdef bint use_cb = False
        with gil:
            use_cb = self.pricer_cb is not None
        if self.pricer != NULL or use_cb:
            if self.pr_jobs.shape[0] < self.nj:
                with gil:
                    self._grow_pricer_buffers()
            for j in range(self.nj):
                if self.pricer_ok[self.j_phase[j]]:
                    self.pr_jobs[n] = j; n += 1
                else:
                    self.j_conv[j] = (<CPhase>self.pptr[self.j_phase[j]]).price(
                        &self.t_mu[self.j_cond[j], 0], &self.j_y0[j, 0], &self.j_y[j, 0], &self.j_x[j, 0],
                        &self.j_G[j], &self.j_tpd[j])
            if n > 0:
                if use_cb:
                    with gil:
                        self._call_pricer(n)
                else:
                    self.pricer(n, &self.pr_jobs[0], &self.j_phase[0], &self.j_cond[0], &self.t_mu[0, 0],
                                &self.j_y0[0, 0], self.Dmax, &self.j_y[0, 0], &self.j_x[0, 0], &self.j_G[0],
                                &self.j_tpd[0], &self.j_conv[0], &self.pr_evals[0], &self.pr_di[0], &self.pr_dd[0],
                                &self.pr_ip[0], &self.pr_dp[0], self.C, self.pr_RT, &self.pr_sv[0])
                self.device_jobs += n
                for k in range(n):
                    j = self.pr_jobs[k]; pp = self.pptr[self.j_phase[j]]
                    (<CPhase>pp).nprice = (<CPhase>pp).nprice + 1
                    (<CPhase>pp).evals1 = (<CPhase>pp).evals1 + self.pr_evals[2 * j]
                    (<CPhase>pp).evals2 = (<CPhase>pp).evals2 + self.pr_evals[2 * j + 1]
            return
        for j in range(self.nj):
            self.j_conv[j] = (<CPhase>self.pptr[self.j_phase[j]]).price(
                &self.t_mu[self.j_cond[j], 0], &self.j_y0[j, 0], &self.j_y[j, 0], &self.j_x[j, 0], &self.j_G[j],
                &self.j_tpd[j])

    cdef bint _consume(self, int t, bint polished, bint* no_polish) noexcept nogil:
        """Condition t's job results, as _pricing_round treats its minimizations (points below the tangent plane
        stored, certification of a polished state, skip state); returns whether anything new was found."""
        cdef int C = self.C, cp, j, ph, all_conv, p, i
        cdef double best_tpd, tpd, G, tt
        cdef bint added = False, seed_added = False
        cdef void* pp
        cdef double* mu = &self.mu[0]
        self.cert_kind = 0
        for cp in range(self.cp_first[t], self.cp_last[t]):
            ph = self.cp_phase[cp]; pp = self.pptr[ph]
            best_tpd = INFINITY; all_conv = 1
            for j in range(self.cp_start[cp], self.cp_end[cp]):
                if not self.j_conv[j]:
                    all_conv = 0
                G = self.j_G[j]; tpd = self.j_tpd[j]
                if not isfinite(G):
                    tpd = INFINITY
                self.priced += 1
                if tpd < -self.tol:
                    if self.add_point(ph, &self.j_y[j, 0], &self.j_x[j, 0], G) >= 0:
                        seed_added = True
                if tpd < best_tpd:
                    best_tpd = tpd
                    memcpy(&self.ybest[0], &self.j_y[j, 0], (<CPhase>pp).D * sizeof(double))
            if polished and best_tpd < -self.tol:
                if self.cert_kind == 0 or best_tpd < self.cert_tpd:
                    memcpy(&self.ycert[0], &self.ybest[0], (<CPhase>pp).D * sizeof(double))
                    self.cert_kind = 1; self.cert_phase = ph; self.cert_tpd = best_tpd
                added = True; no_polish[0] = True
            if all_conv:
                (<CPhase>pp).has_last = True; (<CPhase>pp).last_tpd = best_tpd
                for i in range(C): (<CPhase>pp).last_mu[i] = mu[i]
            else:
                (<CPhase>pp).has_last = False
        if polished and not added:
            for p in range(self.npts):
                tt = self.PG[p] / self.RT
                for i in range(C): tt -= mu[i] * self.Px[p, i]
                if tt < -self.tol:
                    self.cert_kind = 2; self.cert_phase = self.Pph[p]; self.cert_tpd = tt
                    memcpy(&self.ycert[0], &self.Py[p, 0], (<CPhase>self.pptr[self.Pph[p]]).D * sizeof(double))
                    added = True; no_polish[0] = True
                    break
        if seed_added and not added:
            self.cert_kind = 3
        return added or seed_added

    cdef void _load_own(self, int t, int n0) noexcept nogil:
        """Condition t's own points of the round back into the store (after the n0 shared ones, in insertion
        order)."""
        cdef int s = self.Shead[t], p, j, ph
        while s >= 0:
            if self.npts >= self.cap:
                with gil:
                    self._grow()
            if self.npts - n0 >= self.own_s.shape[0]:
                with gil:
                    self._grow_own()
            p = self.npts; ph = self.Sph[s]
            for j in range(self.Dmax): self.Py[p, j] = self.Sy[s, j]
            for j in range(self.C): self.Px[p, j] = self.Sx[s, j]
            self.PG[p] = self.SG[s]; self.Pph[p] = ph; self.Pcode[p] = self.Scode[s]
            self.Pnext[p] = self.head[ph]; self.head[ph] = p
            self.npts += 1
            self.Sidx[s] = p; self.own_s[p - n0] = s
            s = self.Snext[s]

    cdef void _save_own(self, int t, int n0) noexcept nogil:
        """Condition t's own points (store entries from n0) to its chain (new ones appended), its saved basis
        pointing at them there; the store back to the n0 shared points."""
        cdef int p, s, j, k, b, nloaded = 0
        s = self.Shead[t]
        while s >= 0:
            nloaded += 1; s = self.Snext[s]
        for p in range(n0 + nloaded, self.npts):
            if self.nst >= self.scap:
                with gil:
                    self._grow_staged()
            if p - n0 >= self.own_s.shape[0]:
                with gil:
                    self._grow_own()
            s = self.nst; self.nst += 1
            for j in range(self.Dmax): self.Sy[s, j] = self.Py[p, j]
            for j in range(self.C): self.Sx[s, j] = self.Px[p, j]
            self.SG[s] = self.PG[p]; self.Sph[s] = self.Pph[p]; self.Scode[s] = self.Pcode[p]
            self.Sown[s] = t; self.Snext[s] = -1
            if self.Stail[t] >= 0:
                self.Snext[self.Stail[t]] = s
            else:
                self.Shead[t] = s
            self.Stail[t] = s
            self.own_s[p - n0] = s
        for k in range(self.K):
            b = self.t_basis[t, k]
            if b >= n0 and b < STAGED:
                self.t_basis[t, k] = STAGED + self.own_s[b - n0]
        self.npts = n0
        for k in range(self.nph): self.head[k] = self.head0[k]

    cdef void _save_cond(self, int t, bint polished, bint skipped, bint added, int aug, int kind) noexcept nogil:
        """Everything about condition t a later sub-round needs (with _save: basis, potentials, skip state)."""
        cdef int i, C = self.C
        self._save(t)
        for i in range(C): self.t_xB[t, i] = self.xB[i]
        for i in range(C * self.Dmax): self.t_pol_y[t, i] = self.pol_y[i]
        for i in range(C * C): self.t_pol_x[t, i] = self.pol_x[i]
        for i in range(C):
            self.t_pol_n[t, i] = self.pol_n[i]; self.t_pol_mu[t, i] = self.pol_mu[i]; self.t_pol_G[t, i] = self.pol_G[i]
            self.t_pol_ph[t, i] = self.pol_ph[i]
        for i in range(self.Dmax): self.t_ycert[t, i] = self.ycert[i]
        self.t_pol_na[t] = self.pol_na
        self.t_cert_kind[t] = self.cert_kind; self.t_cert_phase[t] = self.cert_phase; self.t_cert_tpd[t] = self.cert_tpd
        self.t_polished[t] = polished; self.t_skipped[t] = skipped; self.t_added[t] = added
        self.t_aug[t] = aug; self.t_kind[t] = kind

    cdef void _load_cond(self, int t, double* R, double* b, double* mf, int* fx, int K) noexcept nogil:
        """Condition t back as the solver's current one (own points first, then its basis pointing at them)."""
        cdef int i, k, C = self.C, bb
        self._set_conditions_c(R, b, mf, fx, K)
        self._load(t)
        for k in range(self.K):
            bb = self.basis[k]
            if bb >= STAGED:
                self.basis[k] = self.Sidx[bb - STAGED]
        for i in range(C): self.mu[i] = self.t_mu[t, i]; self.xB[i] = self.t_xB[t, i]
        for i in range(C * self.Dmax): self.pol_y[i] = self.t_pol_y[t, i]
        for i in range(C * C): self.pol_x[i] = self.t_pol_x[t, i]
        for i in range(C):
            self.pol_n[i] = self.t_pol_n[t, i]; self.pol_mu[i] = self.t_pol_mu[t, i]; self.pol_G[i] = self.t_pol_G[t, i]
            self.pol_ph[i] = self.t_pol_ph[t, i]
        for i in range(self.Dmax): self.ycert[i] = self.t_ycert[t, i]
        self.pol_na = self.t_pol_na[t]
        self.cert_kind = self.t_cert_kind[t]; self.cert_phase = self.t_cert_phase[t]; self.cert_tpd = self.t_cert_tpd[t]

    cdef int _merge_chains(self, int[::1] members, int B) noexcept nogil:
        """The conditions' own points into the shared store, condition by condition in insertion order (duplicates
        of stored or earlier points dropped); the saved bases follow. Returns the number of new points."""
        cdef int m, t, s, p, i, k, n0 = self.npts
        for m in range(members.shape[0]):
            t = members[m]
            s = self.Shead[t]
            while s >= 0:
                p = self.find_point(self.Sph[s], &self.Sy[s, 0])
                if p < 0:
                    p = self.add_point(self.Sph[s], &self.Sy[s, 0], &self.Sx[s, 0], self.SG[s])
                self.Smap[s] = p
                s = self.Snext[s]
            self.Shead[t] = -1; self.Stail[t] = -1
        for m in range(members.shape[0]):
            t = members[m]
            for k in range(self.K):
                if self.t_basis[t, k] >= STAGED:
                    p = self.Smap[self.t_basis[t, k] - STAGED]
                    self.t_basis[t, k] = p if p >= 0 else -(1 + k)
        self.nst = 0
        return self.npts - n0

    def solve_tile_batched(self, R, b, mu_fixed, fixed, seeds=None, int max_rounds=60, stage=None, init_from=None):
        """solve_tile with the pricing of every round done as one batch of jobs per sub-round: each condition does its
        LP and polish and lists its pricing jobs; all jobs run (_execute_jobs); each condition takes its results. A
        further Newton pass with a minimum found below a polished tangent plane, or the polish of a round that had
        skipped it, makes another sub-round of the same round. Same results as solve_tile. self.job_batches: the
        number of jobs of each sub-round, per round."""
        cdef double[:, :, ::1] Rv = np.ascontiguousarray(R, dtype=float)
        cdef double[:, ::1] bv = np.ascontiguousarray(b, dtype=float)
        cdef double[:, ::1] mfv = np.ascontiguousarray(mu_fixed, dtype=float)
        cdef int[:, ::1] fxv = np.ascontiguousarray(fixed, dtype=np.int32)
        cdef int B = Rv.shape[0], K = Rv.shape[1], C = self.C, Dm = self.Dmax, t, k, st, rounds, n0, nact, m
        cdef int nstaged, nmem, kind, aug
        cdef bint polished, added, no_polish, skipped, use_seeds = seeds is not None, ok
        cdef double obj = 0.0
        # condition steps: MAIN / AUG / FINAL = its jobs await consumption; NEED_AUG / NEED_FINAL = its next jobs
        # are to be generated; IDLE = nothing more this round
        cdef int IDLE = 0, MAIN = 1, AUG = 2, FINAL = 3, NEED_AUG = 4, NEED_FINAL = 5
        stage = np.ones(B, dtype=np.int32) if stage is None else np.asarray(stage, dtype=np.int32)
        init_from = np.zeros(B, dtype=np.int32) if init_from is None else np.asarray(init_from, dtype=np.int32)
        if B and (stage[0] != 0 or (stage[1:] < 1).any()):
            raise ValueError('condition 0 alone is stage 0; the others need stages >= 1')
        if B and ((stage[init_from[1:]] >= stage[1:]).any()):
            raise ValueError('a condition must start from one of an earlier stage')
        status = np.zeros(B, dtype=np.int32); nout = np.zeros(B, dtype=np.int32)
        phs = np.full((B, C), -1, dtype=np.int32); amt = np.zeros((B, C)); X = np.zeros((B, C, C))
        Y = np.full((B, C, Dm), np.nan); G = np.zeros((B, C)); mu = np.zeros((B, C))
        cdef int[::1] status_v = status, nout_v = nout
        cdef int[:, ::1] phs_v = phs
        cdef double[:, ::1] amt_v = amt, G_v = G, mu_v = mu
        cdef double[:, :, ::1] X_v = X, Y_v = Y
        done = np.zeros(B, dtype=np.uint8)
        cdef unsigned char[::1] done_v = done
        cdef int[::1] members_v, init_v = init_from
        self.stats = []; self.job_batches = []
        if B == 0:
            return status, nout, phs, amt, X, Y, G, mu
        self.RT = (<CPhase>self.phases[0]).RT
        if use_seeds:
            self._pack_seeds(seeds)
        self.t_basis = np.zeros((B, max(K, 1)), dtype=np.int32); self.t_mu = np.zeros((B, C))
        self.t_has = np.zeros((B, self.nph), dtype=np.uint8); self.t_ltpd = np.zeros((B, self.nph))
        self.t_lmu = np.zeros((B, self.nph, C)); self.t_nopol = np.zeros(B, dtype=np.uint8)
        self.t_xB = np.zeros((B, C)); self.t_pol_y = np.zeros((B, C * Dm)); self.t_pol_x = np.zeros((B, C * C))
        self.t_pol_n = np.zeros((B, C)); self.t_pol_mu = np.zeros((B, C)); self.t_pol_G = np.zeros((B, C))
        self.t_ycert = np.zeros((B, Dm)); self.t_pol_ph = np.zeros((B, C), dtype=np.int32)
        self.t_pol_na = np.zeros(B, dtype=np.int32); self.t_cert_kind = np.zeros(B, dtype=np.int32)
        self.t_cert_phase = np.zeros(B, dtype=np.int32); self.t_cert_tpd = np.zeros(B)
        self.t_aug = np.zeros(B, dtype=np.int32); self.t_kind = np.zeros(B, dtype=np.int32)
        self.t_polished = np.zeros(B, dtype=np.uint8); self.t_skipped = np.zeros(B, dtype=np.uint8)
        self.t_added = np.zeros(B, dtype=np.uint8)
        self.Shead = np.full(B, -1, dtype=np.int32); self.Stail = np.full(B, -1, dtype=np.int32)
        self.cp_first = np.zeros(B, dtype=np.int32); self.cp_last = np.zeros(B, dtype=np.int32)
        if self.jcap == 0:
            self._grow_jobs()
        if self.cpcap == 0:
            self._grow_cp()
        w0 = self._work()
        with nogil:
            self._set_conditions_c(&Rv[0, 0, 0], &bv[0, 0], &mfv[0, 0], &fxv[0, 0], K)
            st = self._solve_c(use_seeds, max_rounds, &polished, &rounds)
            self._record(0, st, polished, status_v, nout_v, phs_v, amt_v, X_v, Y_v, G_v, mu_v)
            self._save(0)
        done_v[0] = 1
        w1 = self._work()
        self.stats.append({'stage': 0, 'round': 0, 'active': 1, 'rounds_first': rounds, 'priced': w1[0] - w0[0],
                           'evals1': w1[1] - w0[1], 'evals2': w1[2] - w0[2], 'lp_iters': w1[3] - w0[3],
                           'staged': 0, 'new': 0, 'store': self.npts, 'jobs': []})
        self.nst = 0
        for s_ in sorted(set(stage[1:].tolist())):
            members = np.nonzero(stage == s_)[0].astype(np.int32)
            members_v = members
            nmem = members.shape[0]
            with nogil:
                for m in range(nmem):
                    self._copy_state(init_v[members_v[m]], members_v[m])
            for rounds in range(1, max_rounds + 1):
                w0 = self._work()
                batches = []
                nact = 0
                with nogil:
                    for k in range(self.nph): self.head0[k] = self.head[k]
                    n0 = self.npts
                    self.nj = 0; self.ncp = 0
                    # each active condition: LP, polish, its pricing jobs
                    for m in range(nmem):
                        t = members_v[m]
                        if done_v[t]:
                            continue
                        nact += 1
                        self._set_conditions_c(&Rv[t, 0, 0], &bv[t, 0], &mfv[t, 0], &fxv[t, 0], K)
                        self._load(t)
                        st = self.lp_solve(&obj)
                        if st != 0:
                            self._record(t, st, False, status_v, nout_v, phs_v, amt_v, X_v, Y_v, G_v, mu_v)
                            done_v[t] = 1
                            self._save_cond(t, False, False, False, 0, IDLE); self._save_own(t, n0)
                            continue
                        self.gauge_fix()
                        skipped = self.t_nopol[t]
                        polished = False
                        if self.use_polish and not self.t_nopol[t]:
                            self.npolish_try += 1
                            polished = self.polish() == 1
                        self.t_nopol[t] = False
                        self._gen_jobs(t, rounds)
                        self._save_cond(t, polished, skipped, False, 0, MAIN); self._save_own(t, n0)
                while self.nj > 0:
                    batches.append(self.nj)
                    with nogil:
                        self._execute_jobs()
                    if self.cb_error is not None:
                        err, self.cb_error = self.cb_error, None
                        raise err
                    with nogil:
                        # each condition takes its results
                        for m in range(nmem):
                            t = members_v[m]
                            kind = self.t_kind[t]
                            if kind != MAIN and kind != AUG and kind != FINAL:
                                continue
                            self._load_own(t, n0)
                            self._load_cond(t, &Rv[t, 0, 0], &bv[t, 0], &mfv[t, 0], &fxv[t, 0], K)
                            polished = self.t_polished[t]; skipped = self.t_skipped[t]; aug = self.t_aug[t]
                            no_polish = self.t_nopol[t]
                            added = self._consume(t, polished, &no_polish)
                            self.t_nopol[t] = no_polish
                            if kind == FINAL:
                                if added:
                                    polished = False
                                kind = IDLE
                            elif added and polished and (self.cert_kind == 1 or self.cert_kind == 2) and aug < C:
                                kind = NEED_AUG
                            elif not added and not polished and skipped and self.use_polish:
                                kind = NEED_FINAL
                            else:
                                kind = IDLE
                            if kind == IDLE and (not added or rounds == max_rounds):
                                self._record(t, 6 if added else 0, polished, status_v, nout_v, phs_v, amt_v, X_v,
                                             Y_v, G_v, mu_v)
                                done_v[t] = 1
                            self._save_cond(t, polished, skipped, added, aug, kind); self._save_own(t, n0)
                        # the next jobs: another Newton pass, or the polish of a round that skipped it
                        self.nj = 0; self.ncp = 0
                        for m in range(nmem):
                            t = members_v[m]
                            kind = self.t_kind[t]
                            if kind != NEED_AUG and kind != NEED_FINAL:
                                continue
                            self._load_own(t, n0)
                            self._load_cond(t, &Rv[t, 0, 0], &bv[t, 0], &mfv[t, 0], &fxv[t, 0], K)
                            polished = self.t_polished[t]; skipped = self.t_skipped[t]; aug = self.t_aug[t]
                            added = self.t_added[t]
                            if kind == NEED_AUG:
                                aug += 1
                                if self._polish_augment():
                                    self.t_nopol[t] = False
                                    self._gen_jobs(t, rounds)
                                    kind = AUG
                                else:
                                    polished = False
                                    kind = IDLE   # (added: the condition goes on next round)
                            else:
                                self.npolish_try += 1
                                if self.polish() == 1:
                                    polished = True
                                    self._gen_jobs(t, rounds)
                                    kind = FINAL
                                else:
                                    kind = IDLE
                                    if not added or rounds == max_rounds:
                                        self._record(t, 6 if added else 0, polished, status_v, nout_v, phs_v,
                                                     amt_v, X_v, Y_v, G_v, mu_v)
                                        done_v[t] = 1
                            if kind == IDLE and added and rounds == max_rounds and not done_v[t]:
                                self._record(t, 6, polished, status_v, nout_v, phs_v, amt_v, X_v, Y_v, G_v, mu_v)
                                done_v[t] = 1
                            self._save_cond(t, polished, skipped, added, aug, kind); self._save_own(t, n0)
                with nogil:
                    nstaged = self.nst
                    new = self._merge_chains(members_v, B)
                w1 = self._work()
                self.job_batches.append(batches)
                self.stats.append({'stage': int(s_), 'round': rounds, 'active': nact, 'priced': w1[0] - w0[0],
                                   'evals1': w1[1] - w0[1], 'evals2': w1[2] - w0[2], 'lp_iters': w1[3] - w0[3],
                                   'staged': nstaged, 'new': new, 'store': self.npts, 'jobs': batches})
                if nact == 0 or done[members].all():
                    break
        return status, nout, phs, amt, X, Y, G, mu
