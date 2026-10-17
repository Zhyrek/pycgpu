// cg pricing as portable C++: the local minimization of a phase's tangent-plane distance (CPhase.price in
// _core.pyx), one job per call, for batched execution on the host or a GPU (the same source for both).
//
// Port of CPhase.price for phases whose formula moles are linear in the site fractions and that have no
// constraints beyond the sublattice balances (no charge balance, no phase-local conditions): the energy and its
// derivatives come from the generated model functions (pycalphad.gpu.gpu_codegen) through cg_eval(), which the
// including source defines. LAPACK's Cholesky and symmetric eigensolver are replaced by small routines here
// (Cholesky, cyclic Jacobi), so results agree with the Cython version to rounding, not bit for bit.
//
// The including source defines CG_DEVICE (empty on the host, __device__ on a GPU), CG_DMAX (the largest number
// of site fractions), CG_CMAX (elements), CG_NMAX (largest derivative vector: state variables the energy depends
// on plus site fractions), CG_NS (state variables in the energy's input) and
//     CG_DEVICE void cg_eval(int model, int hess, double* eg, double* H, const double* x);
// (eg = [G, dG/dv], H row-major d2G/dv2 over the model's derivative variables v; G per formula unit).
#pragma once

// the big shared routines are out of line on a device (CG_NOINLINE: __noinline__): inlined into each kernel and
// call site, the device compiler takes several times longer
#ifndef CG_NOINLINE
#define CG_NOINLINE
#endif

#define CG_Y_FLOOR 1e-14
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__CUDACC_RTC__) || defined(__HIPCC_RTC__)
#define CG_ISFINITE(v) isfinite(v)
#else
#include <cmath>
#define CG_ISFINITE(v) std::isfinite(v)   // (g++'s <cmath> has no global isfinite)
#endif

// Phase description: integers (desc_i) and doubles (desc_d) per phase; offsets index the int / double pools
enum {
    CGI_D = 0, CGI_NB, CGI_MODEL, CGI_NSYM, CGI_NDER, CGI_DOFF,
    CGI_BOFF, CGI_BIDX, CGI_BLK, CGI_SINGLE, CGI_PINNED, CGI_SYMA, CGI_SYMB, CGI_SYMOFF, CGI_A, CGI_ASUM,
    CGI_ELEMOF, CGI_STOICH, CGI_COUNT
};
enum { CGD_MLO = 0, CGD_PENW, CGD_MAXDZ, CGD_COUNT };

struct CgPhase {
    int D, C, nb, model, nsym, nder, doff;
    const int *boff, *bidx, *blk, *single, *pinned, *sym_a, *sym_b, *sym_off;
    const double *A, *asum;
    double m_lo, pen_w, max_dz, RT;
};

// work arrays of one job, sized for phases of at most DM site fractions and NM derivative variables (a kernel per
// size class keeps them small; CgWork: the largest of the system)
template <int DM, int NM>
struct CgWorkT {
    double y[DM], z[DM], zt[DM], yt[DM], gz[DM], gzt[DM], gy[DM], gyt[DM], gl[DM], d[DM], w[DM], coef[DM], sc[DM],
        bt[DM], nu[DM];
    int free_[DM];
    double mval[CG_CMAX], mu[CG_CMAX];
    double inp[CG_NS + DM];
    double eg[1 + NM];
    double H[NM * NM];
    double Hy[DM * DM], Mx[DM * DM], Hz[DM * DM], V[DM * DM], Hs[DM * DM], Ev[DM * DM];
    double Mtot, G_fu;
    int last_indef, evals1, evals2;
};
typedef CgWorkT<CG_DMAX, CG_NMAX> CgWork;

CG_DEVICE static inline double cg_fabs(double a) { return a < 0 ? -a : a; }

CG_DEVICE static void cg_y_of_z(const CgPhase& P, const double* z, double* y) {
    for (int b = 0; b < P.nb; ++b) {
        double m = -INFINITY;
        for (int k = P.boff[b]; k < P.boff[b + 1]; ++k) {
            int j = P.bidx[k];
            if (z[j] > m && !P.pinned[j]) m = z[j];
        }
        double s = 0.0;
        for (int k = P.boff[b]; k < P.boff[b + 1]; ++k) {
            int j = P.bidx[k];
            if (P.pinned[j]) y[j] = 0.0;
            else { y[j] = exp(z[j] - m); s += y[j]; }
        }
        for (int k = P.boff[b]; k < P.boff[b + 1]; ++k) {
            int j = P.bidx[k];
            y[j] = s > 0 ? y[j] / s : 0.0;
            if (y[j] < CG_Y_FLOOR) y[j] = CG_Y_FLOOR;
        }
    }
}

template <class WT>
CG_DEVICE static void cg_moles(const CgPhase& P, WT& W, const double* y) {
    W.Mtot = 0.0;
    for (int j = 0; j < P.D; ++j) W.Mtot += P.asum[j] * y[j];
    for (int i = 0; i < P.C; ++i) {
        double s = 0.0;
        for (int j = 0; j < P.D; ++j) s += P.A[i * P.D + j] * y[j];
        W.mval[i] = s;
    }
}

// energy and gradient (hess: also the Hessian), formula moles
template <class WT>
CG_DEVICE static void cg_evaluate(const CgPhase& P, WT& W, const double* y, int hess) {
    if (hess) W.evals2++;
    W.evals1++;
    for (int j = 0; j < P.D; ++j) W.inp[CG_NS + j] = y[j];
    cg_eval(P.model, hess, W.eg, W.H, W.inp);
    W.G_fu = W.eg[0];
    cg_moles(P, W, y);
}

// after cg_evaluate at y: TPD/RT (without the vacancy penalty), its gradient in y
template <class WT>
CG_DEVICE static double cg_tpd_parts(const CgPhase& P, WT& W, double* gy) {
    double M = W.Mtot, u = W.G_fu / P.RT;
    for (int i = 0; i < P.C; ++i) u -= W.mu[i] * W.mval[i];
    double f = u / M;
    for (int j = 0; j < P.D; ++j) {
        double s = W.eg[1 + P.doff + j] / P.RT - W.nu[j];
        gy[j] = (s - f * P.asum[j]) / M;
    }
    return f;
}

template <class WT>
CG_DEVICE static double cg_fgz(const CgPhase& P, WT& W, const double* z, double* y, double* gz, double* gy) {
    cg_y_of_z(P, z, y);
    cg_evaluate(P, W, y, 0);
    double f = cg_tpd_parts(P, W, gy), M = W.Mtot;
    if (M < P.m_lo) {
        double t = P.m_lo - M;
        f += P.pen_w * t * t * t;
        for (int j = 0; j < P.D; ++j) gy[j] -= 3 * P.pen_w * t * t * P.asum[j];
    }
    for (int b = 0; b < P.nb; ++b) {
        double s = 0.0;
        for (int k = P.boff[b]; k < P.boff[b + 1]; ++k) { int j = P.bidx[k]; s += y[j] * gy[j]; }
        for (int k = P.boff[b]; k < P.boff[b + 1]; ++k) { int j = P.bidx[k]; gz[j] = y[j] * (gy[j] - s); }
    }
    return f;
}

template <class WT>
CG_DEVICE static double cg_gznorm(const CgPhase& P, const WT& W) {
    double s = 0.0;
    for (int j = 0; j < P.D; ++j) s += W.gz[j] * W.gz[j];
    return sqrt(s);
}

// stationarity in potentials (see CPhase.kkt)
template <class WT>
CG_DEVICE static double cg_kkt(const CgPhase& P, WT& W) {
    double out = 0.0;
    for (int j = 0; j < P.D; ++j) W.gl[j] = W.gy[j];
    for (int b = 0; b < P.nb; ++b) {
        if (P.single[b]) continue;
        double s = 0.0;
        for (int k = P.boff[b]; k < P.boff[b + 1]; ++k) { int j = P.bidx[k]; s += W.y[j] * W.gl[j]; }
        for (int k = P.boff[b]; k < P.boff[b + 1]; ++k) {
            int j = P.bidx[k];
            double r = W.gl[j] - s;
            if (P.pinned[j]) continue;
            if (W.y[j] > 1.5 * CG_Y_FLOOR) { if (cg_fabs(r) > out) out = cg_fabs(r); }
            else if (-r > out) out = -r;
        }
    }
    return out == out ? out : INFINITY;
}

// Gauss-Newton Hessian over the free variables, Jacobi-scaled (V and Hs, m x m); returns m (see build_hessian)
template <class WT>
CG_DEVICE static int cg_build_hessian(const CgPhase& P, WT& W) {
    const int D = P.D, nd = P.nder, o = P.doff;
    cg_evaluate(P, W, W.y, 1);
    double M = W.Mtot, f = cg_tpd_parts(P, W, W.gyt);
    for (int a = 0; a < D; ++a)
        for (int b = a; b < D; ++b) {
            double hu = W.H[(o + a) * nd + o + b] / P.RT;
            double s = hu - W.gyt[a] * P.asum[b] - P.asum[a] * W.gyt[b];
            s /= M;
            if (M < P.m_lo) s += 6 * P.pen_w * (P.m_lo - M) * P.asum[a] * P.asum[b];
            W.Hy[a * D + b] = s; W.Hy[b * D + a] = s;
        }
    (void)f;
    for (int a = 0; a < D; ++a) {
        for (int b = 0; b < P.nb; ++b) {
            double s = 0.0;
            for (int k = P.boff[b]; k < P.boff[b + 1]; ++k) { int j = P.bidx[k]; s += W.y[j] * W.Hy[a * D + j]; }
            W.bt[b] = s;
        }
        for (int j = 0; j < D; ++j) W.Mx[a * D + j] = W.y[j] * (W.Hy[a * D + j] - W.bt[P.blk[j]]);
    }
    for (int j = 0; j < D; ++j) {
        for (int b = 0; b < P.nb; ++b) {
            double s = 0.0;
            for (int k = P.boff[b]; k < P.boff[b + 1]; ++k) { int a = P.bidx[k]; s += W.y[a] * W.Mx[a * D + j]; }
            W.bt[b] = s;
        }
        for (int a = 0; a < D; ++a) W.Hz[a * D + j] = W.y[a] * (W.Mx[a * D + j] - W.bt[P.blk[a]]);
    }
    int m = 0;
    for (int b = 0; b < P.nb; ++b) {
        if (P.single[b]) continue;
        int i = P.bidx[P.boff[b]];
        double s = 0.0;
        for (int k = P.boff[b]; k < P.boff[b + 1]; ++k) {
            int j = P.bidx[k]; s += W.y[j] * W.gl[j];
            if (W.y[j] > W.y[i]) i = j;
        }
        for (int k = P.boff[b]; k < P.boff[b + 1]; ++k) {
            int j = P.bidx[k];
            if (j != i && !P.pinned[j] && !(W.y[j] <= 1.5 * CG_Y_FLOOR && W.gl[j] > s)) W.free_[m++] = j;
        }
    }
    if (m == 0) return 0;
    for (int p = 0; p < m; ++p) {
        double s = cg_fabs(W.Hz[W.free_[p] * D + W.free_[p]]);
        W.sc[p] = s > 1e-300 ? 1.0 / sqrt(s) : 1.0;
    }
    for (int p = 0; p < m; ++p)
        for (int q = 0; q < m; ++q) W.V[p * m + q] = W.sc[p] * W.Hz[W.free_[p] * D + W.free_[q]] * W.sc[q];
    for (int p = 0; p < m * m; ++p) W.Hs[p] = W.V[p];
    return m;
}

// Cholesky A = L L^T in place (lower triangle, row-major); 0 on success
CG_DEVICE static int cg_cholesky(double* A, int m) {
    for (int j = 0; j < m; ++j) {
        double s = A[j * m + j];
        for (int k = 0; k < j; ++k) s -= A[j * m + k] * A[j * m + k];
        if (!(s > 0)) return 1;
        double l = sqrt(s);
        A[j * m + j] = l;
        for (int i = j + 1; i < m; ++i) {
            double t = A[i * m + j];
            for (int k = 0; k < j; ++k) t -= A[i * m + k] * A[j * m + k];
            A[i * m + j] = t / l;
        }
    }
    return 0;
}

CG_DEVICE static void cg_cholesky_solve(const double* L, int m, double* x) {
    for (int i = 0; i < m; ++i) {
        double s = x[i];
        for (int k = 0; k < i; ++k) s -= L[i * m + k] * x[k];
        x[i] = s / L[i * m + i];
    }
    for (int i = m - 1; i >= 0; --i) {
        double s = x[i];
        for (int k = i + 1; k < m; ++k) s -= L[k * m + i] * x[k];
        x[i] = s / L[i * m + i];
    }
}

// symmetric eigenproblem by cyclic Jacobi: A (destroyed) = sum_k w_k v_k v_k^T; eigenvalues ascending in w,
// eigenvector k in row k of E. 0 on success
CG_DEVICE static int cg_eigh(double* A, int m, double* w, double* E) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j) E[i * m + j] = i == j ? 1.0 : 0.0;
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0, tot = 0.0;
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < m; ++j) {
                double a2 = A[i * m + j] * A[i * m + j];
                tot += a2;
                if (i != j) off += a2;
            }
        if (off <= 1e-30 * tot || off == 0.0) break;
        for (int p = 0; p < m - 1; ++p)
            for (int q = p + 1; q < m; ++q) {
                double apq = A[p * m + q];
                if (cg_fabs(apq) < 1e-300) continue;
                double theta = (A[q * m + q] - A[p * m + p]) / (2 * apq);
                double t = (theta >= 0 ? 1.0 : -1.0) / (cg_fabs(theta) + sqrt(theta * theta + 1));
                double c = 1 / sqrt(t * t + 1), s = t * c;
                for (int k = 0; k < m; ++k) {   // columns p, q
                    double akp = A[k * m + p], akq = A[k * m + q];
                    A[k * m + p] = c * akp - s * akq; A[k * m + q] = s * akp + c * akq;
                }
                for (int k = 0; k < m; ++k) {   // rows p, q
                    double apk = A[p * m + k], aqk = A[q * m + k];
                    A[p * m + k] = c * apk - s * aqk; A[q * m + k] = s * apk + c * aqk;
                }
                for (int k = 0; k < m; ++k) {   // eigenvectors (rows of E)
                    double ep = E[p * m + k], eq = E[q * m + k];
                    E[p * m + k] = c * ep - s * eq; E[q * m + k] = s * ep + c * eq;
                }
            }
    }
    for (int i = 0; i < m; ++i) w[i] = A[i * m + i];
    for (int i = 0; i < m; ++i) {   // ascending (selection sort, with the vectors)
        int k = i;
        for (int j = i + 1; j < m; ++j) if (w[j] < w[k]) k = j;
        if (k != i) {
            double t = w[i]; w[i] = w[k]; w[k] = t;
            for (int j = 0; j < m; ++j) { t = E[i * m + j]; E[i * m + j] = E[k * m + j]; E[k * m + j] = t; }
        }
    }
    for (int i = 0; i < m; ++i) if (!(w[i] == w[i])) return 1;
    return 0;
}

// factor the scaled Hessian (V): Cholesky, else of its eigenvalue-modified version (|w| floored at 1e-8)
template <class WT>
CG_DEVICE static int cg_factor(WT& W, int m) {
    W.last_indef = cg_cholesky(W.V, m) != 0;
    if (!W.last_indef) return 0;
    for (int p = 0; p < m * m; ++p) W.V[p] = W.Hs[p];
    if (cg_eigh(W.V, m, W.w, W.Ev)) return -1;
    for (int p = 0; p < m; ++p)
        for (int q = 0; q < m; ++q) {
            double s = 0.0;
            for (int k = 0; k < m; ++k) {
                double aw = cg_fabs(W.w[k]);
                if (aw < 1e-8) aw = 1e-8;
                s += W.Ev[k * m + p] * aw * W.Ev[k * m + q];
            }
            W.Hy[p * m + q] = s;
        }
    for (int p = 0; p < m * m; ++p) W.V[p] = W.Hy[p];
    return cg_cholesky(W.V, m) == 0 ? 0 : -1;
}

template <class WT>
CG_DEVICE static int cg_symmetric(const CgPhase& P, const WT& W) {
    for (int k = 0; k < P.nsym; ++k) {
        int same = 1;
        for (int q = P.sym_off[k]; q < P.sym_off[k + 1]; ++q)
            if (cg_fabs(W.y[P.sym_a[q]] - W.y[P.sym_b[q]]) > 1e-7) { same = 0; break; }
        if (same) return 1;
    }
    return 0;
}

// at a stationary point: along a direction of negative curvature to a lower point (see escape_saddle)
template <class WT>
CG_DEVICE static int cg_escape_saddle(const CgPhase& P, WT& W, double* f) {
    const int D = P.D;
    double best = *f;
    int m = cg_build_hessian(P, W);
    if (m == 0) return 0;
    if (cg_cholesky(W.V, m) == 0) return 0;
    for (int p = 0; p < m * m; ++p) W.V[p] = W.Hs[p];
    if (cg_eigh(W.V, m, W.w, W.Ev) || W.w[0] > -1e-6) return 0;
    double s = 0.0;
    for (int p = 0; p < D; ++p) W.d[p] = 0.0;
    for (int p = 0; p < m; ++p) {
        W.d[W.free_[p]] = W.sc[p] * W.Ev[p];
        if (cg_fabs(W.d[W.free_[p]]) > s) s = cg_fabs(W.d[W.free_[p]]);
    }
    if (!(s > 0)) return 0;
    for (int p = 0; p < D; ++p) W.d[p] *= 0.5 / s;
    double t = 1.0;
    for (int k = 0; k < 12; ++k) {
        for (int sgn = 0; sgn < 2; ++sgn) {
            for (int p = 0; p < D; ++p) W.zt[p] = W.z[p] + (sgn == 0 ? t : -t) * W.d[p];
            double ft = cg_fgz(P, W, W.zt, W.yt, W.gzt, W.gyt);
            if (CG_ISFINITE(ft) && ft < best - 1e-12) {
                for (int p = 0; p < D; ++p) { W.z[p] = W.zt[p]; W.y[p] = W.yt[p]; W.gz[p] = W.gzt[p]; W.gy[p] = W.gyt[p]; }
                *f = ft;
                return 1;
            }
        }
        t *= 0.5;
    }
    *f = cg_fgz(P, W, W.z, W.y, W.gz, W.gy);
    return 0;
}

template <class WT>
CG_DEVICE static int cg_newton_direction(const CgPhase& P, WT& W, int m) {
    for (int p = 0; p < m; ++p) W.coef[p] = -W.sc[p] * W.gz[W.free_[p]];
    cg_cholesky_solve(W.V, m, W.coef);
    for (int p = 0; p < P.D; ++p) W.d[p] = 0.0;
    for (int p = 0; p < m; ++p) W.d[W.free_[p]] = W.sc[p] * W.coef[p];
    return 0;
}

// minimize the TPD (mu in RT units) of phase P from y0 (see CPhase.price); returns 1 at a stationary point
template <class WT>
CG_DEVICE static int cg_price(const CgPhase& P, WT& W, const double* sv, const double* mu, const double* y0,
                              double* y_out, double* x_out, double* G_out, double* tpd_out) {
    const int D = P.D, C = P.C;
    W.last_indef = 0; W.evals1 = 0; W.evals2 = 0;
    for (int k = 0; k < CG_NS; ++k) W.inp[k] = sv[k];
    for (int j = 0; j < D; ++j) { double s = y0[j]; if (s < CG_Y_FLOOR) s = CG_Y_FLOOR; W.z[j] = log(s); }
    for (int i = 0; i < C; ++i) W.mu[i] = mu[i];
    for (int j = 0; j < D; ++j) {
        double s = 0.0;
        for (int i = 0; i < C; ++i) s += mu[i] * P.A[i * D + j];
        W.nu[j] = s;
    }
    double f = cg_fgz(P, W, W.z, W.y, W.gz, W.gy);
    int escapes = 0;
    for (int it = 0; it < 60; ++it) {
        if (cg_kkt(P, W) < 1e-9) {
            if (escapes < 3 && P.nsym > 0 && (it == 0 || W.last_indef) && cg_symmetric(P, W) && cg_escape_saddle(P, W, &f)) {
                escapes++;
                continue;
            }
            break;
        }
        int m = cg_build_hessian(P, W);
        if (m == 0) break;
        if (cg_factor(W, m) != 0) break;
        if (cg_newton_direction(P, W, m) != 0) break;
        double s = 0.0;
        for (int a = 0; a < D; ++a) if (cg_fabs(W.d[a]) > s) s = cg_fabs(W.d[a]);
        if (s > P.max_dz) for (int a = 0; a < D; ++a) W.d[a] *= P.max_dz / s;
        double slope = 0.0;
        for (int a = 0; a < D; ++a) slope += W.gz[a] * W.d[a];
        double step = 1.0, ft = 0.0;
        int accepted = 0;
        for (int ls = 0; ls < 30; ++ls) {
            for (int j = 0; j < D; ++j) W.zt[j] = W.z[j] + step * W.d[j];
            ft = cg_fgz(P, W, W.zt, W.yt, W.gzt, W.gyt);
            if (CG_ISFINITE(ft) && ft <= f + 1e-4 * step * (slope < 0 ? slope : 0.0) + 1e-12) { accepted = 1; break; }
            step *= 0.5;
        }
        if (!accepted) { f = cg_fgz(P, W, W.z, W.y, W.gz, W.gy); break; }
        for (int j = 0; j < D; ++j) { W.z[j] = W.zt[j]; W.y[j] = W.yt[j]; W.gz[j] = W.gzt[j]; W.gy[j] = W.gyt[j]; }
        f = ft;
    }
    // steepest-descent clean-up if Newton stalled
    double gnorm = cg_gznorm(P, W);
    int it = 0;
    while (cg_kkt(P, W) >= 1e-7 && it < 200 && CG_ISFINITE(gnorm)) {
        double step = 1.0 / (gnorm > 1.0 ? gnorm : 1.0), ft = 0.0;
        int accepted = 0;
        for (int ls = 0; ls < 40; ++ls) {
            for (int j = 0; j < D; ++j) W.zt[j] = W.z[j] - step * W.gz[j];
            ft = cg_fgz(P, W, W.zt, W.yt, W.gzt, W.gyt);
            if (CG_ISFINITE(ft) && ft <= f - 1e-4 * step * gnorm * gnorm) { accepted = 1; break; }
            step *= 0.5;
        }
        if (!accepted) break;
        for (int j = 0; j < D; ++j) { W.z[j] = W.zt[j]; W.y[j] = W.yt[j]; W.gz[j] = W.gzt[j]; W.gy[j] = W.gyt[j]; }
        f = ft;
        gnorm = cg_gznorm(P, W);
        it++;
    }
    cg_evaluate(P, W, W.y, 0);
    for (int j = 0; j < D; ++j) y_out[j] = W.y[j];
    for (int i = 0; i < C; ++i) x_out[i] = W.mval[i] / W.Mtot;
    double GM = W.G_fu / W.Mtot, s = GM / P.RT;
    for (int i = 0; i < C; ++i) s -= mu[i] * x_out[i];
    *tpd_out = s;
    *G_out = W.Mtot < P.m_lo ? NAN : GM;
    return cg_kkt(P, W) < 1e-7 ? 1 : 0;
}

// after cg_price: S = dx/dmu (C x C, row-major, mu in RT units) at the minimum = B H^-1 B^T over the free
// variables (see CPhase.sens_full, unconstrained). 0, or -1 if the Hessian there is not positive definite
template <class WT>
CG_DEVICE CG_NOINLINE static int cg_sens(const CgPhase& P, WT& W, double* S, double* Bm, double* Wm) {
    const int C = P.C;
    for (int i = 0; i < C * C; ++i) S[i] = 0.0;
    for (int j = 0; j < P.D; ++j) W.gl[j] = W.gy[j];
    int m = cg_build_hessian(P, W);
    if (m == 0) return 0;
    if (cg_cholesky(W.V, m) != 0) return -1;
    double M = W.Mtot, xs[CG_CMAX];
    for (int i = 0; i < C; ++i) xs[i] = W.mval[i] / M;
    for (int i = 0; i < C; ++i)
        for (int p = 0; p < m; ++p) {
            int k = W.free_[p], b = P.blk[k];
            double s = 0.0;
            for (int q = P.boff[b]; q < P.boff[b + 1]; ++q) {
                int j = P.bidx[q];
                s += (P.A[i * P.D + j] - xs[i] * P.asum[j]) / M * W.y[j];
            }
            Bm[i * m + p] = W.sc[p] * W.y[k] * ((P.A[i * P.D + k] - xs[i] * P.asum[k]) / M - s);
        }
    for (int i = 0; i < C; ++i) {   // Wm row i = H^-1 (row i of B)
        for (int p = 0; p < m; ++p) Wm[i * m + p] = Bm[i * m + p];
        cg_cholesky_solve(W.V, m, &Wm[i * m]);
    }
    for (int i = 0; i < C; ++i)
        for (int j = 0; j < C; ++j) {
            double s = 0.0;
            for (int p = 0; p < m; ++p) s += Bm[i * m + p] * Wm[j * m + p];
            S[i * C + j] = s;
        }
    return 0;
}

CG_DEVICE static void cg_phase_from_desc(CgPhase& P, int ph, const int* di, const double* dd, const int* ip,
                                         const double* dp, int C, double RT) {
    const int* d = di + ph * CGI_COUNT;
    P.D = d[CGI_D]; P.C = C; P.nb = d[CGI_NB]; P.model = d[CGI_MODEL]; P.nsym = d[CGI_NSYM];
    P.nder = d[CGI_NDER]; P.doff = d[CGI_DOFF];
    P.boff = ip + d[CGI_BOFF]; P.bidx = ip + d[CGI_BIDX]; P.blk = ip + d[CGI_BLK]; P.single = ip + d[CGI_SINGLE];
    P.pinned = ip + d[CGI_PINNED]; P.sym_a = ip + d[CGI_SYMA]; P.sym_b = ip + d[CGI_SYMB];
    P.sym_off = ip + d[CGI_SYMOFF];
    P.A = dp + d[CGI_A]; P.asum = dp + d[CGI_ASUM];
    P.m_lo = dd[ph * CGD_COUNT + CGD_MLO]; P.pen_w = dd[ph * CGD_COUNT + CGD_PENW];
    P.max_dz = dd[ph * CGD_COUNT + CGD_MAXDZ]; P.RT = RT;
}

// one job (index j of the job arrays): phase job_phase[j] (an index into the descriptions), potentials of row
// job_cond[j] of mu (C per row, RT units), start y0[j * ystride ...]
template <class WT>
CG_DEVICE static void cg_price_job(int j, const int* job_phase, const int* job_cond, const double* mu, const double* y0,
                                   int ystride, double* y_out, double* x_out, double* G_out, double* tpd_out,
                                   int* conv_out, int* evals_out, const int* di, const double* dd, const int* ip,
                                   const double* dp, int C, double RT, const double* sv, WT& W) {
    CgPhase P;
    cg_phase_from_desc(P, job_phase[j], di, dd, ip, dp, C, RT);
    conv_out[j] = cg_price(P, W, sv, mu + (long)job_cond[j] * C, y0 + (long)j * ystride, y_out + (long)j * ystride,
                           x_out + (long)j * C, G_out + j, tpd_out + j);
    if (evals_out) { evals_out[2 * j] = W.evals1; evals_out[2 * j + 1] = W.evals2; }
}
