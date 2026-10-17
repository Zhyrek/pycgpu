// cg's per-condition steps as portable C++ (Solver in _core.pyx), for a tile solved on the host or a GPU: the
// LP master problem over the shared point store, the gauge fix of its potentials. (The polish and the pricing
// jobs' generation and consumption follow the same pattern.)
//
// Matrices are column-major like LAPACK's, so the code reads as the Cython it ports; LAPACK's LU is replaced by
// an unblocked LU with partial pivoting (same pivot choices, rounding may differ).
#pragma once

#ifndef CG_DEVICE
#define CG_DEVICE
#endif

// LU with partial pivoting of the n x n column-major A (dgetf2); ipiv 0-based. 0, or k + 1 for a zero pivot
CG_DEVICE static int cg_getrf(int n, double* A, int* ipiv) {
    int info = 0;
    for (int j = 0; j < n; ++j) {
        int p = j;
        double amax = A[j * n + j] < 0 ? -A[j * n + j] : A[j * n + j];
        for (int i = j + 1; i < n; ++i) {
            double a = A[j * n + i] < 0 ? -A[j * n + i] : A[j * n + i];
            if (a > amax) { amax = a; p = i; }
        }
        ipiv[j] = p;
        if (A[j * n + p] != 0.0) {
            if (p != j)
                for (int k = 0; k < n; ++k) { double t = A[k * n + j]; A[k * n + j] = A[k * n + p]; A[k * n + p] = t; }
            double r = 1.0 / A[j * n + j];
            for (int i = j + 1; i < n; ++i) A[j * n + i] *= r;
        } else if (info == 0) {
            info = j + 1;
        }
        for (int k = j + 1; k < n; ++k) {
            double t = A[k * n + j];
            if (t != 0.0)
                for (int i = j + 1; i < n; ++i) A[k * n + i] -= A[j * n + i] * t;
        }
    }
    return info;
}

// solve A x = b (trans 0) or A^T x = b (trans 1) with the factors of cg_getrf; b overwritten by x
CG_DEVICE static void cg_getrs(int n, const double* LU, const int* ipiv, double* b, int trans) {
    if (!trans) {
        for (int i = 0; i < n; ++i) if (ipiv[i] != i) { double t = b[i]; b[i] = b[ipiv[i]]; b[ipiv[i]] = t; }
        for (int j = 0; j < n; ++j) for (int i = j + 1; i < n; ++i) b[i] -= LU[j * n + i] * b[j];   // L (unit)
        for (int j = n - 1; j >= 0; --j) { b[j] /= LU[j * n + j]; for (int i = 0; i < j; ++i) b[i] -= LU[j * n + i] * b[j]; }
    } else {
        for (int j = 0; j < n; ++j) {   // U^T
            double s = b[j];
            for (int i = 0; i < j; ++i) s -= LU[j * n + i] * b[i];
            b[j] = s / LU[j * n + j];
        }
        for (int j = n - 1; j >= 0; --j) {   // L^T (unit)
            double s = b[j];
            for (int i = j + 1; i < n; ++i) s -= LU[j * n + i] * b[i];
            b[j] = s;
        }
        for (int i = n - 1; i >= 0; --i) if (ipiv[i] != i) { double t = b[i]; b[i] = b[ipiv[i]]; b[ipiv[i]] = t; }
    }
}

// what an LP over the point store needs (pointers into the solver's arrays)
struct CgLP {
    int K, C, npts;
    const double *Px, *PG;          // store: compositions (npts x C), energies J/mol of atoms
    const double *R, *b, *mf;       // condition rows (K x C, row-major), right-hand sides, fixed potentials (RT units)
    int* basis;                     // K entries: points, or -(1 + k) for the artificial column of row k
    double *xB, *mu, *dual, *cB, *LU, *tmp;
    int* ipiv;
    double RT;
    long long* lp_iters;
};

CG_DEVICE static inline double cg_lp_cost(const CgLP& L, int j) {
    if (j < 0) return 1e3;
    double c = L.PG[j] / L.RT;
    for (int i = 0; i < L.C; ++i) c -= L.mf[i] * L.Px[(long)j * L.C + i];
    return c;
}

CG_DEVICE static inline void cg_lp_column(const CgLP& L, int j, double* out) {
    for (int k = 0; k < L.K; ++k) {
        if (j < 0) { out[k] = k == -(j + 1) ? 1.0 : 0.0; continue; }
        double s = 0.0;
        for (int i = 0; i < L.C; ++i) s += L.R[k * L.C + i] * L.Px[(long)j * L.C + i];
        out[k] = s;
    }
}

CG_DEVICE static inline int cg_lp_is_basic(const CgLP& L, int var) {
    for (int k = 0; k < L.K; ++k) if (L.basis[k] == var) return 1;
    return 0;
}

CG_DEVICE static inline int cg_lp_bland_rank(const CgLP& L, int var) { return var < 0 ? -(1 + var) : L.K + var; }

// revised primal simplex, warm-started from basis (see Solver.lp_solve); the potentials (RT units) in mu
CG_DEVICE static int cg_lp_solve(CgLP& L, double* obj) {
    const int K = L.K, C = L.C;
    int bland = 0, resets = 0;
    for (int it = 0; it < 20000; ++it) {
        (*L.lp_iters)++;
        for (int k = 0; k < K; ++k) cg_lp_column(L, L.basis[k], &L.LU[k * K]);
        if (cg_getrf(K, L.LU, L.ipiv) != 0) {
            if (resets >= 3) return 1;
            for (int k = 0; k < K; ++k) L.basis[k] = -(1 + k);
            resets++; bland = 0;
            continue;
        }
        for (int i = 0; i < K; ++i) L.xB[i] = L.b[i];
        cg_getrs(K, L.LU, L.ipiv, L.xB, 0);
        int infeasible = 0;
        for (int i = 0; i < K; ++i) if (L.xB[i] < -1e-9) infeasible = 1;
        if (infeasible) {
            if (resets >= 3) return 2;
            for (int k = 0; k < K; ++k) L.basis[k] = -(1 + k);
            resets++; bland = 0;
            continue;
        }
        for (int k = 0; k < K; ++k) L.cB[k] = cg_lp_cost(L, L.basis[k]);
        for (int k = 0; k < K; ++k) L.dual[k] = L.cB[k];
        cg_getrs(K, L.LU, L.ipiv, L.dual, 1);
        for (int i = 0; i < C; ++i) {
            double s = L.mf[i];
            for (int k = 0; k < K; ++k) s += L.R[k * C + i] * L.dual[k];
            L.mu[i] = s;
        }
        if (it > 50 * K + 100) bland = 1;
        double s = 1.0;
        for (int k = 0; k < K; ++k) { double a = L.cB[k] < 0 ? -L.cB[k] : L.cB[k]; if (a > s) s = a; }
        int q = 0, leave = -2;
        double best = -1e-10 * s;
        for (int k = 0; k < K; ++k) {
            double rc = 1e3 - L.dual[k];
            if (rc < best && !cg_lp_is_basic(L, -(1 + k))) {
                best = rc; q = -(1 + k); leave = -1;
                if (bland) break;
            }
        }
        if (leave == -2 || !bland) {
            for (int j = 0; j < L.npts; ++j) {
                double rc = L.PG[j] / L.RT;
                for (int i = 0; i < C; ++i) rc -= L.mu[i] * L.Px[(long)j * C + i];
                if (rc < best && !cg_lp_is_basic(L, j)) {
                    best = rc; q = j; leave = -1;
                    if (bland) break;
                }
            }
        }
        if (leave == -2) {
            *obj = 0.0;
            for (int k = 0; k < K; ++k) *obj += L.cB[k] * L.xB[k];
            return 0;
        }
        cg_lp_column(L, q, L.tmp);
        cg_getrs(K, L.LU, L.ipiv, L.tmp, 0);
        double piv = 1e-11;
        for (int i = 0; i < K; ++i) { double a = L.tmp[i] < 0 ? -L.tmp[i] : L.tmp[i]; if (1e-9 * a > piv) piv = 1e-9 * a; }
        leave = -1;
        double tmin = INFINITY;
        for (int i = 0; i < K; ++i)
            if (L.tmp[i] > piv) {
                double t = (L.xB[i] > 0 ? L.xB[i] : 0.0) / L.tmp[i];
                if (t < tmin) { tmin = t; leave = i; }
            }
        if (leave < 0) return 3;
        if (bland) {
            for (int i = 0; i < K; ++i)
                if (L.tmp[i] > piv) {
                    double t = (L.xB[i] > 0 ? L.xB[i] : 0.0) / L.tmp[i];
                    if (t <= tmin + 1e-12 * (1.0 + tmin) &&
                        cg_lp_bland_rank(L, L.basis[i]) < cg_lp_bland_rank(L, L.basis[leave]))
                        leave = i;
                }
        }
        L.basis[leave] = q;
    }
    return 4;
}

// no component along gauge directions (rows of W, ngauge x C, orthonormal)
CG_DEVICE static void cg_gauge_fix(double* mu, const double* W, int ngauge, int C) {
    for (int k = 0; k < ngauge; ++k) {
        double s = 0.0;
        for (int i = 0; i < C; ++i) s += W[k * C + i] * mu[i];
        for (int i = 0; i < C; ++i) mu[i] -= s * W[k * C + i];
    }
}

// ---- the polish: Newton on (mu, amounts) for the LP solution's points (Solver._polish_core); needs cg_price.h

// minimum-norm least squares A x = b, A n x n column-major, singular values below rcond * the largest taken as
// zero (dgelss), by one-sided Jacobi; U, V: n * n work; b overwritten by x. 0 on success
CG_DEVICE static int cg_lstsq(int n, const double* A, double* b, double rcond, double* U, double* V, double* sv,
                              double* tmp) {
    for (int i = 0; i < n * n; ++i) U[i] = A[i];
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) V[j * n + i] = i == j ? 1.0 : 0.0;
    for (int sweep = 0; sweep < 60; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < n - 1; ++p)
            for (int q = p + 1; q < n; ++q) {
                double a = 0.0, bb = 0.0, g = 0.0;
                for (int i = 0; i < n; ++i) {
                    double up = U[p * n + i], uq = U[q * n + i];
                    a += up * up; bb += uq * uq; g += up * uq;
                }
                if (g == 0.0) continue;
                double r = (g < 0 ? -g : g) / sqrt(a * bb);
                if (r > off) off = r;
                if (r < 1e-15) continue;
                double zeta = (bb - a) / (2 * g);
                double t = (zeta >= 0 ? 1.0 : -1.0) / ((zeta < 0 ? -zeta : zeta) + sqrt(1 + zeta * zeta));
                double c = 1 / sqrt(1 + t * t), s = c * t;
                for (int i = 0; i < n; ++i) {
                    double up = U[p * n + i], uq = U[q * n + i];
                    U[p * n + i] = c * up - s * uq; U[q * n + i] = s * up + c * uq;
                    double vp = V[p * n + i], vq = V[q * n + i];
                    V[p * n + i] = c * vp - s * vq; V[q * n + i] = s * vp + c * vq;
                }
            }
        if (off < 1e-15) break;
    }
    double smax = 0.0;
    for (int j = 0; j < n; ++j) {
        double s = 0.0;
        for (int i = 0; i < n; ++i) s += U[j * n + i] * U[j * n + i];
        sv[j] = sqrt(s);
        if (sv[j] > smax) smax = sv[j];
    }
    if (!(smax == smax)) return 1;
    for (int j = 0; j < n; ++j) {   // tmp_j = (u_j . b) / s_j^2 (u_j unnormalized: s_j u_hat_j)
        double s = 0.0;
        for (int i = 0; i < n; ++i) s += U[j * n + i] * b[i];
        tmp[j] = sv[j] > rcond * smax && sv[j] > 0 ? s / (sv[j] * sv[j]) : 0.0;
    }
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int j = 0; j < n; ++j) s += V[j * n + i] * tmp[j];
        b[i] = s;
    }
    return 0;
}

// a condition's polish state and what it needs (pointers into the solver's arrays)
struct CgPolish {
    int C, K, Dm;                   // Dm: the row stride of pol_y
    const double *R, *b, *mf;       // condition rows (K x C), right-hand sides, fixed potentials (RT units)
    const int* fixed;
    const double* gauge; int ngauge;
    double *pol_y, *pol_x, *pol_S, *pol_J, *pol_F, *pol_mu, *pol_n, *pol_G, *pol_t, *pol_J2, *pol_F2, *pol_sv;
    double *pol_U, *pol_V, *pol_B, *pol_W;   // (lstsq and sensitivity work: 4 C^2, 4 C^2, C Dm, C Dm)
    int *pol_ph, *pol_ip;
    int pol_na, polish_fail;
    // phases (cg_price.h descriptions) and state
    const int* di; const double* dd; const int* ip; const double* dp; const double* sv; double RT;
};

// Newton from the state in pol_* (na points); see Solver._polish_core. Returns 1 if it converged (pol_na points,
// pol_mu), else 0 with polish_fail set
template <class WT>
CG_DEVICE static int cg_polish_core(CgPolish& S, int na, int max_it, int allow_drop, WT& W) {
    const int C = S.C, K = S.K, Dm = S.Dm;
    double* py = S.pol_y; double* px = S.pol_x; double* pS = S.pol_S; double* pJ = S.pol_J; double* pF = S.pol_F;
    double last_res = INFINITY;
    if (allow_drop) {
        while (na > K) {
            int adrop = 0;
            for (int a = 1; a < na; ++a) if (S.pol_n[a] < S.pol_n[adrop]) adrop = a;
            for (int a = adrop; a < na - 1; ++a) {
                S.pol_ph[a] = S.pol_ph[a + 1]; S.pol_n[a] = S.pol_n[a + 1];
                for (int j = 0; j < Dm; ++j) py[a * Dm + j] = py[(a + 1) * Dm + j];
            }
            na -= 1;
        }
    }
    int it;
    for (it = 0; it < max_it; ++it) {
        for (int a = 0; a < na; ++a) {
            CgPhase P;
            cg_phase_from_desc(P, S.pol_ph[a], S.di, S.dd, S.ip, S.dp, C, S.RT);
            double yb[CG_DMAX], G = 0.0, t = 0.0;
            if (!cg_price(P, W, S.sv, S.pol_mu, &py[a * Dm], yb, &px[a * C], &G, &t)) { S.polish_fail = 1; return 0; }
            if (!CG_ISFINITE(G)) return 0;
            for (int j = 0; j < P.D; ++j) py[a * Dm + j] = yb[j];
            S.pol_G[a] = G; S.pol_t[a] = t;
            if (cg_sens(P, W, &pS[a * C * C], S.pol_B, S.pol_W) < 0) { S.polish_fail = 2; return 0; }
        }
        // merge composition sets that met, and ordering variants of one state
        int w = 0;
        for (int a = 0; a < na; ++a) {
            int merged = 0;
            int D = S.di[S.pol_ph[a] * CGI_COUNT + CGI_D];
            for (int bb = 0; bb < w; ++bb) {
                if (S.pol_ph[bb] != S.pol_ph[a]) continue;
                double s = 0.0;
                for (int j = 0; j < D; ++j) { double d = py[bb * Dm + j] - py[a * Dm + j]; d = d < 0 ? -d : d; if (d > s) s = d; }
                double dG = S.pol_G[bb] - S.pol_G[a]; dG = dG < 0 ? -dG : dG;
                double aG = S.pol_G[a] < 0 ? -S.pol_G[a] : S.pol_G[a];
                if (s >= 1e-6 && dG < 1e-9 * (1.0 + aG)) {
                    s = 0.0;
                    for (int i = 0; i < C; ++i) { double d = px[bb * C + i] - px[a * C + i]; d = d < 0 ? -d : d; if (d > s) s = d; }
                    s *= 1e3;
                }
                if (s < 1e-6) { S.pol_n[bb] += S.pol_n[a]; merged = 1; break; }
            }
            if (!merged) {
                if (w != a) {
                    S.pol_ph[w] = S.pol_ph[a]; S.pol_n[w] = S.pol_n[a]; S.pol_G[w] = S.pol_G[a]; S.pol_t[w] = S.pol_t[a];
                    for (int j = 0; j < Dm; ++j) py[w * Dm + j] = py[a * Dm + j];
                    for (int i = 0; i < C; ++i) px[w * C + i] = px[a * C + i];
                    for (int i = 0; i < C * C; ++i) pS[w * C * C + i] = pS[a * C * C + i];
                }
                w += 1;
            }
        }
        na = w;
        // residuals
        double res = 0.0;
        for (int a = 0; a < na; ++a) { pF[a] = S.pol_t[a]; double r = pF[a] < 0 ? -pF[a] : pF[a]; if (r > res) res = r; }
        for (int k = 0; k < K; ++k) {
            double s = -S.b[k];
            for (int a = 0; a < na; ++a) for (int i = 0; i < C; ++i) s += S.pol_n[a] * S.R[k * C + i] * px[a * C + i];
            pF[na + k] = s;
            double r = s < 0 ? -s : s; if (r > res) res = r;
        }
        int row = na + K;
        for (int i = 0; i < C; ++i)
            if (S.fixed[i]) {
                pF[row] = S.pol_mu[i] - S.mf[i];
                double r = pF[row] < 0 ? -pF[row] : pF[row]; if (r > res) res = r;
                row += 1;
            }
        if (res < 1e-11 || (res < 1e-8 && res > 0.1 * last_res)) break;
        last_res = res;
        // Jacobian (column-major), unknowns [mu (C), n (na)]
        const int N = na + C;
        for (int i = 0; i < N * N; ++i) pJ[i] = 0.0;
        for (int a = 0; a < na; ++a) {
            for (int i = 0; i < C; ++i) pJ[i * N + a] = -px[a * C + i];
            for (int k = 0; k < K; ++k) {
                double s = 0.0;
                for (int i = 0; i < C; ++i) s += S.R[k * C + i] * px[a * C + i];
                pJ[(C + a) * N + na + k] = s;
            }
        }
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < C; ++j) {
                double s = 0.0;
                for (int a = 0; a < na; ++a) for (int i = 0; i < C; ++i) s += S.pol_n[a] * S.R[k * C + i] * pS[a * C * C + i * C + j];
                pJ[j * N + na + k] = s;
            }
        row = na + K;
        for (int i = 0; i < C; ++i) if (S.fixed[i]) { pJ[i * N + row] = 1.0; row += 1; }
        for (int i = 0; i < N; ++i) pF[i] = -pF[i];
        for (int i = 0; i < N * N; ++i) S.pol_J2[i] = pJ[i];
        for (int i = 0; i < N; ++i) S.pol_F2[i] = pF[i];
        if (cg_getrf(N, pJ, S.pol_ip) == 0) {
            cg_getrs(N, pJ, S.pol_ip, pF, 0);
        } else {
            // dependent rows: least squares
            for (int i = 0; i < N; ++i) pF[i] = S.pol_F2[i];
            if (cg_lstsq(N, S.pol_J2, pF, 1e-10, S.pol_U, S.pol_V, S.pol_sv, S.pol_F2)) { S.polish_fail = 3; return 0; }
        }
        for (int k = 0; k < S.ngauge; ++k) {
            double s = 0.0;
            for (int i = 0; i < C; ++i) s += S.gauge[k * C + i] * pF[i];
            for (int i = 0; i < C; ++i) pF[i] -= s * S.gauge[k * C + i];
        }
        double s = 0.0;
        for (int i = 0; i < C; ++i) { double a = pF[i] < 0 ? -pF[i] : pF[i]; if (a > s) s = a; }
        double sc = s > 5.0 ? 5.0 / s : 1.0;
        int adrop = -1;
        if (allow_drop) {
            for (int a = 0; a < na; ++a)
                if (S.pol_n[a] + sc * pF[C + a] < 0 && pF[C + a] < 0) {
                    double alpha = S.pol_n[a] / -pF[C + a];
                    if (alpha < sc) { sc = alpha; adrop = a; }
                }
        }
        for (int i = 0; i < C; ++i) S.pol_mu[i] += sc * pF[i];
        for (int a = 0; a < na; ++a) {
            S.pol_n[a] += sc * pF[C + a];
            if (S.pol_n[a] < -1e-9 && !allow_drop) { S.polish_fail = 4; return 0; }
        }
        if (adrop >= 0) {
            if (na == 1) { S.polish_fail = 5; return 0; }
            for (int a = adrop; a < na - 1; ++a) {
                S.pol_ph[a] = S.pol_ph[a + 1]; S.pol_n[a] = S.pol_n[a + 1];
                for (int j = 0; j < Dm; ++j) py[a * Dm + j] = py[(a + 1) * Dm + j];
            }
            na -= 1;
            last_res = INFINITY;
        }
    }
    if (it >= max_it) { S.polish_fail = 6; return 0; }
    S.pol_na = na;
    for (int a = 0; a < na; ++a) if (S.pol_n[a] < 0) S.pol_n[a] = 0.0;
    return 1;
}

// the point store as the polish reads it
struct CgStoreView {
    const double *Py, *Px, *PG;
    const int* Pph;
    int Dm;   // row stride of Py
};

// the LP solution's points as the polish's starting state (Solver._polish_start); -1 if an artificial column is in use
CG_DEVICE static int cg_polish_start(CgPolish& S, const CgStoreView& st, const int* basis, const double* xB,
                                     const double* mu) {
    int na = 0;
    for (int c = 0; c < S.C; ++c) S.pol_mu[c] = mu[c];
    for (int k = 0; k < S.K; ++k) {
        if (!(xB[k] > 1e-14)) continue;
        int p = basis[k];
        if (p < 0) {
            if (xB[k] > 1e-9) return -1;
            continue;
        }
        S.pol_ph[na] = st.Pph[p]; S.pol_n[na] = xB[k];
        int D = S.di[st.Pph[p] * CGI_COUNT + CGI_D];
        for (int j = 0; j < D; ++j) S.pol_y[na * S.Dm + j] = st.Py[(long)p * st.Dm + j];
        na++;
    }
    return na;
}

// the polish from the LP solution (Solver.polish): again with dropping if an amount went negative
template <class WT>
CG_DEVICE static int cg_polish(CgPolish& S, const CgStoreView& st, const int* basis, const double* xB, const double* mu,
                               WT& W) {
    int na = cg_polish_start(S, st, basis, xB, mu);
    if (na <= 0) return 0;
    if (cg_polish_core(S, na, 40, 0, W) == 1) return 1;
    if (S.polish_fail != 4) return 0;
    na = cg_polish_start(S, st, basis, xB, mu);
    if (na <= 0) return 0;
    return cg_polish_core(S, na, 40, 1, W);
}

// Newton from the last polished state plus a point that failed its certification (phase, y) at zero amount
// (Solver._polish_augment)
template <class WT>
CG_DEVICE static int cg_polish_augment(CgPolish& S, int cert_phase, const double* ycert, WT& W) {
    int na = S.pol_na;
    if (na <= 0 || na + 1 > S.K) return 0;
    S.pol_ph[na] = cert_phase; S.pol_n[na] = 0.0;
    int D = S.di[cert_phase * CGI_COUNT + CGI_D];
    for (int j = 0; j < D; ++j) S.pol_y[na * S.Dm + j] = ycert[j];
    return cg_polish_core(S, na + 1, 40, 1, W) == 1;
}
