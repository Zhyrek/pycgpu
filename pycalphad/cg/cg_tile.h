// A tile of conditions solved by cg's lockstep rounds (TileSolver.solve_tile_batched in _core.pyx) in portable
// C++: every per-condition step is a function of the tile state and a condition index (one thread per condition
// on a GPU, a loop on the host); the pricing jobs between them run as one batch (cg_price_job). Needs cg_price.h
// and cg_round.h.
//
// The store is shared by the tile's conditions; during a round each condition adds points only to its own slab
// (seen by it alone), and at the end of the round the slabs are merged into the store, condition by condition
// (cg_tile_merge), as TileSolver does: the results are those of the Cython solver to rounding.
#pragma once

enum { CG_IDLE = 0, CG_MAIN = 1, CG_AUG = 2, CG_FINAL = 3, CG_NEED_AUG = 4, CG_NEED_FINAL = 5 };

struct CgTile {
    // sizes and settings
    int B, C, K, Dm, nph, ngauge, max_late, use_polish;
    double RT, prune, prune_late, skip_margin, tol, seed_prune;
    // phases (cg_price.h descriptions), state variables, gauge directions (ngauge x C)
    const int* di; const double* dd; const int* ip; const double* dp; const double* sv; const double* gauge;
    // shared store: points 0..nshared-1; per-phase lists, newest first (head, Pnext)
    double *Py, *Px, *PG; int *Pph, *Pnext, *head; long long* Pcode; int nshared, cap;
    // own slabs: condition t's at rows own_off[t] .. own_off[t + 1] - 1 (its capacity), nown[t] of them used
    double *Oy, *Ox, *OG; int* Oph; long long* Ocode; int* nown; const long* own_off;
    // conditions (scaled rows as Solver.set_conditions makes them): R (B x K x C), b (B x K), mf (B x C), fixed
    const double *R, *b, *mf; const int* fixed;
    // seeds of the first condition's first round: rows of seedY (Dm stride), seed_off per phase (nph + 1)
    const double* seedY; const int* seed_off;
    // per-condition state
    int* basis; double *xB, *mu; unsigned char *has_last, *nopol, *polished, *skipped, *added, *done;
    double *last_tpd, *last_mu;   // (B x nph, B x nph x C)
    double *pol_y, *pol_x, *pol_n, *pol_mu, *pol_G; int *pol_ph, *pol_na;   // (B x C Dm, B x C C, B x C ...)
    int *cert_kind, *cert_phase, *aug, *kind; double *cert_tpd, *ycert;
    long long *lp_iters, *priced;   // (per condition counters)
    // pricing jobs and their results; per (condition, phase) the range of its jobs
    int *j_cond, *j_phase, *j_keep, *j_conv; double *j_y0, *j_y, *j_x, *j_G, *j_tpd;
    int *cp_phase, *cp_start, *cp_end, *cp_first, *cp_last;
    int nj, ncp, jcap, cpcap;
    // results (as Solver.solve_sequence): status, points, their phases, amounts, X, Y, GM, and mu (J/mol)
    int *status, *nout, *phs; double *amt, *X, *Y, *G, *mu_out;
};

// ---- the store as one condition sees it: shared points, then its own

CG_DEVICE static inline int cg_npts(const CgTile& T, int t) { return T.nshared + T.nown[t]; }
CG_DEVICE static inline double* cg_py(const CgTile& T, int t, int i) {
    return i < T.nshared ? T.Py + (long)i * T.Dm : T.Oy + (T.own_off[t] + i - T.nshared) * T.Dm;
}
CG_DEVICE static inline double* cg_px(const CgTile& T, int t, int i) {
    return i < T.nshared ? T.Px + (long)i * T.C : T.Ox + (T.own_off[t] + i - T.nshared) * T.C;
}
CG_DEVICE static inline double cg_pg(const CgTile& T, int t, int i) {
    return i < T.nshared ? T.PG[i] : T.OG[T.own_off[t] + i - T.nshared];
}
CG_DEVICE static inline int cg_pph(const CgTile& T, int t, int i) {
    return i < T.nshared ? T.Pph[i] : T.Oph[T.own_off[t] + i - T.nshared];
}
CG_DEVICE static inline long long cg_pcode(const CgTile& T, int t, int i) {
    return i < T.nshared ? T.Pcode[i] : T.Ocode[T.own_off[t] + i - T.nshared];
}
// the points of phase ph, newest first (Solver's per-phase list): own points from the last, then the shared list
CG_DEVICE static inline int cg_first(const CgTile& T, int t, int ph) {
    for (int k = T.nown[t] - 1; k >= 0; --k) if (T.Oph[T.own_off[t] + k] == ph) return T.nshared + k;
    return T.head[ph];
}
CG_DEVICE static inline int cg_next(const CgTile& T, int t, int ph, int i) {
    if (i >= T.nshared) {
        for (int k = i - T.nshared - 1; k >= 0; --k) if (T.Oph[T.own_off[t] + k] == ph) return T.nshared + k;
        return T.head[ph];
    }
    return T.Pnext[i];
}

// ordering pattern: the dominant species on each sublattice (CPhase.pattern)
CG_DEVICE static long long cg_pattern(const CgPhase& P, const double* y) {
    long long code = 0, mult = 1;
    for (int b = 0; b < P.nb; ++b) {
        int best = 0; double m = -1.0;
        for (int k = P.boff[b]; k < P.boff[b + 1]; ++k) if (y[P.bidx[k]] > m) { m = y[P.bidx[k]]; best = k - P.boff[b]; }
        code += best * mult;
        mult *= (P.boff[b + 1] - P.boff[b]);
    }
    return code;
}

// each mixing sublattice set to the composition comp (CPhase.composition_start)
CG_DEVICE static void cg_composition_start(const CgPhase& P, const int* elem_of, const double* comp, double* y) {
    for (int j = 0; j < P.D; ++j) {
        int e = elem_of[j];
        y[j] = 1e-6;
        if (e >= 0 && comp[e] > 1e-6) y[j] = comp[e];
    }
    for (int b = 0; b < P.nb; ++b) {
        if (P.single[b]) { y[P.bidx[P.boff[b]]] = 1.0; continue; }
        double s = 0.0;
        for (int k = P.boff[b]; k < P.boff[b + 1]; ++k) s += y[P.bidx[k]];
        for (int k = P.boff[b]; k < P.boff[b + 1]; ++k) y[P.bidx[k]] /= s;
    }
}

CG_DEVICE static inline void cg_phase(const CgTile& T, CgPhase& P, int ph) {
    cg_phase_from_desc(P, ph, T.di, T.dd, T.ip, T.dp, T.C, T.RT);
}

// a point of phase ph into condition t's slab unless it duplicates a point it sees (Solver.add_point): its index,
// or -1
CG_DEVICE static int cg_add_point(CgTile& T, int t, int ph, const double* y, const double* x, double G) {
    if (!CG_ISFINITE(G)) return -1;
    for (int i = 0; i < T.C; ++i) if (!CG_ISFINITE(x[i])) return -1;
    CgPhase P; cg_phase(T, P, ph);
    for (int p = cg_first(T, t, ph); p >= 0; p = cg_next(T, t, ph, p)) {
        const double* q = cg_py(T, t, p);
        double m = 0.0;
        for (int j = 0; j < P.D; ++j) { double d = q[j] - y[j]; d = d < 0 ? -d : d; if (d > m) m = d; }
        if (m < 1e-7) return -1;
    }
    int k = T.nown[t];
    if (T.own_off[t] + k >= T.own_off[t + 1]) return -1;   // (the slab is full: the point is not kept)
    long o = T.own_off[t] + k;
    for (int j = 0; j < T.Dm; ++j) T.Oy[o * T.Dm + j] = j < P.D ? y[j] : 0.0;
    for (int i = 0; i < T.C; ++i) T.Ox[o * T.C + i] = x[i];
    T.OG[o] = G; T.Oph[o] = ph; T.Ocode[o] = cg_pattern(P, y);
    T.nown[t] = k + 1;
    return T.nshared + k;
}

// ---- pricing jobs

CG_DEVICE static inline int cg_push_job(CgTile& T, int t, int ph, const double* y, int D, int keep) {
    if (!T.j_cond) { T.nj++; return 1; }   // (no job buffers: only counting, as the device driver sizes ranges)
    if (T.nj >= T.jcap) return 0;   // (the caller sized the buffers; see cg_tile_job_bound)
    int j = T.nj++;
    T.j_cond[j] = t; T.j_phase[j] = ph; T.j_keep[j] = keep;
    for (int i = 0; i < D; ++i) T.j_y0[(long)j * T.Dm + i] = y[i];
    return 1;
}

// the most jobs one condition can list in one sub-round (with seeds: plus the first round's seeds)
CG_DEVICE static inline int cg_tile_job_bound(const CgTile& T, int with_seeds) {
    return T.nph * (1 + T.K + 6 + 1) + (with_seeds ? T.seed_off[T.nph] : 0);
}

// condition t's pricing jobs at T.mu[t] (TileSolver._gen_jobs / Solver._pricing_round's starts), phase by phase:
// skip rule; the best stored point, the LP solution's points of the phase, the best of other ordering patterns,
// a composition start in a first round; with use_seeds, also the seeds (stored if new even when above the plane)
template <class WT>
CG_DEVICE static void cg_gen_jobs(CgTile& T, int t, int rounds, int use_seeds, WT& W) {
    const int C = T.C, K = T.K;
    const double* mu = T.mu + (long)t * C;
    long long codes[64]; int code_pt[64]; double code_tpd[64];
    double ystart[CG_DMAX];
    T.cp_first[t] = T.ncp;
    for (int ph = 0; ph < T.nph; ++ph) {
        const int* d = T.di + ph * CGI_COUNT;
        if (d[CGI_STOICH] || d[CGI_D] == 0) continue;
        CgPhase P; cg_phase(T, P, ph);
        int k = -1, ncodes = 0;
        double best_tpd = INFINITY;
        for (int p = cg_first(T, t, ph); p >= 0; p = cg_next(T, t, ph, p)) {
            double tt = cg_pg(T, t, p) / T.RT;
            const double* x = cg_px(T, t, p);
            for (int i = 0; i < C; ++i) tt -= mu[i] * x[i];
            if (tt < best_tpd) { best_tpd = tt; k = p; }
            if (tt < (rounds == 1 ? T.prune : T.prune_late)) {
                long long code = cg_pcode(T, t, p);
                int si;
                for (si = 0; si < ncodes; ++si)
                    if (codes[si] == code) { if (tt < code_tpd[si]) { code_tpd[si] = tt; code_pt[si] = p; } break; }
                if (si == ncodes && ncodes < 64) { codes[ncodes] = code; code_tpd[ncodes] = tt; code_pt[ncodes] = p; ncodes++; }
            }
        }
        long sk = (long)t * T.nph + ph;
        if (T.has_last[sk] && best_tpd > T.skip_margin) {
            double maxdmu = 0.0;
            for (int i = 0; i < C; ++i) {
                double a = mu[i] - T.last_mu[sk * C + i]; a = a < 0 ? -a : a;
                if (a > maxdmu) maxdmu = a;
            }
            if (T.last_tpd[sk] - maxdmu > T.skip_margin) continue;
        }
        int seeds_here = use_seeds && rounds == 1;
        if (k < 0 && !seeds_here) continue;
        int cp = T.ncp++;
        T.cp_phase[cp] = ph; T.cp_start[cp] = T.nj;
        if (k >= 0) {
            cg_push_job(T, t, ph, cg_py(T, t, k), P.D, 0);
            for (int si = 0; si < ncodes; ++si)
                for (int sj = si + 1; sj < ncodes; ++sj)
                    if (code_tpd[sj] < code_tpd[si]) {
                        double tt = code_tpd[si]; code_tpd[si] = code_tpd[sj]; code_tpd[sj] = tt;
                        int pp = code_pt[si]; code_pt[si] = code_pt[sj]; code_pt[sj] = pp;
                    }
            const int* basis = T.basis + (long)t * K;
            const double* xB = T.xB + (long)t * K;
            for (int si = 0; si < K; ++si) {
                int p = basis[si];
                if (p >= 0 && p != k && cg_pph(T, t, p) == ph && xB[si] > 1e-12) cg_push_job(T, t, ph, cg_py(T, t, p), P.D, 0);
            }
            int lim = rounds == 1 ? 6 : T.max_late;
            for (int si = 0; si < ncodes && si < lim; ++si)
                if (code_pt[si] != k && cg_pcode(T, t, code_pt[si]) != cg_pcode(T, t, k))
                    cg_push_job(T, t, ph, cg_py(T, t, code_pt[si]), P.D, 0);
            if (rounds == 1) {
                cg_composition_start(P, T.ip + d[CGI_ELEMOF], cg_px(T, t, k), ystart);
                cg_push_job(T, t, ph, ystart, P.D, 0);
            }
        }
        if (seeds_here) {
            for (int si = T.seed_off[ph]; si < T.seed_off[ph + 1]; ++si) {
                const double* ys = T.seedY + (long)si * T.Dm;
                // seeds already far above the tangent plane are not worth a local minimization
                for (int k2 = 0; k2 < CG_NS; ++k2) W.inp[k2] = T.sv[k2];
                cg_evaluate(P, W, ys, 0);
                double tt = W.G_fu / W.Mtot / T.RT;
                for (int i = 0; i < C; ++i) tt -= mu[i] * W.mval[i] / W.Mtot;
                if (!(tt < T.seed_prune)) continue;
                cg_push_job(T, t, ph, ys, P.D, 1);
            }
        }
        T.cp_end[cp] = T.nj;
    }
    T.cp_last[t] = T.ncp;
}

// condition t's job results, as Solver._pricing_round treats its minimizations (TileSolver._consume): points below
// the tangent plane (and kept seeds) stored, certification of a polished state, skip state. Returns whether
// anything new was found
CG_DEVICE static int cg_consume(CgTile& T, int t, int polished, int* no_polish) {
    const int C = T.C;
    const double* mu = T.mu + (long)t * C;
    int added = 0, seed_added = 0;
    T.cert_kind[t] = 0;
    double ybest[CG_DMAX];
    for (int cp = T.cp_first[t]; cp < T.cp_last[t]; ++cp) {
        int ph = T.cp_phase[cp], D = T.di[ph * CGI_COUNT + CGI_D];
        double best_tpd = INFINITY;
        int all_conv = 1;
        for (int j = T.cp_start[cp]; j < T.cp_end[cp]; ++j) {
            if (!T.j_conv[j]) all_conv = 0;
            double G = T.j_G[j], tpd = T.j_tpd[j];
            if (!CG_ISFINITE(G)) tpd = INFINITY;
            T.priced[t]++;
            if (T.j_keep[j] || tpd < -T.tol)
                if (cg_add_point(T, t, ph, T.j_y + (long)j * T.Dm, T.j_x + (long)j * C, G) >= 0 && tpd < -T.tol)
                    seed_added = 1;
            if (tpd < best_tpd) {
                best_tpd = tpd;
                for (int i = 0; i < D; ++i) ybest[i] = T.j_y[(long)j * T.Dm + i];
            }
        }
        if (polished && best_tpd < -T.tol) {
            if (T.cert_kind[t] == 0 || best_tpd < T.cert_tpd[t]) {
                for (int i = 0; i < D; ++i) T.ycert[(long)t * T.Dm + i] = ybest[i];
                T.cert_kind[t] = 1; T.cert_phase[t] = ph; T.cert_tpd[t] = best_tpd;
            }
            added = 1; *no_polish = 1;
        }
        long sk = (long)t * T.nph + ph;
        if (all_conv) {
            T.has_last[sk] = 1; T.last_tpd[sk] = best_tpd;
            for (int i = 0; i < C; ++i) T.last_mu[sk * C + i] = mu[i];
        } else {
            T.has_last[sk] = 0;
        }
    }
    if (polished && !added) {
        int n = cg_npts(T, t);
        for (int p = 0; p < n; ++p) {
            double tt = cg_pg(T, t, p) / T.RT;
            const double* x = cg_px(T, t, p);
            for (int i = 0; i < C; ++i) tt -= mu[i] * x[i];
            if (tt < -T.tol) {
                int ph = cg_pph(T, t, p), D = T.di[ph * CGI_COUNT + CGI_D];
                T.cert_kind[t] = 2; T.cert_phase[t] = ph; T.cert_tpd[t] = tt;
                const double* y = cg_py(T, t, p);
                for (int i = 0; i < D; ++i) T.ycert[(long)t * T.Dm + i] = y[i];
                added = 1; *no_polish = 1;
                break;
            }
        }
    }
    if (seed_added && !added) T.cert_kind[t] = 3;
    return added || seed_added;
}

// ---- the polish of a condition (cg_round.h) on its own state; the polished points into its slab

// transient work of one condition's polish (per thread on a GPU)
struct CgPolishWork {
    double S[CG_CMAX * CG_CMAX * CG_CMAX], J[4 * CG_CMAX * CG_CMAX], F[2 * CG_CMAX], t[CG_CMAX],
        J2[4 * CG_CMAX * CG_CMAX], F2[2 * CG_CMAX], sv[2 * CG_CMAX], U[4 * CG_CMAX * CG_CMAX],
        V[4 * CG_CMAX * CG_CMAX], Bm[CG_CMAX * CG_DMAX], Wm[CG_CMAX * CG_DMAX];
    int ipv[2 * CG_CMAX];
};

CG_DEVICE static void cg_polish_setup(CgTile& T, int t, CgPolish& S, CgPolishWork& PW) {
    const int C = T.C;
    S.C = C; S.K = T.K; S.Dm = T.Dm;
    S.R = T.R + (long)t * T.K * C; S.b = T.b + (long)t * T.K; S.mf = T.mf + (long)t * C; S.fixed = T.fixed + (long)t * C;
    S.gauge = T.gauge; S.ngauge = T.ngauge;
    S.pol_y = T.pol_y + (long)t * C * T.Dm; S.pol_x = T.pol_x + (long)t * C * C; S.pol_n = T.pol_n + (long)t * C;
    S.pol_mu = T.pol_mu + (long)t * C; S.pol_G = T.pol_G + (long)t * C; S.pol_ph = T.pol_ph + (long)t * C;
    S.pol_S = PW.S; S.pol_J = PW.J; S.pol_F = PW.F; S.pol_t = PW.t; S.pol_J2 = PW.J2; S.pol_F2 = PW.F2;
    S.pol_sv = PW.sv; S.pol_U = PW.U; S.pol_V = PW.V; S.pol_B = PW.Bm; S.pol_W = PW.Wm; S.pol_ip = PW.ipv;
    S.pol_na = T.pol_na[t]; S.polish_fail = 0;
    S.di = T.di; S.dd = T.dd; S.ip = T.ip; S.dp = T.dp; S.sv = T.sv; S.RT = T.RT;
}

// after a converged polish: its points stored, its potentials the condition's (end of Solver._polish_core)
CG_DEVICE static void cg_polish_done(CgTile& T, int t, CgPolish& S) {
    T.pol_na[t] = S.pol_na;
    for (int a = 0; a < S.pol_na; ++a)
        cg_add_point(T, t, S.pol_ph[a], S.pol_y + a * T.Dm, S.pol_x + a * T.C, S.pol_G[a]);
    for (int i = 0; i < T.C; ++i) T.mu[(long)t * T.C + i] = S.pol_mu[i];
}

template <class WT>
CG_DEVICE static int cg_tile_polish(CgTile& T, int t, WT& W, CgPolishWork& PW) {
    CgPolish S; cg_polish_setup(T, t, S, PW);
    CgStoreView st{T.Py, T.Px, T.PG, T.Pph, T.Dm};
    int ok = cg_polish(S, st, T.basis + (long)t * T.K, T.xB + (long)t * T.K, T.mu + (long)t * T.C, W);
    if (ok) cg_polish_done(T, t, S);
    return ok;
}

template <class WT>
CG_DEVICE static int cg_tile_polish_augment(CgTile& T, int t, WT& W, CgPolishWork& PW) {
    CgPolish S; cg_polish_setup(T, t, S, PW);
    int ok = cg_polish_augment(S, T.cert_phase[t], T.ycert + (long)t * T.Dm, W);
    if (ok) cg_polish_done(T, t, S);
    return ok;
}

// ---- results

// condition t's result (TileSolver._record): the polished points, else the LP solution's
CG_DEVICE static void cg_record(CgTile& T, int t, int st, int polished) {
    const int C = T.C, Dm = T.Dm;
    int na = 0;
    if (st == 0 && polished) {
        for (int a = 0; a < T.pol_na[t]; ++a) {
            int ph = T.pol_ph[(long)t * C + a], D = T.di[ph * CGI_COUNT + CGI_D];
            T.phs[(long)t * C + na] = ph; T.amt[(long)t * C + na] = T.pol_n[(long)t * C + a];
            T.G[(long)t * C + na] = T.pol_G[(long)t * C + a];
            for (int i = 0; i < C; ++i) T.X[((long)t * C + na) * C + i] = T.pol_x[((long)t * C + a) * C + i];
            for (int j = 0; j < D; ++j) T.Y[((long)t * C + na) * Dm + j] = T.pol_y[((long)t * C + a) * Dm + j];
            na++;
        }
    } else if (st == 0) {
        for (int k = 0; k < T.K; ++k) {
            double x = T.xB[(long)t * T.K + k];
            if (!(x > 1e-300)) continue;
            int p = T.basis[(long)t * T.K + k];
            if (p < 0) { if (x > 1e-9) { st = 5; break; } continue; }
            int ph = cg_pph(T, t, p), D = T.di[ph * CGI_COUNT + CGI_D];
            T.phs[(long)t * C + na] = ph; T.amt[(long)t * C + na] = x; T.G[(long)t * C + na] = cg_pg(T, t, p);
            const double* px = cg_px(T, t, p); const double* py = cg_py(T, t, p);
            for (int i = 0; i < C; ++i) T.X[((long)t * C + na) * C + i] = px[i];
            for (int j = 0; j < D; ++j) T.Y[((long)t * C + na) * Dm + j] = py[j];
            na++;
        }
    }
    T.status[t] = st; T.nout[t] = na;
    for (int i = 0; i < C; ++i) T.mu_out[(long)t * C + i] = T.mu[(long)t * C + i] * T.RT;
    T.done[t] = 1;
}

// ---- the steps of a round (TileSolver.solve_tile_batched), one condition each

// condition t's LP over the store from the basis, basic amounts and potentials at basis, xB, mu (its own in T, or
// a lane's copies), its iterations counted in *iters; the scans by the lanes G (cg_lp_solve). Its status
CG_DEVICE static int cg_tile_lp(const CgTile& T, int t, int* basis, double* xB, double* mu, long long* iters,
                                const CgLanes& G) {
    const int C = T.C, K = T.K;
    double obj = 0.0, dual[CG_CMAX], cB[CG_CMAX], LU[CG_CMAX * CG_CMAX], tmp[CG_CMAX];
    int ipiv[CG_CMAX];
    CgLP L{K, C, T.nshared, T.Px, T.PG, T.R + (long)t * K * C, T.b + (long)t * K, T.mf + (long)t * C,
           basis, xB, mu, dual, cB, LU, tmp, ipiv, T.RT, iters};
    return cg_lp_solve(L, &obj, G);
}

// cg_step_lp after the LP (status st): gauge fix, polish
template <class WT>
CG_DEVICE static int cg_step_lp_polish(CgTile& T, int t, int st, WT& W, CgPolishWork& PW) {
    const int C = T.C;
    if (st != 0) { cg_record(T, t, st, 0); T.kind[t] = CG_IDLE; return 0; }
    cg_gauge_fix(T.mu + (long)t * C, T.gauge, T.ngauge, C);
    T.skipped[t] = T.nopol[t];
    int polished = 0;
    if (T.use_polish && !T.nopol[t]) polished = cg_tile_polish(T, t, W, PW);
    T.nopol[t] = 0;
    T.polished[t] = polished; T.added[t] = 0; T.aug[t] = 0; T.kind[t] = CG_MAIN;
    return 1;
}

// LP, gauge fix, polish (unless the last certification failed); whether pricing jobs follow (cg_gen_jobs)
template <class WT>
CG_DEVICE static int cg_step_lp(CgTile& T, int t, WT& W, CgPolishWork& PW) {
    double best; int arg;
    CgLanes G{0, 1, 1, &best, &arg};
    int st = cg_tile_lp(T, t, T.basis + (long)t * T.K, T.xB + (long)t * T.K, T.mu + (long)t * T.C, T.lp_iters + t, G);
    return cg_step_lp_polish(T, t, st, W, PW);
}

// the start of a round: LP, gauge fix, polish, the pricing jobs
template <class WT>
CG_DEVICE static void cg_step_start(CgTile& T, int t, int rounds, int use_seeds, WT& W, CgPolishWork& PW) {
    if (cg_step_lp(T, t, W, PW)) cg_gen_jobs(T, t, rounds, use_seeds, W);
}

// the job results; what comes next
CG_DEVICE static void cg_step_consume(CgTile& T, int t, int rounds, int max_rounds) {
    int kind = T.kind[t];
    if (kind != CG_MAIN && kind != CG_AUG && kind != CG_FINAL) return;
    int polished = T.polished[t], no_polish = T.nopol[t];
    int added = cg_consume(T, t, polished, &no_polish);
    T.nopol[t] = no_polish;
    if (kind == CG_FINAL) {
        if (added) polished = 0;
        kind = CG_IDLE;
    } else if (added && polished && (T.cert_kind[t] == 1 || T.cert_kind[t] == 2) && T.aug[t] < T.C) {
        kind = CG_NEED_AUG;
    } else if (!added && !polished && T.skipped[t] && T.use_polish) {
        kind = CG_NEED_FINAL;
    } else {
        kind = CG_IDLE;
    }
    T.polished[t] = polished; T.added[t] = added; T.kind[t] = kind;
    if (kind == CG_IDLE && (!added || rounds == max_rounds)) cg_record(T, t, added ? 6 : 0, polished);
}

// another Newton pass with the minimum found below a polished tangent plane, or the polish of a round that
// skipped it; whether pricing jobs follow (cg_gen_jobs, without seeds)
template <class WT>
CG_DEVICE static int cg_step_repolish(CgTile& T, int t, int rounds, int max_rounds, WT& W, CgPolishWork& PW) {
    int kind = T.kind[t], jobs = 0;
    if (kind != CG_NEED_AUG && kind != CG_NEED_FINAL) return 0;
    int added = T.added[t];
    if (kind == CG_NEED_AUG) {
        T.aug[t]++;
        if (cg_tile_polish_augment(T, t, W, PW)) {
            T.nopol[t] = 0;
            jobs = 1; kind = CG_AUG;
        } else {
            T.polished[t] = 0;
            kind = CG_IDLE;
        }
    } else {
        if (cg_tile_polish(T, t, W, PW)) {
            T.polished[t] = 1;
            jobs = 1; kind = CG_FINAL;
        } else {
            kind = CG_IDLE;
            if (!added || rounds == max_rounds) { cg_record(T, t, added ? 6 : 0, T.polished[t]); }
        }
    }
    T.kind[t] = kind;
    if (kind == CG_IDLE && added && rounds == max_rounds && !T.done[t]) cg_record(T, t, 6, T.polished[t]);
    return jobs;
}

// the next step of a sub-round and its pricing jobs
template <class WT>
CG_DEVICE static void cg_step_next(CgTile& T, int t, int rounds, int max_rounds, WT& W, CgPolishWork& PW) {
    if (cg_step_repolish(T, t, rounds, max_rounds, W, PW)) cg_gen_jobs(T, t, rounds, 0, W);
}

// ---- the end of a round: the conditions' own points into the store, condition by condition (one thread)

CG_DEVICE static int cg_find_shared(const CgTile& T, int ph, const double* y, int D) {
    for (int p = T.head[ph]; p >= 0; p = T.Pnext[p]) {
        double m = 0.0;
        for (int j = 0; j < D; ++j) { double d = T.Py[(long)p * T.Dm + j] - y[j]; d = d < 0 ? -d : d; if (d > m) m = d; }
        if (m < 1e-7) return p;
    }
    return -1;
}

CG_DEVICE static int cg_tile_merge(CgTile& T, const int* members, int nmem) {
    int n0 = T.nshared;
    for (int m = 0; m < nmem; ++m) {
        int t = members[m];
        for (int k = 0; k < T.nown[t]; ++k) {
            long o = T.own_off[t] + k;
            int ph = T.Oph[o], D = T.di[ph * CGI_COUNT + CGI_D];
            if (cg_find_shared(T, ph, T.Oy + o * T.Dm, D) >= 0) continue;
            if (T.nshared >= T.cap) return -1;   // (the caller sized the store)
            int p = T.nshared++;
            for (int j = 0; j < T.Dm; ++j) T.Py[(long)p * T.Dm + j] = T.Oy[o * T.Dm + j];
            for (int i = 0; i < T.C; ++i) T.Px[(long)p * T.C + i] = T.Ox[o * T.C + i];
            T.PG[p] = T.OG[o]; T.Pph[p] = ph; T.Pcode[p] = T.Ocode[o];
            T.Pnext[p] = T.head[ph]; T.head[ph] = p;
        }
        T.nown[t] = 0;
    }
    return T.nshared - n0;
}
