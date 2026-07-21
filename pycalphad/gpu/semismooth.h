#ifndef SEMISMOOTH_H
#define SEMISMOOTH_H
// Semismooth-Newton straggler solver (gpu-fast pass 2, PYCGPU_SS=1).
//
// Solves the equilibrium KKT system with Fischer-Burmeister
// complementarity over a fixed candidate compset list, in fully
// nondimensional variables:
//   z = [ lam/RT (ncomp) | a_p (P) | u_p = ln(y_p) (pd_p each)
//         | nu_p/RT (nic_p each) ]
// Rows (all O(1)):
//   balance:      sum_p a_p fm_p(y) - b                    (ncomp)
//   stationarity: (dGfu - M^T lam - E^T nu) / RT           (pd_p)
//   internal:     g_int(y)  (sublattice sums - 1)          (nic_p)
//   FB:           a + s - sqrt(a^2 + s^2 + 2 mu^2),
//                 s = (Gfu - lam.fm) / (RT * atoms_fu)     (1)
// Validated against the Python oracle (scratchpad ss_study.py):
// AlCuFe 72% / alcocrni 48% production-acceptance, median ~50 iters.
// Failure => leave the condition's cap flag set; the faithful pass-3
// rerun handles it (this header must never be the last line of defense).
//
// v1 STATUS (local GeForce validation, 2026-07-20): ternary AlCuFe
// CLEAN (0 diffs >5 J vs faithful over 392 conditions, suite 11/11,
// 16-30/33 stragglers accepted). Quaternary alcocrni NOT yet
// production-quality: ~18/73 accepted stragglers land in wrong basins
// up to ~1e3 J above the reference while passing every gate (balance,
// full-grid df, hull-energy bound) — the grid cannot see those basins.
// PYCGPU_SS therefore stays opt-in experimental; v2 needs stronger
// acceptance (relaxed-candidate df probes) and quat-tuned candidates.
// Perf note: the dense per-thread LM is FP64-heavy; consumer cards run
// it slowly (~1.5 s/straggler) — MI-class cards are the target.

// Candidate cap: real candidate sets are hull vertices + a few near-hull
// phases; capping well below MAX_PHASES keeps the per-thread J at
// SS_NVAR^2 manageable (~250 KB vs ~1.8 MB at MAX_PHASES=22).
#define SS_MAX_CANDS   (8)

// Defined later in the assembled module source (generated globals block).
extern __device__ PhaseRecord g_phase_records_array[];
#define SS_NVAR (MAX_COMPONENTS + SS_MAX_CANDS * (1 + MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS))
#define SS_U_LO   (-230.0)
#define SS_U_HI   (0.02)
#define SS_STEP_CLAMP (2.0)

// Per-thread workspace, carved out of a caller-provided flat buffer.
typedef struct SSCand {
    const PhaseRecord* pr;
    int pd;       // phase dof (site fractions)
    int nic;      // internal constraints (sublattice sums)
    int iy;       // z offset of u block
    int inu;      // z offset of nu block
    bool keep;    // active-set membership during polish
} SSCand;

typedef struct SSSys {
    int ncomp;
    int nsv;      // spec num_statevars
    int P;        // candidate count
    int nvar;
    double RT;
    double b[MAX_COMPONENTS];
    double statevars[MAX_STATEVARS];
    SSCand cand[SS_MAX_CANDS];
} SSSys;

// ---- residual + Jacobian assembly -------------------------------------
// mode_active: FB row replaced by s = 0 for kept cands; dropped cands are
// excluded from all rows (their z entries are frozen).
// Returns squared norm of F. J may be null.
__device__ static double ss_residual(
    const SSSys* S, const double* z, double* F, double* J,
    double smooth_mu, bool mode_active,
    // scratch (sized by caller): dof, eg, hess, fm, mjac, icv, icjac
    double* dof, double* eg, double* hess, double* fm, double* mjac,
    double* icv, double* icjac)
{
    const int n = S->nvar;
    const int ncomp = S->ncomp;
    const double RT = S->RT;
    for (int i = 0; i < n; ++i) F[i] = 0.0;
    if (J) for (long long i = 0; i < (long long)n * n; ++i) J[i] = 0.0;
    for (int c = 0; c < ncomp; ++c) F[c] = -S->b[c];

    int row = ncomp;
    for (int k = 0; k < S->P; ++k) {
        const SSCand* ck = &S->cand[k];
        const int pd = ck->pd, nic = ck->nic;
        const int row_stat = row, row_ic = row + pd, row_fb = row + pd + nic;
        row = row_fb + 1;
        if (mode_active && !ck->keep) {
            // frozen: pin the excluded block's variables with identity rows
            if (J) {
                J[(long long)row_stat * 0] = J[(long long)row_stat * 0]; // no-op
                for (int j = 0; j < pd; ++j)
                    J[(long long)(row_stat + j) * n + (ck->iy + j)] = 1.0;
                for (int j = 0; j < nic; ++j)
                    J[(long long)(row_ic + j) * n + (ck->inu + j)] = 1.0;
                J[(long long)row_fb * n + (ncomp + k)] = 1.0;
            }
            continue;
        }
        // dof = [statevars, y]
        for (int sv = 0; sv < S->nsv; ++sv) dof[sv] = S->statevars[sv];
        for (int j = 0; j < pd; ++j) {
            double u = z[ck->iy + j];
            if (u < SS_U_LO) u = SS_U_LO;
            if (u > SS_U_HI) u = SS_U_HI;
            dof[S->nsv + j] = exp(u);
        }
        const double* y = &dof[S->nsv];
        const double a = z[ncomp + k];

        // energy / grad / hess per formula unit ([G, dT, dy...] layout)
        if (ck->pr->formulafused) {
            ck->pr->formulafused(eg, hess, dof);
        } else {
            eg[0] = ck->pr->formulaobj(dof);
            ck->pr->formulagrad(&eg[1], dof);   // wait: formulagrad writes [dT,dy...] into arg
            ck->pr->formulahess(hess, dof);
        }
        const double Gfu = eg[0];
        // fm_c and M = d fm / d(dof) (rows ncomp x cols nsv+pd)
        ck->pr->formulamole_obj(fm, dof);
        if (J) ck->pr->formulamole_grad(mjac, dof);
        // internal constraints g(y) and rows x (nsv+pd) jacobian
        ck->pr->internal_cons_func(icv, dof);
        if (J) ck->pr->internal_cons_jac(icjac, dof);

        double lam_fm = 0.0, atoms_fu = 0.0;
        for (int c = 0; c < ncomp; ++c) {
            lam_fm += (z[c] * RT) * fm[c];
            atoms_fu += fm[c];
            F[c] += a * fm[c];
        }
        if (atoms_fu < 1e-12) atoms_fu = 1e-12;

        // stationarity rows: (dG/dy_j - (M^T lam)_j - (E^T nu)_j)/RT
        for (int j = 0; j < pd; ++j)
            F[row_stat + j] = eg[2 + j] / RT;   // dG/dy_j / RT
        // (M^T lam)/RT and (E^T nu)/RT contributions need the jacobians even
        // in value-only calls: mjac/icjac are cheap generated functions.
        if (!J) { ck->pr->formulamole_grad(mjac, dof); ck->pr->internal_cons_jac(icjac, dof); }
        const int mcols = 1 + pd;  // grad-codegen layout: [dT, dy...]
        for (int j = 0; j < pd; ++j) {
            double ml = 0.0;
            for (int c = 0; c < ncomp; ++c)
                ml += z[c] * mjac[c * mcols + 1 + j];   // lam/RT * dfm/dy: RT cancels
            double en = 0.0;
            for (int s = 0; s < nic; ++s)
                en += z[ck->inu + s] * icjac[s * mcols + 1 + j];
            F[row_stat + j] -= (ml + en);
        }
        // internal rows
        for (int s = 0; s < nic; ++s) F[row_ic + s] = icv[s];
        // FB / s row
        const double s_p = (Gfu - lam_fm) / (RT * atoms_fu);
        double r_ = sqrt(a * a + s_p * s_p + 2.0 * smooth_mu * smooth_mu);
        if (r_ < 1e-300) r_ = 1e-300;
        F[row_fb] = mode_active ? s_p : (a + s_p - r_);

        if (J) {
            const double da = mode_active ? 0.0 : (1.0 - a / r_);
            const double ds = mode_active ? 1.0 : (1.0 - s_p / r_);
            // balance rows
            for (int c = 0; c < ncomp; ++c) {
                J[(long long)c * n + (ncomp + k)] = fm[c];
                for (int j = 0; j < pd; ++j)
                    J[(long long)c * n + (ck->iy + j)] +=
                        a * mjac[c * mcols + 1 + j] * y[j];
            }
            // stationarity block: d/du via y-scaled Hessian y-block, minus
            // lam/nu columns
            for (int i2 = 0; i2 < pd; ++i2) {
                for (int j = 0; j < pd; ++j)
                    J[(long long)(row_stat + i2) * n + (ck->iy + j)] =
                        hess[(1 + i2) * (1 + pd) + (1 + j)] * y[j] / RT;
                for (int c = 0; c < ncomp; ++c)
                    J[(long long)(row_stat + i2) * n + c] =
                        -mjac[c * mcols + 1 + i2];
                for (int s = 0; s < nic; ++s)
                    J[(long long)(row_stat + i2) * n + (ck->inu + s)] =
                        -icjac[s * mcols + 1 + i2];
            }
            // internal rows: d g_s/du_j = icjac * y
            for (int s = 0; s < nic; ++s)
                for (int j = 0; j < pd; ++j)
                    J[(long long)(row_ic + s) * n + (ck->iy + j)] =
                        icjac[s * mcols + 1 + j] * y[j];
            // FB row
            J[(long long)row_fb * n + (ncomp + k)] = mode_active ? 0.0 : da;
            for (int c = 0; c < ncomp; ++c)
                J[(long long)row_fb * n + c] = ds * (-fm[c]) / atoms_fu;
            for (int j = 0; j < pd; ++j) {
                // ds/du_j = ((dG/dy - M lam)/RT - s * datoms/dy)/atoms * y
                double ml = 0.0;
                for (int c = 0; c < ncomp; ++c)
                    ml += z[c] * mjac[c * mcols + 1 + j];
                double datoms = 0.0;
                for (int c = 0; c < ncomp; ++c)
                    datoms += mjac[c * mcols + 1 + j];
                double dsdy = (eg[2 + j] / RT - ml - s_p * datoms) / atoms_fu;
                J[(long long)row_fb * n + (ck->iy + j)] = ds * dsdy * y[j];
            }
        }
    }
    double nf2 = 0.0;
    for (int i = 0; i < n; ++i) nf2 += F[i] * F[i];
    return nf2;
}


// ---- linear algebra: (J^T J + lam I) dz = -J^T F via LU -----------------
__device__ static bool ss_lm_solve(const double* J, const double* F,
                                   double lam_lm, int n,
                                   double* A, double* dz, int* ipiv)
{
    // A = J^T J + lam I (symmetric; row-major == col-major)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double acc = 0.0;
            for (int r = 0; r < n; ++r)
                acc += J[(long long)r * n + i] * J[(long long)r * n + j];
            A[(long long)i * n + j] = acc;
            A[(long long)j * n + i] = acc;
        }
        A[(long long)i * n + i] += lam_lm;
    }
    for (int i = 0; i < n; ++i) {
        double acc = 0.0;
        for (int r = 0; r < n; ++r)
            acc -= J[(long long)r * n + i] * F[r];
        dz[i] = acc;
    }
    int info = 0;
    pyclap_dgetrf(n, n, A, n, ipiv, &info);
    if (info != 0) return false;
    // apply pivots then unit-lower / upper triangular solves (col-major LU,
    // symmetric input so storage order is immaterial before factorization)
    for (int i = 0; i < n; ++i) {
        int p = ipiv[i] - 1;
        if (p != i) { double t = dz[i]; dz[i] = dz[p]; dz[p] = t; }
    }
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < i; ++j)
            dz[i] -= A[(long long)j * n + i] * dz[j];   // L (unit diag), col-major
    for (int i = n - 1; i >= 0; --i) {
        for (int j = i + 1; j < n; ++j)
            dz[i] -= A[(long long)j * n + i] * dz[j];   // U, col-major
        double d = A[(long long)i * n + i];
        if (fabs(d) < 1e-300) return false;
        dz[i] /= d;
    }
    for (int i = 0; i < n; ++i)
        if (!(dz[i] == dz[i])) return false;            // NaN guard
    return true;
}

// ---- LM stage runner ----------------------------------------------------
// Runs the smoothing ladder (or one sharp stage in active mode) on z.
// Returns final ||F||^2. scratch layout provided by caller.
typedef struct SSScratch {
    double *J, *A, *F, *Ft, *dz, *zt;
    double *dof, *eg, *hess, *fm, *mjac, *icv, *icjac;
    int* ipiv;
} SSScratch;

__device__ static double ss_run(const SSSys* S, double* z, bool mode_active,
                                int budget, double tol2, SSScratch* W)
{
    const int n = S->nvar;
    const double stages_mu[4] = {1e-2, 1e-4, 1e-7, 0.0};
    const int stages_it[4] = {40, 40, 40, 80};
    const int first = mode_active ? 3 : 0;
    double lam_lm = 1e-6;
    double recent[5];
    int nrecent = 0;
    int total = 0;
    double nf2 = 1e300;
    for (int st = first; st < 4; ++st) {
        const double smu = mode_active ? 0.0 : stages_mu[st];
        int stall = 0;
        for (int it = 0; it < stages_it[st]; ++it) {
            nf2 = ss_residual(S, z, W->F, W->J, smu, mode_active,
                              W->dof, W->eg, W->hess, W->fm, W->mjac,
                              W->icv, W->icjac);
            if (nf2 < tol2 || ++total >= budget) break;
            bool accepted = false;
            for (int t = 0; t < 5; ++t) {
                if (ss_lm_solve(W->J, W->F, lam_lm, n, W->A, W->dz, W->ipiv)) {
                    for (int i = 0; i < n; ++i) {
                        double d = W->dz[i];
                        if (d > SS_STEP_CLAMP) d = SS_STEP_CLAMP;
                        if (d < -SS_STEP_CLAMP) d = -SS_STEP_CLAMP;
                        W->zt[i] = z[i] + d;
                    }
                    for (int k = 0; k < S->P; ++k)
                        for (int j = 0; j < S->cand[k].pd; ++j) {
                            double* u = &W->zt[S->cand[k].iy + j];
                            if (*u < SS_U_LO) *u = SS_U_LO;
                            if (*u > SS_U_HI) *u = SS_U_HI;
                        }
                    double nft2 = ss_residual(S, W->zt, W->Ft, nullptr, smu,
                                              mode_active, W->dof, W->eg,
                                              W->hess, W->fm, W->mjac,
                                              W->icv, W->icjac);
                    double ref = nf2;
                    for (int r = 0; r < nrecent; ++r)
                        if (recent[r] > ref) ref = recent[r];
                    if (nft2 < (1.0 - 1e-6) * ref) {
                        for (int i = 0; i < n; ++i) z[i] = W->zt[i];
                        lam_lm = lam_lm * 0.33;
                        if (lam_lm < 1e-10) lam_lm = 1e-10;
                        accepted = true;
                        break;
                    }
                }
                lam_lm = lam_lm * 8.0;
                if (lam_lm > 1e6) lam_lm = 1e6;
            }
            recent[nrecent % 5] = nf2;
            if (nrecent < 5) ++nrecent;
            if (!accepted && ++stall >= 3) break;
            if (accepted) stall = 0;
        }
        if (nf2 < tol2) break;
    }
    return nf2;
}

// ---- polish: active-set solve + drop-to-solvable + improvement drops ----
// On success writes the final state into z (all-candidate layout; dropped
// candidates have a=0) and returns true with *gm_out set (per mole atom).
__device__ static bool ss_polish(SSSys* S, double* z, SSScratch* W,
                                 double* z_best, double* gm_out,
                                 int dbg, int tid)
{
    const int ncomp = S->ncomp;
    // initial keep: raw a above dust
    int nkeep = 0;
    for (int k = 0; k < S->P; ++k) {
        S->cand[k].keep = (z[ncomp + k] > 1e-4);
        if (S->cand[k].keep) ++nkeep;
    }
    if (nkeep == 0) {
        int kmax = 0;
        for (int k = 1; k < S->P; ++k)
            if (z[ncomp + k] > z[ncomp + kmax]) kmax = k;
        S->cand[kmax].keep = true;
        nkeep = 1;
    }
    // drop-to-solvable
    bool ok = false;
    for (int attempt = 0; attempt < 4; ++attempt) {
        for (int i = 0; i < S->nvar; ++i) z_best[i] = z[i];
        for (int k = 0; k < S->P; ++k)
            if (S->cand[k].keep && z_best[ncomp + k] < 1e-6)
                z_best[ncomp + k] = 1e-6;
        double nf2 = ss_run(S, z_best, true, 40, 1e-22, W);
        bool feas = (nf2 < 1e-16);
        for (int k = 0; k < S->P; ++k)
            if (S->cand[k].keep && z_best[ncomp + k] < -1e-8) feas = false;
        if (dbg && tid < 2) {
            printf("[SS] tid %d polish attempt %d nkeep=%d nf2=%.3e amounts:", tid, attempt, nkeep, nf2);
            for (int k = 0; k < S->P; ++k)
                if (S->cand[k].keep) printf(" %.3e", z_best[ncomp + k]);
            printf("\n");
        }
        if (feas) { ok = true; break; }
        if (nkeep <= 1) break;
        int kdrop = -1; double amin = 1e300;
        for (int k = 0; k < S->P; ++k)
            if (S->cand[k].keep && z_best[ncomp + k] < amin) {
                amin = z_best[ncomp + k]; kdrop = k;
            }
        S->cand[kdrop].keep = false;
        --nkeep;
    }
    if (!ok) return false;
    // GM of the accepted state (per mole atom); improvement drops omitted in
    // v1 kernel (they only trim tie-degenerate extras; acceptance gates do
    // not depend on them)
    double gtot = 0.0, atoms = 0.0;
    for (int k = 0; k < S->P; ++k) {
        if (!S->cand[k].keep) { z_best[ncomp + k] = 0.0; continue; }
        const SSCand* ck = &S->cand[k];
        for (int sv = 0; sv < S->nsv; ++sv) W->dof[sv] = S->statevars[sv];
        for (int j = 0; j < ck->pd; ++j)
            W->dof[S->nsv + j] = exp(z_best[ck->iy + j]);
        double Gfu = ck->pr->formulaobj(W->dof);
        ck->pr->formulamole_obj(W->fm, W->dof);
        double at = 0.0;
        for (int c = 0; c < ncomp; ++c) at += W->fm[c];
        double a = z_best[ncomp + k];
        gtot += a * Gfu;
        atoms += a * at;
    }
    if (atoms < 1e-12) return false;
    *gm_out = gtot / atoms;
    return true;
}


// ---- grid blob decode ---------------------------------------------------
// The host grid blob is NOT a DeviceGrid struct: [8 i4 header][Y data]
// [X data][GM data][PhaseID data][phase_grid_indices_start/stop][n_map].
// Mirror eqsolver's reconstruction (incl. external-pointer mode).
__device__ static bool ss_grid_decode(const void* raw, int num_unique,
                                      DeviceGrid* g)
{
    if (raw == nullptr) return false;
    const char* b = (const char*)raw;
    const int* hdr = (const int*)b;
    g->num_grid_points_total = hdr[0];
    g->phase_dof_stride_Y = hdr[1];
    g->num_components_stride_X = hdr[2];
    const int ysz = hdr[3], xsz = hdr[4], gsz = hdr[5], psz = hdr[6];
    const int ext = hdr[7];
    size_t yo = 8 * sizeof(int);
    size_t xo = yo + (size_t)ysz * sizeof(double);
    size_t go = xo + (size_t)xsz * sizeof(double);
    size_t po = go + (size_t)gsz * sizeof(double);
    size_t io = po + (size_t)psz * sizeof(int);
    if (ext == 1) {
        g->Y_ptr = *(const double* const*)(b + yo);
        g->X_ptr = *(const double* const*)(b + xo);
        g->GM_ptr = *(const double* const*)(b + go);
        g->PhaseID_ptr = *(const int* const*)(b + po);
    } else {
        g->Y_ptr = (const double*)(b + yo);
        g->X_ptr = (const double*)(b + xo);
        g->GM_ptr = (const double*)(b + go);
        g->PhaseID_ptr = (const int*)(b + po);
    }
    g->phase_grid_indices_start = (const int*)(b + io);
    g->phase_grid_indices_stop = (const int*)(b + io) + num_unique;
    g->num_mappable_phases_in_grid = num_unique;
    return g->num_grid_points_total > 0;
}

// ---- setup, acceptance, kernel entry ------------------------------------
// Flat-results row offsets (must mirror gpu_equilibrium result layout):
//   [0] GM | [1..MC] MU | [1+MC..+MP] NP | conv | num_stable | T | P |
//   cap-flag/status | Y (MP*MD) | X (MP*MC) | phase_ids (MP)
// Host passes ncomp/nsv/b explicitly; the spec blob is never parsed here.

__device__ static bool ss_setup(
    SSSys* S, double* z, const DevicePhaseData* phase_data,
    const DeviceGrid* grid, const double* ipd_row, const double* cond_row,
    const double* b_row, int ncomp, int nsv, int* rec_idx_out)
{
    S->ncomp = ncomp; S->nsv = nsv;
    for (int c = 0; c < ncomp; ++c) {
        if (!(b_row[c] == b_row[c])) return false;   // NaN = not eligible
        S->b[c] = b_row[c];
    }
    for (int sv = 0; sv < nsv && sv < MAX_STATEVARS; ++sv)
        S->statevars[sv] = cond_row[sv];
    const double T = S->statevars[nsv - 1];          // [N,P,T] convention
    if (!(T > 0.0)) return false;
    S->RT = 8.31446261815324 * T;

    // ipd layout offsets (all-double InitialPhaseData)
    const int off_amt = MAX_PHASES;
    const int off_sf = 2 * MAX_PHASES;
    const int off_mu = 2 * MAX_PHASES + MAX_PHASES * MAX_DOF_PER_PHASE
                       + MAX_PHASES * MAX_COMPONENTS;
    const int nipd = (int)ipd_row[off_mu + MAX_COMPONENTS];
    double lam0[MAX_COMPONENTS];
    for (int c = 0; c < ncomp; ++c) lam0[c] = ipd_row[off_mu + c];

    S->P = 0;
    int var = ncomp;   // a-block placed after (assigned below)
    // hull/ipd compsets
    for (int i = 0; i < nipd && i < MAX_PHASES; ++i) {
        int pidx = (int)ipd_row[i];
        double amt = ipd_row[off_amt + i];
        if (pidx < 0 || pidx >= phase_data->num_unique_phase_records) continue;
        if (S->P >= SS_MAX_CANDS) break;
        SSCand* ck = &S->cand[S->P];
        ck->pr = &phase_data->phase_records_array[pidx];
        if (!ck->pr->formulamole_obj || !ck->pr->internal_cons_func) continue;
        ck->pd = ck->pr->phase_dof;
        ck->nic = ck->pr->num_internal_cons;
        ck->keep = true;
        rec_idx_out[S->P] = pidx;
        z[ncomp + S->P] = (amt > 1e-8) ? amt : 1e-6;
        // u seeded below once offsets are known; stash sf source index via iy
        ck->iy = i;      // TEMP: ipd slot; fixed up after layout pass
        ++S->P;
    }
    if (S->P == 0) return false;
    // nearly-stable absent phases from the grid at lam0
    if (grid != nullptr) {
        for (int ph = 0; ph < grid->num_mappable_phases_in_grid
                          && S->P < SS_MAX_CANDS; ++ph) {
            const PhaseRecord* pr = &phase_data->phase_records_array[ph];
            if (!pr->formulamole_obj || !pr->internal_cons_func) continue;
            bool present = false;
            for (int k = 0; k < S->P; ++k)
                if (S->cand[k].pr == pr) present = true;
            if (present) continue;
            int lo = grid->phase_grid_indices_start[ph];
            int hi = grid->phase_grid_indices_stop[ph];
            double best_df = -1e300; int best_j = -1;
            for (int j = lo; j < hi; ++j) {
                double df = -grid->GM_ptr[j];
                for (int c = 0; c < ncomp; ++c)
                    df += grid->X_ptr[(long long)j * grid->num_components_stride_X + c] * lam0[c];
                if (df > best_df) { best_df = df; best_j = j; }
            }
            if (best_j >= 0 && best_df > -0.2 * S->RT) {
                SSCand* ck = &S->cand[S->P];
                ck->pr = pr; ck->pd = pr->phase_dof;
                ck->nic = pr->num_internal_cons; ck->keep = true;
                rec_idx_out[S->P] = ph;
                z[ncomp + S->P] = 1e-6;
                ck->iy = -(best_j + 1);   // TEMP: negative = grid row source
                ++S->P;
            }
        }
    }
    // final variable layout + u/nu seeding
    var = ncomp + S->P;
    for (int k = 0; k < S->P; ++k) S->cand[k].inu = 0;  // second pass below
    for (int k = 0; k < S->P; ++k) {
        SSCand* ck = &S->cand[k];
        int src = ck->iy;
        ck->iy = var; var += ck->pd;
        for (int j = 0; j < ck->pd; ++j) {
            double yv;
            if (src >= 0)
                yv = ipd_row[off_sf + src * MAX_DOF_PER_PHASE + j];
            else
                yv = grid->Y_ptr[(long long)(-src - 1) * grid->phase_dof_stride_Y + j];
            if (!(yv > 1e-12)) yv = 1e-12;
            if (yv > 1.0) yv = 1.0;
            z[ck->iy + j] = log(yv);
        }
    }
    for (int k = 0; k < S->P; ++k) {
        S->cand[k].inu = var; var += S->cand[k].nic;
        for (int j = 0; j < S->cand[k].nic; ++j) z[S->cand[k].inu + j] = 0.0;
    }
    S->nvar = var;
    if (S->nvar > SS_NVAR) return false;
    for (int c = 0; c < ncomp; ++c) z[c] = lam0[c] / S->RT;
    return true;
}

// nu warm start: per-sublattice mean of (dG/dy - M lam), in RT units
__device__ static void ss_init_nu(const SSSys* S, double* z, SSScratch* W)
{
    for (int k = 0; k < S->P; ++k) {
        const SSCand* ck = &S->cand[k];
        for (int sv = 0; sv < S->nsv; ++sv) W->dof[sv] = S->statevars[sv];
        for (int j = 0; j < ck->pd; ++j) W->dof[S->nsv + j] = exp(z[ck->iy + j]);
        if (ck->pr->formulafused) ck->pr->formulafused(W->eg, W->hess, W->dof);
        else { W->eg[0] = ck->pr->formulaobj(W->dof);
               ck->pr->formulagrad(&W->eg[1], W->dof); }
        ck->pr->formulamole_grad(W->mjac, W->dof);
        ck->pr->internal_cons_jac(W->icjac, W->dof);
        const int mcols = 1 + ck->pd;
        for (int sI = 0; sI < ck->nic; ++sI) {
            double acc = 0.0; int cnt = 0;
            for (int j = 0; j < ck->pd; ++j) {
                if (W->icjac[sI * mcols + 1 + j] > 0.5) {
                    double ml = 0.0;
                    for (int c = 0; c < S->ncomp; ++c)
                        ml += (z[c] * S->RT) * W->mjac[c * mcols + 1 + j];
                    acc += W->eg[2 + j] - ml;
                    ++cnt;
                }
            }
            z[ck->inu + sI] = cnt ? (acc / cnt) / S->RT : 0.0;
        }
    }
}

extern "C" __global__ void pycgpu_ss_pass(
    const double* condition_args_doubles, int condition_stride,
    double* results_flat, int results_stride,
    const double* initial_phase_data, int ipd_stride,
    int num_unique_records, const void* grid_raw,
    const int* grid_block_indices, long long grid_block_stride_bytes,
    const double* b_all, int ncomp, int nsv, int num_conditions,
    double* work_base, long long work_stride, int* ipiv_base, int ipiv_stride,
    int dbg)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_conditions) return;
#define SS_BAIL(code) do { if (dbg && tid < 4) \
    printf("[SS] tid %d bail: %s\n", tid, code); return; } while (0)
    double* w = &work_base[(long long)tid * work_stride];
    // carve the per-thread workspace
    SSScratch W;
    long long o = 0;
    double* z      = &w[o]; o += SS_NVAR;
    double* z_best = &w[o]; o += SS_NVAR;
    W.F  = &w[o]; o += SS_NVAR;
    W.Ft = &w[o]; o += SS_NVAR;
    W.dz = &w[o]; o += SS_NVAR;
    W.zt = &w[o]; o += SS_NVAR;
    W.dof = &w[o]; o += MAX_STATEVARS + MAX_DOF_PER_PHASE;
    W.eg = &w[o]; o += 2 + MAX_DOF_PER_PHASE;
    W.hess = &w[o]; o += (MAX_DOF_PER_PHASE + 1) * (MAX_DOF_PER_PHASE + 1);
    W.fm = &w[o]; o += MAX_COMPONENTS;
    W.mjac = &w[o]; o += MAX_COMPONENTS * (MAX_STATEVARS + MAX_DOF_PER_PHASE);
    W.icv = &w[o]; o += MAX_INTERNAL_CONSTRAINTS;
    W.icjac = &w[o]; o += MAX_INTERNAL_CONSTRAINTS * (MAX_STATEVARS + MAX_DOF_PER_PHASE);
    W.J = &w[o]; o += (long long)SS_NVAR * SS_NVAR;
    W.A = &w[o]; o += (long long)SS_NVAR * SS_NVAR;
    W.ipiv = &ipiv_base[(long long)tid * ipiv_stride];

    DevicePhaseData pd_local;
    pd_local.phase_records_array = g_phase_records_array;
    pd_local.num_unique_phase_records = num_unique_records;
    const DevicePhaseData* phase_data = &pd_local;
    const void* my_grid_raw = grid_raw;
    if (grid_raw != nullptr && grid_block_indices != nullptr
            && grid_block_stride_bytes > 0)
        my_grid_raw = (const void*)((const char*)grid_raw +
            (long long)grid_block_indices[tid] * grid_block_stride_bytes);
    DeviceGrid grid_local;
    const DeviceGrid* grid_data =
        ss_grid_decode(my_grid_raw, num_unique_records, &grid_local)
        ? &grid_local : nullptr;
    SSSys S;
    int rec_idx[SS_MAX_CANDS];
    for (int i = 0; i < SS_NVAR; ++i) z[i] = 0.0;
    const double* cond_row = &condition_args_doubles[(long long)tid * condition_stride];
    const double* ipd_row = &initial_phase_data[(long long)tid * ipd_stride];
    const double* b_row = &b_all[(long long)tid * MAX_COMPONENTS];
    double* res_row = &results_flat[(long long)tid * results_stride];
    if (!ss_setup(&S, z, phase_data, grid_data, ipd_row, cond_row, b_row,
                  ncomp, nsv, rec_idx))
        SS_BAIL("setup");             // flag stays set -> faithful pass 3
    // add the PASS-1 partial state's compsets as candidates (basin info the
    // hull does not carry, e.g. a second same-phase compset mid-tussle)
    {
        const int MCc = MAX_COMPONENTS, MPc = MAX_PHASES, MDc = MAX_DOF_PER_PHASE;
        for (int slot = 0; slot < MPc && S.P < SS_MAX_CANDS; ++slot) {
            int pidx = (int)res_row[6 + MCc + MPc + MPc * MDc + MPc * MCc + slot];
            double npv = res_row[1 + MCc + slot];
            if (pidx < 0 || pidx >= num_unique_records) continue;
            if (!(npv > 1e-10)) continue;
            const PhaseRecord* pr = &g_phase_records_array[pidx];
            if (!pr->formulamole_obj || !pr->internal_cons_func) continue;
            const double* yrow = &res_row[6 + MCc + MPc + slot * MDc];
            bool finite_ok = true;
            for (int j = 0; j < pr->phase_dof; ++j)
                if (!(yrow[j] == yrow[j])) finite_ok = false;
            if (!finite_ok) continue;
            bool dup = false;
            for (int k = 0; k < S.P; ++k) {
                if (S.cand[k].pr != pr) continue;
                double dmax = 0.0;
                for (int j = 0; j < pr->phase_dof; ++j) {
                    double yk = exp(z[S.cand[k].iy + j]);
                    double d = fabs(yk - yrow[j]);
                    if (d > dmax) dmax = d;
                }
                if (dmax < 0.08) { dup = true; break; }
            }
            if (dup) continue;
            SSCand* ck = &S.cand[S.P];
            ck->pr = pr; ck->pd = pr->phase_dof;
            ck->nic = pr->num_internal_cons; ck->keep = true;
            rec_idx[S.P] = pidx;
            ck->iy = S.nvar; S.nvar += ck->pd;
            ck->inu = S.nvar; S.nvar += ck->nic;
            if (S.nvar > SS_NVAR) { S.nvar = ck->iy; break; }
            z[ncomp + S.P] = 1e-3;
            for (int j = 0; j < ck->pd; ++j) {
                double yv = yrow[j];
                if (!(yv > 1e-12)) yv = 1e-12;
                if (yv > 1.0) yv = 1.0;
                z[ck->iy + j] = log(yv);
            }
            for (int j = 0; j < ck->nic; ++j) z[ck->inu + j] = 0.0;
            ++S.P;
        }
    }
    ss_init_nu(&S, z, &W);
    if (dbg && tid == 0) {
        // FD Jacobian self-check at the initial point (debug only)
        double nf0 = ss_residual(&S, z, W.F, W.J, 1e-2, false, W.dof, W.eg,
                                 W.hess, W.fm, W.mjac, W.icv, W.icjac);
        printf("[SS] tid 0 initial nf2=%.3e (mu=1e-2)\n", nf0);
        for (int j = 0; j < S.nvar; j += (S.nvar / 12) + 1) {
            double h = 1e-7;
            for (int i = 0; i < S.nvar; ++i) W.zt[i] = z[i];
            W.zt[j] += h;
            ss_residual(&S, W.zt, W.Ft, nullptr, 1e-2, false, W.dof, W.eg,
                        W.hess, W.fm, W.mjac, W.icv, W.icjac);
            double worst = 0.0; int wrow = -1;
            for (int i = 0; i < S.nvar; ++i) {
                double fd = (W.Ft[i] - W.F[i]) / h;
                double d = fabs(fd - W.J[(long long)i * S.nvar + j]);
                double sc = fabs(fd) > 1.0 ? fabs(fd) : 1.0;
                if (d / sc > worst) { worst = d / sc; wrow = i; }
            }
            if (worst > 1e-4) {
                double fd = 0.0;
                for (int i2 = 0; i2 < S.nvar; ++i2) W.zt[i2] = z[i2];
                W.zt[j] += h;
                ss_residual(&S, W.zt, W.Ft, nullptr, 1e-2, false, W.dof, W.eg,
                            W.hess, W.fm, W.mjac, W.icv, W.icjac);
                fd = (W.Ft[wrow] - W.F[wrow]) / h;
                printf("[SS] J-CHECK col %d worst rel err %.2e at row %d "
                       "(fd=%.5e J=%.5e)\n", j, worst, wrow, fd,
                       W.J[(long long)wrow * S.nvar + j]);
            }
        }
    }
    double nf2fb = ss_run(&S, z, false, 150, 1e-18, &W);
    if (dbg && tid < 2)
        printf("[SS] tid %d P=%d nvar=%d fb-final nf2=%.3e\n", tid, S.P, S.nvar, nf2fb);
    double gm = 0.0;
    if (!ss_polish(&S, z, &W, z_best, &gm, dbg, tid)) SS_BAIL("polish");
    // ---- acceptance gates ----
    double bal[MAX_COMPONENTS];
    for (int c = 0; c < S.ncomp; ++c) bal[c] = -S.b[c];
    double lamJ[MAX_COMPONENTS];
    for (int c = 0; c < S.ncomp; ++c) lamJ[c] = z_best[c] * S.RT;
    for (int k = 0; k < S.P; ++k) {
        if (!S.cand[k].keep) continue;
        if (z_best[S.ncomp + k] < -1e-8) SS_BAIL("neg-amount");
        for (int sv = 0; sv < S.nsv; ++sv) W.dof[sv] = S.statevars[sv];
        for (int j = 0; j < S.cand[k].pd; ++j)
            W.dof[S.nsv + j] = exp(z_best[S.cand[k].iy + j]);
        S.cand[k].pr->formulamole_obj(W.fm, W.dof);
        for (int c = 0; c < S.ncomp; ++c)
            bal[c] += z_best[S.ncomp + k] * W.fm[c];
    }
    for (int c = 0; c < S.ncomp; ++c)
        if (fabs(bal[c]) > 1e-6) SS_BAIL("balance");
    {
        // the hull start is a FEASIBLE mixture, so true GM <= mu_hull . b;
        // anything above it is a wrong basin regardless of grid-df cleanliness
        const int off_mu_i = 2 * MAX_PHASES + MAX_PHASES * MAX_DOF_PER_PHASE
                             + MAX_PHASES * MAX_COMPONENTS;
        double gm_hull = 0.0;
        for (int c = 0; c < S.ncomp; ++c)
            gm_hull += ipd_row[off_mu_i + c] * S.b[c];
        if (gm > gm_hull + 1.0) SS_BAIL("hull-gm");
    }
    if (grid_data != nullptr) {                        // grid-df gate
        for (int j = 0; j < grid_data->num_grid_points_total; ++j) {
            double df = -grid_data->GM_ptr[j];
            for (int c = 0; c < S.ncomp; ++c)
                df += grid_data->X_ptr[(long long)j * grid_data->num_components_stride_X + c] * lamJ[c];
            if (df > 1e-3) SS_BAIL("grid-df");
        }
    }
    // ---- accepted: write the result row ----
    double* res = res_row;
    const int MC = MAX_COMPONENTS, MP = MAX_PHASES, MD = MAX_DOF_PER_PHASE;
    res[0] = gm;
    for (int c = 0; c < MC; ++c) res[1 + c] = (c < S.ncomp) ? lamJ[c] : 0.0;
    for (int i = 0; i < MP; ++i) res[1 + MC + i] = 0.0;
    for (int i = 0; i < MP * MD; ++i) res[6 + MC + MP + i] = 0.0;
    for (int i = 0; i < MP * MC; ++i) res[6 + MC + MP + MP * MD + i] = 0.0;
    for (int i = 0; i < MP; ++i)
        res[6 + MC + MP + MP * MD + MP * MC + i] = -1.0;
    int slot = 0;
    for (int k = 0; k < S.P && slot < MP; ++k) {
        if (!S.cand[k].keep || z_best[S.ncomp + k] <= 1e-9) continue;
        for (int sv = 0; sv < S.nsv; ++sv) W.dof[sv] = S.statevars[sv];
        for (int j = 0; j < S.cand[k].pd; ++j)
            W.dof[S.nsv + j] = exp(z_best[S.cand[k].iy + j]);
        S.cand[k].pr->formulamole_obj(W.fm, W.dof);
        double at = 0.0;
        for (int c = 0; c < S.ncomp; ++c) at += W.fm[c];
        if (at < 1e-12) at = 1.0;
        res[1 + MC + slot] = z_best[S.ncomp + k] * at;           // NP (moles)
        for (int j = 0; j < S.cand[k].pd && j < MD; ++j)
            res[6 + MC + MP + slot * MD + j] = W.dof[S.nsv + j]; // Y
        for (int c = 0; c < S.ncomp && c < MC; ++c)
            res[6 + MC + MP + MP * MD + slot * MC + c] = W.fm[c] / at; // X
        res[6 + MC + MP + MP * MD + MP * MC + slot] = (double)rec_idx[k];
        ++slot;
    }
    res[1 + MC + MP] = 1.0;              // converged
    res[2 + MC + MP] = (double)slot;     // num_stable
    res[3 + MC + MP] = S.statevars[S.nsv - 1];
    res[4 + MC + MP] = (S.nsv >= 2) ? S.statevars[S.nsv - 2] : 0.0;
    res[5 + MC + MP] = 0.0;              // CLEAR the cap flag -> accepted
}

#endif // SEMISMOOTH_H
