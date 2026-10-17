// A tile on a GPU: cg_tile.h's per-condition steps as kernels, one thread per condition of a list (conds), and
// the round's end in parallel; the pricing between the steps is cg_price_kernel (device.py). The driver
// (pycalphad.cg.device_tile) keeps the tile's arrays on the device, binds them into one CgTile there
// (cg_dev_bind, generated with the driver's field table) and runs the rounds as cg_tile_host.h does. Needs
// cg_price.h, cg_round.h and cg_tile.h.
//
// - Job buffers cannot grow on a device: each condition first counts its jobs (cg_gen_jobs without buffers), the
//   driver lays the ranges out in the list's order (a prefix sum) and the conditions fill them; condition t's
//   pricing groups (cp_*) take the nph slots from t * nph. Each thread works on its own copy of the CgTile (the
//   job and group counters are its own; the arrays are shared).
// - The steps' work arrays (CgWork, CgPolishWork) are one per condition in device memory: on the stack they would
//   need a stack limit that the device reserves for every resident thread.
// - Every kernel's first argument is its number of threads (the launchers on the host loop over them), except
//   cg_dev_lp's: its number of blocks (of CG_LANES threads; on the host one thread takes the lanes in turn).
#pragma once

extern "C" __global__ void cg_dev_sizes(int n, long long* out) {
    out[0] = sizeof(CgTile); out[1] = sizeof(CgWork); out[2] = sizeof(CgPolishWork);
}

// the conditions of a stage start from the state of their starting conditions (solved in earlier stages)
extern "C" __global__ void cg_dev_stage_init(int n, const CgTile* T, const int* conds, const int* init_from) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const CgTile& L = *T;
    long t = conds[i], f = init_from[t];
    for (int k = 0; k < L.K; ++k) L.basis[t * L.K + k] = L.basis[f * L.K + k];
    for (int c = 0; c < L.C; ++c) L.mu[t * L.C + c] = L.mu[f * L.C + c];
    for (int p = 0; p < L.nph; ++p) {
        L.has_last[t * L.nph + p] = L.has_last[f * L.nph + p];
        L.last_tpd[t * L.nph + p] = L.last_tpd[f * L.nph + p];
    }
    for (int p = 0; p < L.nph * L.C; ++p) L.last_mu[t * L.nph * L.C + p] = L.last_mu[f * L.nph * L.C + p];
    L.nopol[t] = 0;
}

// a round's start, the LP (cg_tile_lp): a block of CG_LANES threads per condition (n blocks), which split the scans
// over the store, each on its own copies of the condition's basis, amounts and potentials (lane 0 writes them
// back); st[i]: the LP's status
extern "C" __global__ void cg_dev_lp(int n, const CgTile* T, const int* conds, int* st) {
    __shared__ double best[CG_LANES];
    __shared__ int arg[CG_LANES];
    int i = blockIdx.x;
    if (i >= n) return;
    const CgTile& L = *T;
    const int K = L.K, C = L.C;
    long t = conds[i];
    int basis[CG_CMAX];
    double xB[CG_CMAX], mu[CG_CMAX];
    long long iters = 0;
    for (int k = 0; k < K; ++k) { basis[k] = L.basis[t * K + k]; xB[k] = L.xB[t * K + k]; }
    for (int c = 0; c < C; ++c) mu[c] = L.mu[t * C + c];
    CgLanes G{(int)threadIdx.x, (int)blockDim.x, CG_LANES, best, arg};
    int s = cg_tile_lp(L, t, basis, xB, mu, &iters, G);
    if (threadIdx.x != 0) return;
    for (int k = 0; k < K; ++k) { L.basis[t * K + k] = basis[k]; L.xB[t * K + k] = xB[k]; }
    for (int c = 0; c < C; ++c) L.mu[t * C + c] = mu[c];
    L.lp_iters[t] += iters;
    st[i] = s;
}

// the rest of a round's start (cg_step_lp_polish), one thread per condition; want[i]: the LP's status (cg_dev_lp),
// then whether condition conds[i] lists pricing jobs now
extern "C" __global__ void cg_dev_lp_polish(int n, const CgTile* T, const int* conds, CgWork* work,
                                            CgPolishWork* pwork, int* want) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    CgTile L = *T;
    int t = conds[i];
    want[i] = cg_step_lp_polish(L, t, want[i], work[t], pwork[t]);
}

// the job results (cg_step_consume)
extern "C" __global__ void cg_dev_consume(int n, const CgTile* T, const int* conds, int rounds, int max_rounds) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    CgTile L = *T;
    cg_step_consume(L, conds[i], rounds, max_rounds);
}

// the next step of a sub-round (cg_step_repolish); want as cg_dev_lp
extern "C" __global__ void cg_dev_repolish(int n, const CgTile* T, const int* conds, int rounds, int max_rounds,
                                           CgWork* work, CgPolishWork* pwork, int* want) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    CgTile L = *T;
    int t = conds[i];
    want[i] = cg_step_repolish(L, t, rounds, max_rounds, work[t], pwork[t]);
}

// the pricing jobs of the conditions that want them: counted into count[i] (fill 0), or listed in their ranges
// start[i] .. start[i] + count[i] - 1 (fill 1)
extern "C" __global__ void cg_dev_jobs(int n, const CgTile* T, const int* conds, const int* want, int rounds,
                                       int use_seeds, CgWork* work, int fill, const int* start, int* count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!want[i]) { if (!fill) count[i] = 0; return; }
    CgTile L = *T;
    int t = conds[i];
    L.ncp = t * L.nph;
    if (fill) {
        L.nj = start[i]; L.jcap = start[i] + count[i];
    } else {
        L.nj = 0; L.j_cond = nullptr;
    }
    cg_gen_jobs(L, t, rounds, use_seeds, work[t]);
    if (!fill) count[i] = L.nj;
}

// ---- the end of a round (cg_tile_merge's result, in parallel where it is sequential)
//
// A condition's own points never duplicate the store's or each other (cg_add_point checked them, and the store
// does not change during a round), so a point is kept iff no kept point before it in merge order duplicates it.
// The driver lists the members' own points in merge order (rows[o]: slab rows) and sorted by phase, then o
// (order[s], with pos[order[s]] = s and group[s] the first position of the phase).

CG_DEVICE static inline int cg_dev_same(const double* a, const double* b, int D) {
    double m = 0.0;
    for (int j = 0; j < D; ++j) { double d = a[j] - b[j]; d = d < 0 ? -d : d; if (d > m) m = d; }
    return m < 1e-7;
}

// the first point before o (in merge order) that duplicates it: first[o], or -1
extern "C" __global__ void cg_dev_merge_mark(int n, const CgTile* T, const long* rows, const int* order,
                                             const int* group, int* first) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n) return;
    const CgTile& L = *T;
    int o = order[s];
    long r = rows[o];
    int D = L.di[L.Oph[r] * CGI_COUNT + CGI_D];
    first[o] = -1;
    for (int s2 = group[s]; s2 < s; ++s2)
        if (cg_dev_same(L.Oy + rows[order[s2]] * L.Dm, L.Oy + r * L.Dm, D)) { first[o] = order[s2]; break; }
}

// (one thread) which points are kept, their rows in the store (dest[o], or -1) and the per-phase lists; the store's
// new size in nshared_out
extern "C" __global__ void cg_dev_merge_resolve(int n, CgTile* T, int npts, const long* rows, const int* order,
                                                const int* pos, const int* first, int* dest, int* nshared_out) {
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
    for (int o = 0; o < npts; ++o) {
        int f = first[o], keep = 1;
        long r = rows[o];
        int ph = T->Oph[r];
        if (f >= 0 && dest[f] >= 0) keep = 0;
        else if (f >= 0) {   // (its first duplicate was dropped: a later one may be kept)
            int D = T->di[ph * CGI_COUNT + CGI_D];
            for (int s2 = pos[f] + 1; s2 < pos[o] && keep; ++s2) {
                int q = order[s2];
                if (dest[q] >= 0 && cg_dev_same(T->Oy + rows[q] * T->Dm, T->Oy + r * T->Dm, D)) keep = 0;
            }
        }
        if (!keep) { dest[o] = -1; continue; }
        int p = T->nshared++;
        dest[o] = p;
        T->Pnext[p] = T->head[ph]; T->head[ph] = p;
    }
    *nshared_out = T->nshared;
}

// the kept points into the store
extern "C" __global__ void cg_dev_merge_copy(int n, const CgTile* T, const long* rows, const int* dest) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n || dest[o] < 0) return;
    const CgTile& L = *T;
    long r = rows[o], p = dest[o];
    for (int j = 0; j < L.Dm; ++j) L.Py[p * L.Dm + j] = L.Oy[r * L.Dm + j];
    for (int c = 0; c < L.C; ++c) L.Px[p * L.C + c] = L.Ox[r * L.C + c];
    L.PG[p] = L.OG[r]; L.Pph[p] = L.Oph[r]; L.Pcode[p] = L.Ocode[r];
}
