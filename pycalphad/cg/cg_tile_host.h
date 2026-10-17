// A tile solved by cg_tile.h's steps on the host, one condition after the other within each step (the order a
// GPU's per-condition kernels would not need): the reference for the GPU driver, and a check against the Cython
// TileSolver. Buffers grow as needed.
#pragma once
#include <vector>
#include <algorithm>

struct CgHostTile {
    std::vector<double> Py, Px, PG, Oy, Ox, OG, j_y0, j_y, j_x, j_G, j_tpd;
    std::vector<int> Pph, Pnext, head, Oph, nown, j_cond, j_phase, j_keep, j_conv, cp_phase, cp_start, cp_end;
    std::vector<long long> Pcode, Ocode;
    std::vector<long> own_off;

    void store_capacity(CgTile& T, int need) {
        if (need <= T.cap) return;
        int cap = std::max(need, 2 * T.cap);
        Py.resize((size_t)cap * T.Dm); Px.resize((size_t)cap * T.C); PG.resize(cap); Pph.resize(cap); Pnext.resize(cap);
        Pcode.resize(cap);
        T.Py = Py.data(); T.Px = Px.data(); T.PG = PG.data(); T.Pph = Pph.data(); T.Pnext = Pnext.data();
        T.Pcode = Pcode.data(); T.cap = cap;
    }
    void job_capacity(CgTile& T, int need) {
        if (need <= T.jcap) return;
        int cap = std::max(need, 2 * T.jcap);
        j_y0.resize((size_t)cap * T.Dm); j_y.resize((size_t)cap * T.Dm); j_x.resize((size_t)cap * T.C);
        j_G.resize(cap); j_tpd.resize(cap); j_cond.resize(cap); j_phase.resize(cap); j_keep.resize(cap); j_conv.resize(cap);
        T.j_y0 = j_y0.data(); T.j_y = j_y.data(); T.j_x = j_x.data(); T.j_G = j_G.data(); T.j_tpd = j_tpd.data();
        T.j_cond = j_cond.data(); T.j_phase = j_phase.data(); T.j_keep = j_keep.data(); T.j_conv = j_conv.data();
        T.jcap = cap;
    }
    void cp_capacity(CgTile& T, int need) {
        if (need <= T.cpcap) return;
        int cap = std::max(need, 2 * T.cpcap);
        cp_phase.resize(cap); cp_start.resize(cap); cp_end.resize(cap);
        T.cp_phase = cp_phase.data(); T.cp_start = cp_start.data(); T.cp_end = cp_end.data(); T.cpcap = cap;
    }
};

extern "C" int cg_tile_solve_host(
        int B, int C, int K, int Dm, int nph, int ngauge, int max_late, int use_polish, int max_rounds,
        double RT, double prune, double prune_late, double skip_margin, double tol, double seed_prune,
        const int* di, const double* dd, const int* ip, const double* dp, const double* sv, const double* gauge,
        int npts0, const double* Py0, const double* Px0, const double* PG0, const int* Pph0, const long long* Pcode0,
        const int* Pnext0, const int* head0,
        const double* R, const double* b, const double* mf, const int* fixed,
        const double* seedY, const int* seed_off, int use_seeds, int own_first, int own_other,
        const int* stage, const int* init_from,
        const int* basis0, const unsigned char* has_last0, const double* last_tpd0, const double* last_mu0,
        int* status, int* nout, int* phs, double* amt, double* X, double* Y, double* G, double* mu_out,
        long long* counters) {
    CgHostTile H;
    CgTile T{};
    T.B = B; T.C = C; T.K = K; T.Dm = Dm; T.nph = nph; T.ngauge = ngauge; T.max_late = max_late;
    T.use_polish = use_polish; T.RT = RT; T.prune = prune; T.prune_late = prune_late; T.skip_margin = skip_margin;
    T.tol = tol; T.seed_prune = seed_prune;
    T.di = di; T.dd = dd; T.ip = ip; T.dp = dp; T.sv = sv; T.gauge = gauge;
    T.cap = 0; H.store_capacity(T, std::max(2 * npts0, 1024));
    T.nshared = npts0;
    std::copy(Py0, Py0 + (size_t)npts0 * Dm, H.Py.begin()); std::copy(Px0, Px0 + (size_t)npts0 * C, H.Px.begin());
    std::copy(PG0, PG0 + npts0, H.PG.begin()); std::copy(Pph0, Pph0 + npts0, H.Pph.begin());
    std::copy(Pcode0, Pcode0 + npts0, H.Pcode.begin()); std::copy(Pnext0, Pnext0 + npts0, H.Pnext.begin());
    H.head.assign(head0, head0 + nph); T.head = H.head.data();
    // own slabs: the first condition's holds its seeds' points
    H.own_off.resize(B + 1); H.own_off[0] = 0;
    for (int t = 0; t < B; ++t) H.own_off[t + 1] = H.own_off[t] + (t == 0 ? own_first : own_other);
    long nown_rows = H.own_off[B];
    H.Oy.resize((size_t)nown_rows * Dm); H.Ox.resize((size_t)nown_rows * C); H.OG.resize(nown_rows);
    H.Oph.resize(nown_rows); H.Ocode.resize(nown_rows); H.nown.assign(B, 0);
    T.Oy = H.Oy.data(); T.Ox = H.Ox.data(); T.OG = H.OG.data(); T.Oph = H.Oph.data(); T.Ocode = H.Ocode.data();
    T.nown = H.nown.data(); T.own_off = H.own_off.data();
    T.R = R; T.b = b; T.mf = mf; T.fixed = fixed; T.seedY = seedY; T.seed_off = seed_off;
    std::vector<int> basis((size_t)B * K), pol_ph((size_t)B * C), pol_na(B, 0), cert_kind(B, 0), cert_phase(B, 0),
        aug(B, 0), kind(B, 0);
    std::vector<double> xB((size_t)B * K), mu((size_t)B * C), last_tpd((size_t)B * nph), last_mu((size_t)B * nph * C),
        pol_y((size_t)B * C * Dm), pol_x((size_t)B * C * C), pol_n((size_t)B * C), pol_mu((size_t)B * C),
        pol_G((size_t)B * C), cert_tpd(B), ycert((size_t)B * Dm);
    std::vector<unsigned char> has_last((size_t)B * nph), nopol(B, 0), polished(B, 0), skipped(B, 0), added(B, 0),
        done(B, 0);
    std::vector<long long> lp_iters(B, 0), priced(B, 0);
    T.basis = basis.data(); T.xB = xB.data(); T.mu = mu.data(); T.has_last = has_last.data(); T.nopol = nopol.data();
    T.polished = polished.data(); T.skipped = skipped.data(); T.added = added.data(); T.done = done.data();
    T.last_tpd = last_tpd.data(); T.last_mu = last_mu.data(); T.pol_y = pol_y.data(); T.pol_x = pol_x.data();
    T.pol_n = pol_n.data(); T.pol_mu = pol_mu.data(); T.pol_G = pol_G.data(); T.pol_ph = pol_ph.data();
    T.pol_na = pol_na.data(); T.cert_kind = cert_kind.data(); T.cert_phase = cert_phase.data(); T.aug = aug.data();
    T.kind = kind.data(); T.cert_tpd = cert_tpd.data(); T.ycert = ycert.data(); T.lp_iters = lp_iters.data();
    T.priced = priced.data();
    std::vector<int> cp_first(B, 0), cp_last(B, 0);
    T.cp_first = cp_first.data(); T.cp_last = cp_last.data();
    T.jcap = 0; T.cpcap = 0; H.job_capacity(T, 1024); H.cp_capacity(T, 256);
    T.status = status; T.nout = nout; T.phs = phs; T.amt = amt; T.X = X; T.Y = Y; T.G = G; T.mu_out = mu_out;
    // the first condition starts from the solver's state; the others from their stage's starting condition
    std::copy(basis0, basis0 + K, basis.begin());
    std::copy(has_last0, has_last0 + nph, has_last.begin());
    std::copy(last_tpd0, last_tpd0 + nph, last_tpd.begin());
    std::copy(last_mu0, last_mu0 + (size_t)nph * C, last_mu.begin());
    static thread_local CgWork W;
    static thread_local CgPolishWork PW;
    long long rounds_total = 0, subrounds = 0, jobs_total = 0, own_full = 0;
    int smax = 0;
    for (int t = 0; t < B; ++t) smax = std::max(smax, stage[t]);
    std::vector<int> members;
    for (int s = 0; s <= smax; ++s) {
        members.clear();
        for (int t = 0; t < B; ++t) if (stage[t] == s) members.push_back(t);
        if (members.empty()) continue;
        if (s > 0)
            for (int t : members) {
                int f = init_from[t];
                std::copy(&basis[(size_t)f * K], &basis[(size_t)f * K] + K, &basis[(size_t)t * K]);
                std::copy(&mu[(size_t)f * C], &mu[(size_t)f * C] + C, &mu[(size_t)t * C]);
                std::copy(&has_last[(size_t)f * nph], &has_last[(size_t)f * nph] + nph, &has_last[(size_t)t * nph]);
                std::copy(&last_tpd[(size_t)f * nph], &last_tpd[(size_t)f * nph] + nph, &last_tpd[(size_t)t * nph]);
                std::copy(&last_mu[(size_t)f * nph * C], &last_mu[(size_t)f * nph * C] + (size_t)nph * C,
                          &last_mu[(size_t)t * nph * C]);
                nopol[t] = 0;
            }
        for (int rounds = 1; rounds <= max_rounds; ++rounds) {
            rounds_total++;
            int nact = 0;
            T.nj = 0; T.ncp = 0;
            int seeds_now = use_seeds && s == 0 && rounds == 1;
            for (int t : members) {
                if (done[t]) continue;
                nact++;
                H.job_capacity(T, T.nj + cg_tile_job_bound(T, seeds_now)); H.cp_capacity(T, T.ncp + nph + 1);
                cg_step_start(T, t, rounds, seeds_now, W, PW);
            }
            while (T.nj > 0) {
                subrounds++; jobs_total += T.nj;
                for (int j = 0; j < T.nj; ++j)
                    cg_price_job(j, T.j_phase, T.j_cond, T.mu, T.j_y0, Dm, T.j_y, T.j_x, T.j_G, T.j_tpd, T.j_conv,
                                 nullptr, di, dd, ip, dp, C, RT, sv, W);
                for (int t : members) cg_step_consume(T, t, rounds, max_rounds);
                T.nj = 0; T.ncp = 0;
                for (int t : members) {
                    if (T.kind[t] != CG_NEED_AUG && T.kind[t] != CG_NEED_FINAL) continue;
                    H.job_capacity(T, T.nj + cg_tile_job_bound(T, 0)); H.cp_capacity(T, T.ncp + nph + 1);
                    cg_step_next(T, t, rounds, max_rounds, W, PW);
                }
            }
            long want = T.nshared;
            for (int t : members) { want += T.nown[t]; if (T.own_off[t] + T.nown[t] >= T.own_off[t + 1]) own_full++; }
            H.store_capacity(T, (int)want);
            if (cg_tile_merge(T, members.data(), (int)members.size()) < 0) return -1;
            int all = 1;
            for (int t : members) if (!done[t]) all = 0;
            if (nact == 0 || all) break;
        }
    }
    long long pr = 0, lp = 0;
    for (int t = 0; t < B; ++t) { pr += priced[t]; lp += lp_iters[t]; }
    counters[0] = pr; counters[1] = lp; counters[2] = rounds_total; counters[3] = subrounds; counters[4] = jobs_total;
    counters[5] = own_full; counters[6] = T.nshared;
    return 0;
}
